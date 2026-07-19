use crate::field_features::{
  CHANNELS, GLOBAL_FEATURES, HISTORY_CHANNELS, field_features_len, field_features_to_vec, global_to_vec,
};
use crate::model::Model;
use either::Either;
use ndarray::{Array, ArrayView2, s};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, Hash, NonZeroPos, Pos},
  player::Player,
};
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Gamma, Open01, StandardNormal};
use std::cell::LazyCell;
use std::collections::VecDeque;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem;
use std::{iter, iter::Sum};

/// Pass-through hasher for the transposition table.
///
/// The keys are Zobrist hashes, which are already uniformly distributed 64-bit
/// values, so there is nothing to gain from running them through a general
/// purpose hash function.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
  fn finish(&self) -> u64 {
    self.0
  }

  fn write(&mut self, bytes: &[u8]) {
    for &byte in bytes {
      self.0 = (self.0 << 8) | byte as u64;
    }
  }

  fn write_u64(&mut self, i: u64) {
    self.0 = i;
  }
}

type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<IdentityHasher>>;

/// Radius of the square local pattern used as part of a [`BiasKey`].
///
/// KataGo keys subtree value bias buckets by, among other things, the 5x5
/// pattern surrounding the last move. A radius of 2 reproduces that 5x5 window.
const BIAS_PATTERN_RADIUS: i32 = 2;
/// Side length of the local pattern window (`2 * radius + 1`).
const BIAS_PATTERN_SIDE: usize = (2 * BIAS_PATTERN_RADIUS + 1) as usize;
/// Number of cells in the local pattern window.
const BIAS_PATTERN_CELLS: usize = BIAS_PATTERN_SIDE * BIAS_PATTERN_SIDE;

/// Classifies a board cell relative to the player who made the last move, used
/// to build the local pattern of a [`BiasKey`].
///
/// The classification is relative to the mover (own / opponent) rather than
/// absolute (red / black), so that the same tactic played by either player
/// shares a bucket - this mirrors the player-relative features the net itself
/// sees and keeps the perspective of the bucketed bias consistent.
fn classify_cell(field: &Field, x: i32, y: i32, mover: Player) -> u8 {
  if x < 0 || y < 0 || x >= field.width() as i32 || y >= field.height() as i32 {
    // Off-board / border.
    return 0;
  }
  let cell = field.cell(field.to_pos(x as u32, y as u32));
  match cell.get_owner() {
    None => 1, // empty / neutral
    Some(p) if p == mover => 2, // owned by the mover
    Some(_) => 3, // owned by the opponent
  }
}

/// Identifies a subtree value bias bucket.
///
/// Nodes are bucketed by the local context of the last move so that the
/// observed bias of the neural net for a given tactic can be shared across the
/// many places that tactic appears in the search tree. The key mirrors
/// KataGo's: the location of the last move, the location of the move before it,
/// and the local pattern surrounding the last move (relative to the mover).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BiasKey {
  /// Location of the last move.
  last: Pos,
  /// Location of the move before the last move (`0` if there was none).
  prev: Pos,
  /// Local pattern surrounding the last move, relative to the mover.
  pattern: [u8; BIAS_PATTERN_CELLS],
}

/// Accumulated observed bias for a single bucket.
///
/// `delta_sum` is `sum_n (ChildrenUtility(n) - NNUtility(n)) * ChildVisits(n)^alpha`
/// and `weight_sum` is `sum_n ChildVisits(n)^alpha`, both summed over the nodes
/// `n` currently in the bucket. The bucket's observed bias is their ratio.
#[derive(Clone, PartialEq, Debug)]
pub struct BiasEntry<N: Float> {
  pub delta_sum: N,
  pub weight_sum: N,
}

/// Represents an edge from a parent to a child in the graph.
#[derive(Clone, PartialEq, Debug)]
pub struct Edge<N: Float> {
  pub pos: Pos,
  /// Zobrist hash of the child state
  pub hash: Hash,
  /// Number of times this specific edge was traversed (N(n, a))
  pub visits: u64,
  /// The raw policy prediction P(a)
  pub prior: N,
  /// Virtual losses to reduce parallelization conflicts
  pub virtual_losses: u64,
}

/// Represents a single state in the game graph.
#[derive(Clone, PartialEq, Debug)]
pub struct Node<N: Float> {
  /// N(n): Total visits to this node.
  /// Note: In MCGS, this is effectively 1 + sum(edge.visits).
  pub visits: u64,
  /// Q(n): Expected utility.
  /// Calculated recursively: (U(n) + sum(edge.visits * child.Q)) / N(n)
  pub value: N,
  /// U(n): Raw utility from the neural net for this state (NNUtility). The
  /// bias-corrected NodeUtility used by the MCTS recurrence is derived from this
  /// plus the bucket's observed bias; see [`Search::update_node`].
  pub raw_value: N,
  /// Q²(n): Expected squared utility, propagated recursively the same way as
  /// `value`: (U(n)² + sum(edge.visits * child.Q²)) / N(n). Squares are
  /// perspective-independent, so no sign flips are needed. Together with
  /// `value` it estimates the variance of the value for LCB move selection.
  pub value_sq: N,
  /// Edges to children.
  pub children: Vec<Edge<N>>,
  /// Subtree value bias bucket this node belongs to, if any. Computed once when
  /// the node is created and then kept constant. `None` for the root, terminal
  /// nodes, and nodes with no preceding move.
  pub bias_key: Option<BiasKey>,
  /// The node's most recent contribution to its bucket, i.e. the last values it
  /// added to [`BiasEntry::delta_sum`] and [`BiasEntry::weight_sum`]. Tracked so
  /// that recomputing the node's bias updates the bucket by the delta rather
  /// than double-counting.
  pub last_bias_delta: N,
  pub last_bias_weight: N,
}

impl<N: Float> Node<N> {
  pub fn new() -> Self {
    Node {
      visits: 0,
      value: N::zero(),
      raw_value: N::zero(),
      value_sq: N::zero(),
      children: Vec::new(),
      bias_key: None,
      last_bias_delta: N::zero(),
      last_bias_weight: N::zero(),
    }
  }
}

impl<N> Node<N>
where
  N: Float + Sum,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  pub fn apply_temperature(&mut self, temperature: N) {
    let max_ln = self.children.iter().map(|edge| edge.prior).fold(N::zero(), N::max).ln();
    let mut sum = N::zero();
    for edge in self.children.iter_mut() {
      // Numerically stable way to raise to power and normalize
      edge.prior = ((edge.prior.ln() - max_ln) / temperature).exp();
      sum = sum + edge.prior;
    }
    for edge in self.children.iter_mut() {
      edge.prior = edge.prior / sum;
    }
  }

  /// Adds shaped Dirichlet noise to the children priors.
  ///
  /// `total_concentration` is the sum of the Dirichlet alphas. Instead of spreading it
  /// uniformly across the legal moves, half of it is spread uniformly and the other
  /// half is concentrated on the moves whose (clamped) log policy is above the
  /// average - i.e. the moves that still stand out from the field. Such "blind spot"
  /// moves usually have a much higher prior than most arbitrary moves on the board even
  /// when their absolute prior is tiny, so this raises the chance that they get noised
  /// and explored.
  pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, epsilon: N, total_concentration: N) {
    if self.children.is_empty() {
      return;
    }
    let legal_count = N::from(self.children.len()).unwrap();

    // Shape the alpha distribution based on the log of the policy prior. Priors are
    // clamped at 0.01 so any sufficiently likely move is treated equally, and the small
    // additive constant avoids `ln(0)` for moves with a zero prior.
    let cap = N::from(0.01).unwrap();
    let offset = N::from(1e-20).unwrap();
    let mut alpha = self
      .children
      .iter()
      .map(|edge| (edge.prior.min(cap) + offset).ln())
      .collect::<Vec<_>>();
    let log_mean = alpha.iter().copied().sum::<N>() / legal_count;
    let mut prop_sum = N::zero();
    for a in alpha.iter_mut() {
      *a = (*a - log_mean).max(N::zero());
      prop_sum = prop_sum + *a;
    }
    let uniform = N::one() / legal_count;
    if prop_sum <= N::zero() {
      // All priors equal: fall back to symmetric Dirichlet.
      for a in alpha.iter_mut() {
        *a = uniform;
      }
    } else {
      let half = N::from(0.5).unwrap();
      for a in alpha.iter_mut() {
        *a = half * (*a / prop_sum + uniform);
      }
    }

    // Draw an independent Gamma per move with the shaped alpha and normalize to get the
    // Dirichlet sample, reusing `alpha` in place. The shaped alphas sum to 1, so they sum
    // to `total_concentration` once scaled.
    let mut dirichlet = alpha;
    let mut sum = N::zero();
    for eta in dirichlet.iter_mut() {
      let shape = *eta * total_concentration;
      *eta = if shape > N::zero() {
        Gamma::<N>::new(shape, N::one()).unwrap().sample(rng)
      } else {
        N::zero()
      };
      sum = sum + *eta;
    }
    if sum == N::zero() {
      return;
    }
    for eta in dirichlet.iter_mut() {
      *eta = *eta / sum;
    }
    for (child, eta) in self.children.iter_mut().zip(dirichlet) {
      child.prior = child.prior * (N::one() - epsilon) + epsilon * eta;
    }
  }
}

impl<N: Float> Default for Node<N> {
  fn default() -> Self {
    Self::new()
  }
}

pub fn game_result<N: Float>(field: &Field, player: Player, komi_x_2: i32) -> N {
  N::from((field.score(player) * 2 + komi_x_2).signum()).unwrap()
}

#[derive(Clone, PartialEq, Debug)]
pub struct Search<N: Float> {
  /// Index of the root node in `nodes`
  pub root_idx: usize,
  /// Arena allocation for nodes
  pub nodes: Vec<Node<N>>,
  /// Maps Zobrist hash -> index in `nodes`
  pub map: HashMap<Hash, usize>,
  /// Subtree value bias buckets, keyed by the local context of the last move.
  pub bias: std::collections::HashMap<BiasKey, BiasEntry<N>>,
  /// Whether dirichlet noise was added to the root node
  pub dirichlet_noise: bool,
  /// Whether forbid apriori bad moves
  pub forbid_bad: bool,
}

impl<N: Float> Search<N> {
  pub fn new(forbid_bad: bool) -> Self {
    let mut search = Search {
      root_idx: 0,
      nodes: Vec::new(),
      map: HashMap::default(),
      bias: std::collections::HashMap::default(),
      dirichlet_noise: false,
      forbid_bad,
    };

    // Initialize root
    search.nodes.push(Node::new());
    search
  }
}

impl<N: Float + Sum + Copy> Search<N> {
  fn add_node(&mut self, hash: Hash) -> usize {
    *self.map.entry(hash).or_insert_with(|| {
      let idx = self.nodes.len();
      let node = Node::new();
      self.nodes.push(node);
      idx
    })
  }

  /// Subtree value bias correction hyperparameters.
  ///
  /// `lambda` is the fraction of the bucket's observed bias that is mixed into a
  /// node's utility; `alpha` is the exponent applied to a node's child visit
  /// count when weighting its contribution to the bucket. These are KataGo's
  /// current defaults (its methods paper reported 0.35 and 0.8). `free_prop` is
  /// the fraction of a node's bucket contribution that is removed when the node
  /// leaves the reused search tree.
  /// Setting `lambda` to zero disables the correction entirely.
  const BIAS_LAMBDA: f64 = 0.45;
  const BIAS_ALPHA: f64 = 0.85;
  const BIAS_FREE_PROP: f64 = 0.8;

  /// Retrieves the current observed bias of a bucket, i.e.
  /// `delta_sum / weight_sum`, or zero if the bucket has too little weight.
  fn retrieve_bias(bias: &std::collections::HashMap<BiasKey, BiasEntry<N>>, key: &BiasKey) -> N {
    if let Some(entry) = bias.get(key)
      && entry.weight_sum > N::from(1e-3).unwrap()
    {
      return entry.delta_sum / entry.weight_sum;
    }
    N::zero()
  }

  /// Recomputes a node's visit count and MCTS utility from its children,
  /// applying subtree value bias correction.
  ///
  /// This is the recurrence
  /// `MCTSUtility(n) = (NodeUtility(n) + sum_c MCTSUtility(c) * Visits(c)) / (1 + sum_c Visits(c))`
  /// where `NodeUtility(n) = NNUtility(n) + lambda * ObsBias(bucket(n))`. As a
  /// side effect, the node's bucket is updated with its freshly observed error
  /// `ChildrenUtility(n) - NNUtility(n)` before that bias is retrieved back.
  fn update_node(
    map: &HashMap<Hash, usize>,
    nodes: &mut [Node<N>],
    bias: &mut std::collections::HashMap<BiasKey, BiasEntry<N>>,
    node_idx: usize,
  ) {
    let mut sum_values = N::zero();
    let mut sum_values_sq = N::zero();
    let mut sum_visits = 0;

    for edge in nodes[node_idx].children.iter() {
      // Unvisited edges contribute nothing (zero weight in both sums), so skip
      // them to avoid the hash lookup and conversion for the (often many)
      // children that have never been traversed.
      if edge.visits == 0 {
        continue;
      }
      if let Some(&child_idx) = map.get(&edge.hash) {
        let visits = N::from(edge.visits).unwrap();
        sum_values = sum_values - visits * nodes[child_idx].value;
        sum_values_sq = sum_values_sq + visits * nodes[child_idx].value_sq;
      }
      sum_visits += edge.visits;
    }

    // NodeUtility starts as the raw neural net utility and is then corrected
    // towards the observed bias of this node's bucket.
    let raw_value = nodes[node_idx].raw_value;
    let mut node_utility = raw_value;

    if let Some(key) = nodes[node_idx].bias_key {
      // ChildVisits(n) = Visits(n) - 1 = sum of the visits to the children.
      if sum_visits > 0 {
        let sum_visits_n = N::from(sum_visits).unwrap();
        // ChildrenUtility from this node's perspective. `sum_values` already
        // holds `-sum_c value(c) * visits(c)`, i.e. the children's utility from
        // the parent's perspective times their visits.
        let children_utility = sum_values / sum_visits_n;
        let weight = sum_visits_n.powf(N::from(Self::BIAS_ALPHA).unwrap());
        let delta = (children_utility - raw_value) * weight;

        let entry = bias.entry(key).or_insert(BiasEntry {
          delta_sum: N::zero(),
          weight_sum: N::zero(),
        });
        // Replace this node's previous contribution to the bucket with its new one.
        entry.delta_sum = entry.delta_sum + delta - nodes[node_idx].last_bias_delta;
        entry.weight_sum = entry.weight_sum + weight - nodes[node_idx].last_bias_weight;
        nodes[node_idx].last_bias_delta = delta;
        nodes[node_idx].last_bias_weight = weight;
      }

      let obs_bias = Self::retrieve_bias(bias, &key);
      node_utility = raw_value + N::from(Self::BIAS_LAMBDA).unwrap() * obs_bias;
    }

    let node = &mut nodes[node_idx];
    node.visits = 1 + sum_visits;
    let visits = N::from(node.visits).unwrap();
    // The bias-corrected utility enters both moments so that the implied
    // variance stays consistent with the value.
    node.value = (node_utility + sum_values) / visits;
    node.value_sq = (node_utility * node_utility + sum_values_sq) / visits;
  }

  /// Hyperparameter for forced playouts at the root with Dirichlet noise.
  /// nforced(c) = sqrt(k * P(c) * total_visits)
  /// When a root child has visits > 0 but visits < nforced(c), its PUCT score
  /// is set to infinity to ensure it receives enough exploration.
  const FORCED_PLAYOUTS_K: u32 = 2;

  fn select_edge(&self, node_idx: usize, noise: bool) -> Option<usize> {
    let node = &self.nodes[node_idx];
    let total_n = N::from(node.visits).unwrap();
    let total_n_sqrt = total_n.sqrt();

    let mut best_score = -N::infinity();
    let mut best = None;

    let c_puct = N::from(1.1).unwrap();
    let c_fpu = N::from(if noise { 0.0 } else { 0.2 }).unwrap();
    let forced_k = N::from(Self::FORCED_PLAYOUTS_K).unwrap();
    let puct_coeff = c_puct * total_n_sqrt;

    let prior_visited = LazyCell::new(|| {
      node
        .children
        .iter()
        .filter(|edge| edge.visits > 0)
        .map(|edge| edge.prior)
        .sum()
    });

    for (idx, edge) in node.children.iter().enumerate() {
      let total_edge_visits = edge.visits + edge.virtual_losses;
      let q = if total_edge_visits > 0 {
        let child_value = self
          .map
          .get(&edge.hash)
          .map_or(N::zero(), |&child_idx| self.nodes[child_idx].value);
        let visits = N::from(edge.visits).unwrap();
        let virtual_losses = N::from(edge.virtual_losses).unwrap();
        // Child value is from child's perspective.
        // Parent wants to maximize own value, which is -child.value
        (-child_value * visits - virtual_losses) / N::from(total_edge_visits).unwrap()
      } else {
        // FPU
        node.value - c_fpu * *prior_visited
      };
      let p = edge.prior;

      // PUCT formula
      // Score = Q(a) + C * P(a) * sqrt(sum(N)) / (N(a) + 1)
      let score = q + puct_coeff * p / N::from(total_edge_visits + 1).unwrap();

      // Forced playouts
      let score = if noise && edge.visits > 0 {
        let nforced = (forced_k * p * total_n).sqrt();
        if N::from(edge.visits).unwrap() < nforced {
          N::infinity()
        } else {
          score
        }
      } else {
        score
      };

      if score > best_score {
        best_score = score;
        best = Some(idx);
      }
    }

    best
  }

  /// Selects a path from the root to a leaf, returning the traversed edges as
  /// `(node_idx, edge_idx)` pairs and whether the leaf is terminal. The edge
  /// references let later steps update the tree (and replay the moves) by direct
  /// indexing instead of re-scanning each node's children by position.
  fn select_path(&mut self) -> (Vec<(usize, usize)>, bool) {
    let mut idx = self.root_idx;
    let mut noise = self.dirichlet_noise;
    let mut path = Vec::new();
    let mut terminal = self.nodes[idx].visits > 0;

    while let Some(edge_idx) = self.select_edge(idx, noise) {
      noise = false;
      let edge = &mut self.nodes[idx].children[edge_idx];
      edge.virtual_losses += 1;
      path.push((idx, edge_idx));
      let hash = edge.hash;
      if let Some(&child_idx) = self.map.get(&hash) {
        idx = child_idx;
      } else {
        terminal = false;
        break;
      }
    }

    (path, terminal)
  }

  fn revert_virtual_loss(&mut self, path: &[(usize, usize)]) {
    for &(node_idx, edge_idx) in path {
      self.nodes[node_idx].children[edge_idx].virtual_losses -= 1;
    }
  }

  fn add_result(&mut self, path: &[(usize, usize)], result: N, children: Vec<Edge<N>>, bias_key: Option<BiasKey>) {
    for &(node_idx, edge_idx) in path {
      self.nodes[node_idx].children[edge_idx].visits += 1;
    }
    // All non-leaf nodes on the path are already in the map (that is how
    // `select_path` advanced through them); only the leaf may be new.
    let leaf_idx = if let Some(&(node_idx, edge_idx)) = path.last() {
      let hash = self.nodes[node_idx].children[edge_idx].hash;
      self.add_node(hash)
    } else {
      self.root_idx
    };
    self.nodes[leaf_idx].raw_value = result;
    self.nodes[leaf_idx].visits = 1;
    self.nodes[leaf_idx].children = children;
    self.nodes[leaf_idx].bias_key = bias_key;
    self.nodes[leaf_idx].last_bias_delta = N::zero();
    self.nodes[leaf_idx].last_bias_weight = N::zero();
    // The leaf has no children yet, so it cannot update its bucket, but it does
    // immediately retrieve the bucket's current observed bias to correct its own
    // utility (matching KataGo's retrieval on node creation). The corrected
    // utility enters both moments, as in `update_node`.
    let obs_bias = bias_key.map_or(N::zero(), |key| Self::retrieve_bias(&self.bias, &key));
    let value = result + N::from(Self::BIAS_LAMBDA).unwrap() * obs_bias;
    self.nodes[leaf_idx].value = value;
    self.nodes[leaf_idx].value_sq = value * value;
    for &(node_idx, _) in path.iter().rev() {
      Self::update_node(&self.map, &mut self.nodes, &mut self.bias, node_idx);
    }
  }

  /// Computes the subtree value bias bucket for a leaf node from the field state
  /// at the leaf (with all of the path's moves played). Returns `None` when
  /// there is no preceding move to key on.
  fn bias_key(field: &Field) -> Option<BiasKey> {
    let moves = &field.moves;
    let last = *moves.last()?;
    let prev = moves.len().checked_sub(2).map_or(0, |i| moves[i]);
    let mover = field.cell(last).get_player();

    let (lx, ly) = field.to_xy(last);
    let (lx, ly) = (lx as i32, ly as i32);
    let mut pattern = [0u8; BIAS_PATTERN_CELLS];
    let mut i = 0;
    for dy in -BIAS_PATTERN_RADIUS..=BIAS_PATTERN_RADIUS {
      for dx in -BIAS_PATTERN_RADIUS..=BIAS_PATTERN_RADIUS {
        pattern[i] = classify_cell(field, lx + dx, ly + dy, mover);
        i += 1;
      }
    }

    Some(BiasKey { last, prev, pattern })
  }

  const PARALLEL_READOUTS: usize = 8;

  fn make_moves(nodes: &[Node<N>], field: &mut Field, path: &[(usize, usize)], mut player: Player, ground: bool) {
    for &(node_idx, edge_idx) in path {
      let pos = nodes[node_idx].children[edge_idx].pos;
      assert!(field.put_point(pos, player), "can't put point, likely a collision");
      if ground {
        field.update_grounded();
      }
      player = player.next();
    }
  }

  fn create_children<R: Rng>(
    &mut self,
    field: &mut Field,
    player: Player,
    policy: &ArrayView2<N>,
    rng: &mut R,
  ) -> Vec<Edge<N>> {
    let stride = field.stride;
    let mut children = Vec::new();

    let mut all_bad = true;

    for pos in field.min_pos()..=field.max_pos() {
      if !field.is_putting_allowed(pos) {
        continue;
      }

      assert!(field.put_point(pos, player));

      if field.get_delta_score(player) < 0 {
        if self.forbid_bad {
          field.undo();
          continue;
        }
      } else {
        all_bad = false;
      }

      let hash = field.colored_hash(player);
      field.undo();

      if self.forbid_bad && field.is_corner(pos) {
        continue;
      }

      let x = to_x(stride, pos);
      let y = to_y(stride, pos);
      let p = policy[(y as usize, x as usize)];

      children.push(Edge {
        pos,
        hash,
        visits: 0,
        prior: p,
        virtual_losses: 0,
      });
    }

    if all_bad {
      return Vec::new();
    }

    // renormalize
    let sum: N = children.iter().map(|child| child.prior).sum();
    if sum > N::zero() {
      for child in children.iter_mut() {
        child.prior = child.prior / sum;
      }
    } else if !children.is_empty() {
      let uniform = N::one() / N::from(children.len()).unwrap();
      for child in children.iter_mut() {
        child.prior = uniform;
      }
    }

    children.shuffle(rng);

    children
  }

  pub async fn mcgs<M: Model<N>, R: Rng>(
    &mut self,
    field: &mut Field,
    player: Player,
    model: &mut M,
    komi_x_2: i32,
    rng: &mut R,
  ) -> Result<(), M::E> {
    let mut leafs = iter::repeat_with(|| self.select_path())
      .take(Self::PARALLEL_READOUTS)
      .collect::<Vec<_>>();
    for (path, _) in &leafs {
      self.revert_virtual_loss(path);
    }

    leafs.sort_unstable();
    leafs.dedup();

    let features_len = field_features_len(field.width(), field.height());
    let mut features = Vec::with_capacity(features_len * leafs.len());
    let mut global = Vec::with_capacity(GLOBAL_FEATURES * leafs.len());
    let red_komi_x_2 = if player == Player::Red { komi_x_2 } else { -komi_x_2 };

    leafs.retain(|(path, terminal)| {
      Self::make_moves(&self.nodes, field, path, player, true);

      let player = if path.len().is_multiple_of(2) {
        player
      } else {
        player.next()
      };

      let leaf_komi_x_2 = if path.len().is_multiple_of(2) {
        komi_x_2
      } else {
        -komi_x_2
      };

      let result = if *terminal || field.is_game_over(red_komi_x_2) {
        // Terminal nodes get no bias correction, matching KataGo.
        self.add_result(path, game_result(field, player, leaf_komi_x_2), Vec::new(), None);
        false
      } else {
        field_features_to_vec::<N>(field, player, field.width(), field.height(), 0, HISTORY_CHANNELS, &mut features);
        global_to_vec(field, player, leaf_komi_x_2, &mut global);
        true
      };

      for _ in 0..path.len() {
        field.undo();
      }

      result
    });

    if features.is_empty() {
      return Ok(());
    }

    let features = Array::from_shape_vec(
      (
        features.len() / features_len,
        CHANNELS,
        field.height() as usize,
        field.width() as usize,
      ),
      features,
    )
    .unwrap();
    let global = Array::from_shape_vec((global.len() / GLOBAL_FEATURES, GLOBAL_FEATURES), global).unwrap();

    let (policies, values) = model.predict(features, global).await?;

    for (i, (path, _)) in leafs.iter().enumerate() {
      Self::make_moves(&self.nodes, field, path, player, false);

      let player = if path.len().is_multiple_of(2) {
        player
      } else {
        player.next()
      };

      let policy = policies.slice(s![i, .., ..]);
      let value = values[(i, 0)] - values[(i, 1)];

      let children = self.create_children(field, player, &policy, rng);
      // Bucket the leaf by the local context of the move that created it. The
      // field currently has all of the path's moves played, so `field.moves`
      // ends with this node's move and the move before it.
      let bias_key = Self::bias_key(field);
      self.add_result(path, value, children, bias_key);

      for _ in 0..path.len() {
        field.undo();
      }
    }

    Ok(())
  }

  /// Number of standard errors below the mean for the lower confidence bound
  /// used to select the move to play.
  const LCB_STDEVS: f64 = 5.0;

  /// A root child is only eligible for LCB selection once it has at least this
  /// proportion of the most visited child's visits.
  const MIN_VISIT_PROP_FOR_LCB: f64 = 0.15;

  /// Lower confidence bound of a root child's value from the root player's
  /// perspective: Q(a) minus `LCB_STDEVS` standard errors. The variance comes
  /// from the second moment propagated through the graph together with the
  /// value, so both describe the same estimate. To behave well at low visit
  /// counts a prior that the variance is the largest possible (the values span
  /// [-1, 1], a range radius of 1) is mixed in with a small weight, which
  /// diminishes as the count grows.
  fn edge_lcb(&self, edge: &Edge<N>) -> Option<N> {
    let &child_idx = self.map.get(&edge.hash)?;
    let child = &self.nodes[child_idx];
    if child.visits == 0 {
      return None;
    }
    let count = N::from(child.visits).unwrap();
    let value_sq = child.value_sq.max(child.value * child.value + N::from(1e-8).unwrap());
    // Every playout has unit weight, so the effective sample size is the count
    // and the prior weight is count / ess³ = 1 / count².
    let prior_weight = (count * count).recip();
    let value_sq = (value_sq * count + (value_sq + N::one()) * prior_weight) / (count + prior_weight);
    let weight_sum = count + prior_weight;
    let weight_sq_sum = count + prior_weight * prior_weight;
    let ess = weight_sum * weight_sum / weight_sq_sum;
    let variance = value_sq - child.value * child.value;
    let radius = N::from(Self::LCB_STDEVS).unwrap() * (variance / ess).sqrt();
    Some(-child.value - radius)
  }

  /// Minimum visits for a root child to be eligible for LCB selection:
  /// `MIN_VISIT_PROP_FOR_LCB` of the most visited child's visits, and at least
  /// one so a value estimate exists.
  fn min_visits_for_lcb(&self) -> u64 {
    let max_visits = self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| edge.visits)
      .max()
      .unwrap_or(0);
    (N::from(Self::MIN_VISIT_PROP_FOR_LCB).unwrap() * N::from(max_visits).unwrap())
      .ceil()
      .to_u64()
      .unwrap_or(u64::MAX)
      .max(1)
  }

  /// Play selection weight of a root child: its LCB when it has enough visits
  /// and observations, otherwise its visits and prior. `Either` orders
  /// `Left < Right`, so any child with an LCB outranks all children without
  /// one, LCBs compare among themselves, and when no LCB is available the most
  /// visited child wins with the prior as the tie-breaker.
  fn edge_weight(&self, edge: &Edge<N>, min_visits: u64) -> Either<(u64, N), N> {
    if edge.visits >= min_visits
      && let Some(lcb) = self.edge_lcb(edge)
    {
      Either::Right(lcb)
    } else {
      Either::Left((edge.visits, edge.prior))
    }
  }

  /// The root child to play: the child with the best play selection weight.
  /// LCB avoids playing a move whose high value rests on too few playouts to
  /// be trusted.
  fn best_edge(&self) -> Option<&Edge<N>> {
    let min_visits = self.min_visits_for_lcb();
    self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| (edge, self.edge_weight(edge, min_visits)))
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
      .map(|(edge, _)| edge)
  }

  /// Get the best move based on LCB selection
  pub fn best_move(&self) -> Option<NonZeroPos> {
    self.best_edge().and_then(|edge| NonZeroPos::new(edge.pos))
  }

  /// Move the root to the best child
  pub fn next_best_root(&mut self) -> Option<NonZeroPos> {
    self.dirichlet_noise = false;
    if let Some((edge_hash, edge_pos)) = self.best_edge().map(|edge| (edge.hash, edge.pos)) {
      self.root_idx = self.add_node(edge_hash);
      NonZeroPos::new(edge_pos)
    } else {
      *self = Self::new(self.forbid_bad);
      None
    }
  }

  /// Move the root to the child with the given position.
  ///
  /// Returns `true` if a matching child existed and the root was advanced into
  /// the persistent graph, or `false` if no such child was found - in which case
  /// the search is reset to a fresh empty tree.
  pub fn next_root(&mut self, pos: Pos) -> bool {
    self.dirichlet_noise = false;
    if let Some(edge_hash) = self.nodes[self.root_idx]
      .children
      .iter()
      .find(|edge| edge.pos == pos)
      .map(|edge| edge.hash)
    {
      self.root_idx = self.add_node(edge_hash);
      true
    } else {
      *self = Self::new(self.forbid_bad);
      false
    }
  }

  /// Compact the search tree by removing unused nodes
  pub fn compact(&mut self) {
    let mut new_search = Self {
      root_idx: 0,
      nodes: Vec::with_capacity(self.nodes.len()),
      map: HashMap::with_capacity_and_hasher(self.map.len(), BuildHasherDefault::default()),
      // Carry the subtree value bias buckets over; the surviving nodes keep
      // their contributions and the dropped nodes' contributions are decayed
      // below.
      bias: mem::take(&mut self.bias),
      dirichlet_noise: self.dirichlet_noise,
      forbid_bad: self.forbid_bad,
    };

    let mut queue = VecDeque::new();
    for edge in &mut self.nodes[self.root_idx].children {
      queue.push_back(edge.hash);
    }

    new_search.nodes.push(mem::take(&mut self.nodes[self.root_idx]));

    while let Some(hash) = queue.pop_front() {
      if let Some(child_idx) = self.map.remove(&hash) {
        for edge in self.nodes[child_idx]
          .children
          .iter()
          .filter(|edge| self.map.contains_key(&edge.hash))
        {
          queue.push_back(edge.hash);
        }
        new_search.nodes.push(mem::take(&mut self.nodes[child_idx]));
        new_search.map.insert(hash, new_search.nodes.len() - 1);
      }
    }

    // Surviving nodes were removed from `self.map` above; whatever remains are
    // the nodes being dropped. Decay their contribution to their buckets, so
    // that the bias of a reused tactic carries over only partially rather than
    // lingering at full strength forever (KataGo's `subtreeValueBiasFreeProp`).
    let free_prop = N::from(Self::BIAS_FREE_PROP).unwrap();
    for &dropped_idx in self.map.values() {
      let node = &self.nodes[dropped_idx];
      if let Some(key) = node.bias_key
        && let Some(entry) = new_search.bias.get_mut(&key)
      {
        entry.delta_sum = entry.delta_sum - node.last_bias_delta * free_prop;
        entry.weight_sum = entry.weight_sum - node.last_bias_weight * free_prop;
      }
    }

    *self = new_search;
  }

  /// Get the visits for each child of the root node
  pub fn visits_with_prior(&self) -> impl Iterator<Item = (Pos, (u64, N))> + '_ {
    self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| (edge.pos, (edge.visits, edge.prior)))
  }

  /// Get the play selection weight for each child of the root node: the LCB
  /// when available, otherwise visits and prior. Consumers playing the
  /// max-weight move pick the same child as `best_move`.
  pub fn play_selection(&self) -> Vec<(Pos, Either<(u64, N), N>)> {
    let min_visits = self.min_visits_for_lcb();
    self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| (edge.pos, self.edge_weight(edge, min_visits)))
      .collect()
  }

  /// Get the visits for each child of the root node
  pub fn visits(&self) -> impl Iterator<Item = (Pos, u64)> + '_ {
    self
      .visits_with_prior()
      .map(|(pos, (visits, _))| (pos, visits))
      .filter(|(_, visits)| *visits > 0)
  }

  /// Get pruned visits for the policy target.
  ///
  /// This implements policy target pruning:
  /// 1. Find the best child c* (most visits).
  /// 2. Compute PUCT(c*) using final utility estimates.
  /// 3. For each other child c, reduce its visits so that PUCT(c) does not
  ///    exceed PUCT(c*).
  /// 4. Prune children reduced to <= 1 visit.
  ///
  /// This decouples the policy training target from the forced exploration
  /// playouts used during search, producing a cleaner training signal.
  pub fn pruned_visits(&self) -> impl Iterator<Item = (Pos, u64)> + '_ {
    let root = &self.nodes[self.root_idx];
    let children = &root.children;

    // Find the best child (most visits)
    let (best_idx, best_edge) = if let Some(result) = children.iter().enumerate().max_by_key(|(_, edge)| edge.visits) {
      result
    } else {
      return Either::Left(iter::empty());
    };

    if best_edge.visits == 0 {
      return Either::Left(iter::empty());
    }

    let c_puct = N::from(1.1).unwrap();
    let total_n = N::from(root.visits).unwrap();
    let total_n_sqrt = total_n.sqrt();

    let best_child_value = self
      .map
      .get(&best_edge.hash)
      .map_or(N::zero(), |&child_idx| self.nodes[child_idx].value);
    let best_q = -best_child_value;

    // Compute PUCT(c*) for the best child
    let best_puct = best_q + c_puct * best_edge.prior * total_n_sqrt / N::from(best_edge.visits + 1).unwrap();

    let result = children.iter().enumerate().filter_map(move |(idx, edge)| {
      if edge.visits == 0 {
        return None;
      }

      if idx == best_idx {
        return Some((edge.pos, edge.visits));
      }

      let child_value = self
        .map
        .get(&edge.hash)
        .map_or(N::zero(), |&child_idx| self.nodes[child_idx].value);
      let child_q = -child_value;

      // Calculate nforced just like in select_edge
      let nforced = (N::from(Self::FORCED_PLAYOUTS_K).unwrap() * edge.prior * total_n)
        .sqrt()
        .ceil()
        .to_u64()
        .unwrap_or(0);

      // Compute the weight PUCT would have naturally allocated to this child
      // by inverting the PUCT formula:
      //   best_puct = child_q + c_puct * P(c) * sqrt(N_parent) / (N(c) + 1)
      //   N(c) = c_puct * P(c) * sqrt(N_parent) / (best_puct - child_q) - 1
      let explore_component = best_puct - child_q;

      let retrospective_visits = if explore_component <= N::zero() {
        edge.visits
      } else {
        let max_n = c_puct * edge.prior * total_n_sqrt / explore_component - N::one();
        if max_n < N::zero() {
          0u64
        } else {
          max_n.ceil().to_u64().unwrap_or(edge.visits)
        }
      };

      // Cap the visits to what PUCT would have allocated
      let min_allowed_visits = edge.visits.saturating_sub(nforced);
      let reduced = retrospective_visits.clamp(min_allowed_visits, edge.visits);

      // Prune children reduced to <= 1 visit
      if reduced > 1 { Some((edge.pos, reduced)) } else { None }
    });

    Either::Right(result)
  }

  /// Get the value of the root node
  pub fn value(&self) -> N {
    self.nodes[self.root_idx].value
  }

  /// Get the raw neural net value of the root node, without any search.
  pub fn raw_value(&self) -> N {
    self.nodes[self.root_idx].raw_value
  }

  /// Snapshot the policy priors of the root's children into a vector indexed by
  /// position.
  ///
  /// Useful for capturing the raw network priors before they are overwritten in
  /// place by temperature scaling and Dirichlet noise.
  pub fn root_priors(&self, priors: &mut [N]) {
    let children = &self.nodes[self.root_idx].children;
    priors.fill(N::zero());
    for edge in children {
      priors[edge.pos] = edge.prior;
    }
  }

  /// Policy surprise of a policy training target relative to the prior.
  ///
  /// This is the KL divergence from the policy `priors` (indexed by position) to
  /// the `target` distribution: `sum_i target_i * (ln(target_i) - ln(prior_i))`.
  ///
  /// A large value means the search ended up favouring moves quite differently
  /// from what the raw policy expected, i.e. the position was "surprising". It is
  /// used for policy surprise weighting of training samples, overweighting such
  /// positions in the training data.
  pub fn policy_surprise(target: &[(Pos, u64)], priors: &[N]) -> N {
    let total = target.iter().map(|&(_, visits)| visits).sum::<u64>();
    if total == 0 {
      return N::zero();
    }
    let total = N::from(total).unwrap();
    // Floor on the prior to avoid `ln(0)` for targets on moves the prior gave a
    // zero probability (and to bound the surprise of such moves).
    let offset = N::from(1e-30).unwrap();
    let mut surprise = N::zero();
    for &(pos, visits) in target {
      if visits == 0 {
        continue;
      }
      let t = N::from(visits).unwrap() / total;
      surprise = surprise + t * (t.ln() - (priors[pos] + offset).ln());
    }
    // Guard against tiny negative values from floating point imprecision.
    surprise.max(N::zero())
  }
}

impl<N: Float + Sum + SampleUniform> Search<N> {
  /// Move the root to a random child based on visit counts
  pub fn next_root_with_temperature<R: Rng>(&mut self, temperature: N, rng: &mut R) -> Option<NonZeroPos> {
    let root = &self.nodes[self.root_idx];
    let max_logit = N::from(
      root
        .children
        .iter()
        .map(|edge| edge.visits)
        .max()
        .filter(|&visits| visits > 0)?,
    )
    .unwrap()
    .ln();
    let sum_exp: N = root
      .children
      .iter()
      .filter(|edge| edge.visits > 0)
      .map(|edge| ((N::from(edge.visits).unwrap().ln() - max_logit) / temperature).exp())
      .sum();

    let mut sample = rng.random_range(N::zero()..sum_exp);
    let mut chosen_edge = None;

    for edge in root.children.iter() {
      if edge.visits == 0 {
        continue;
      }

      let prob = ((N::from(edge.visits).unwrap().ln() - max_logit) / temperature).exp();

      if prob >= sample {
        chosen_edge = Some((edge.hash, edge.pos));
        break;
      } else {
        sample = sample - prob;
      }
    }

    self.dirichlet_noise = false;
    if let Some((hash, pos)) = chosen_edge {
      self.root_idx = self.add_node(hash);
      NonZeroPos::new(pos)
    } else {
      *self = Self::new(self.forbid_bad);
      None
    }
  }
}

impl<N> Search<N>
where
  N: Float + Sum,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, epsilon: N, total_concentration: N, temperature: N) {
    self.nodes[self.root_idx].apply_temperature(temperature);
    self.nodes[self.root_idx].add_dirichlet_noise(rng, epsilon, total_concentration);
    self.dirichlet_noise = true;
  }
}
