use crate::field_features::{CHANNELS, field_features_len, field_features_to_vec};
use crate::model::Model;
use ndarray::{Array, ArrayView2, s};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Gamma, Open01, StandardNormal};
use std::cell::LazyCell;
use std::collections::VecDeque;
use std::mem;
use std::{collections::HashMap, iter, iter::Sum};

/// Represents an edge from a parent to a child in the graph.
#[derive(Clone, PartialEq, Debug)]
pub struct Edge<N: Float> {
  pub pos: Pos,
  /// Zobrist hash of the child state
  pub hash: u64,
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
  pub value: N, // utilityAvg
  /// U(n): Raw utility from the neural net for this state.
  pub raw_value: N,
  /// Edges to children.
  pub children: Vec<Edge<N>>,
}

impl<N: Float> Node<N> {
  pub fn new() -> Self {
    Node {
      visits: 0,
      value: N::zero(),
      raw_value: N::zero(),
      children: Vec::new(),
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

  pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, epsilon: N, shape: N) {
    let gamma = Gamma::<N>::new(N::from(shape).unwrap(), N::one()).unwrap();
    let mut dirichlet = gamma.sample_iter(rng).take(self.children.len()).collect::<Vec<_>>();
    let sum = dirichlet.iter().cloned().sum::<N>();
    if sum == N::zero() {
      return;
    }
    for eta in dirichlet.iter_mut() {
      *eta = *eta / sum;
    }
    for (child, eta) in self.children.iter_mut().zip(dirichlet.into_iter()) {
      child.prior = child.prior * (N::one() - epsilon) + epsilon * eta;
    }
  }
}

impl<N: Float> Default for Node<N> {
  fn default() -> Self {
    Self::new()
  }
}

pub fn game_result<N: Float>(field: &Field, player: Player) -> N {
  N::from(field.score(player).signum()).unwrap()
}

#[derive(Clone, PartialEq, Debug)]
pub struct Search<N: Float> {
  /// Index of the root node in `nodes`
  pub root_idx: usize,
  /// Arena allocation for nodes
  pub nodes: Vec<Node<N>>,
  /// Maps Zobrist hash -> index in `nodes`
  pub map: HashMap<u64, usize>,
  /// Whether dirichlet noise was added to the root node
  pub dirichlet_noise: bool,
}

impl<N: Float> Search<N> {
  pub fn new() -> Self {
    let mut search = Search {
      root_idx: 0,
      nodes: Vec::new(),
      map: HashMap::new(),
      dirichlet_noise: false,
    };

    // Initialize root
    search.nodes.push(Node::new());
    search
  }
}

impl<N: Float + Sum> Search<N> {
  fn add_node(&mut self, hash: u64) -> usize {
    *self.map.entry(hash).or_insert_with(|| {
      let idx = self.nodes.len();
      let node = Node::new();
      self.nodes.push(node);
      idx
    })
  }

  fn update_node(map: &HashMap<u64, usize>, nodes: &mut [Node<N>], node_idx: usize) {
    let mut sum_values = N::zero();
    let mut sum_visits = 0;

    for edge in nodes[node_idx].children.iter() {
      if let Some(&child_idx) = map.get(&edge.hash) {
        sum_values = sum_values - N::from(edge.visits).unwrap() * nodes[child_idx].value;
      }
      sum_visits += edge.visits;
    }

    nodes[node_idx].visits = 1 + sum_visits;
    nodes[node_idx].value = (nodes[node_idx].raw_value + sum_values) / N::from(nodes[node_idx].visits).unwrap();
  }

  fn select_edge(&self, node_idx: usize, noise: bool) -> Option<usize> {
    let node = &self.nodes[node_idx];
    let total_n_sqrt = N::from(node.visits).unwrap().sqrt();

    let mut best_score = -N::infinity();
    let mut best = None;

    let prior_visited = LazyCell::new(|| {
      node
        .children
        .iter()
        .filter(|edge| edge.visits > 0)
        .map(|edge| edge.prior)
        .sum()
    });

    for (idx, edge) in node.children.iter().enumerate() {
      let child_value = self
        .map
        .get(&edge.hash)
        .map_or(N::zero(), |&child_idx| self.nodes[child_idx].value);

      // Hyperparameter for PUCT
      let c_puct = N::from(1.1).unwrap();
      let c_fpu = N::from(if noise { 0.0 } else { 0.2 }).unwrap();

      // Child value is from child's perspective.
      // Parent wants to maximize own value, which is -child.value
      let q = if edge.visits > 0 {
        let visits = N::from(edge.visits).unwrap();
        let virtual_losses = N::from(edge.virtual_losses).unwrap();
        (-child_value * visits - virtual_losses) / (visits + virtual_losses)
      } else {
        // FPU
        node.value - c_fpu * *prior_visited
      };
      let n = edge.visits + edge.virtual_losses;
      let p = edge.prior;

      // PUCT formula
      // Score = Q(a) + C * P(a) * sqrt(sum(N)) / (N(a) + 1)
      let score = q + c_puct * p * total_n_sqrt / N::from(n + 1).unwrap();

      if score > best_score {
        best_score = score;
        best = Some(idx);
      }
    }

    best
  }

  fn select_path(&mut self) -> Vec<Pos> {
    let mut moves = Vec::new();
    let mut idx = self.root_idx;
    let mut noise = self.dirichlet_noise;

    while let Some(edge_idx) = self.select_edge(idx, noise) {
      noise = false;
      self.nodes[idx].children[edge_idx].virtual_losses += 1;
      moves.push(self.nodes[idx].children[edge_idx].pos);
      if let Some(&child_idx) = self.map.get(&self.nodes[idx].children[edge_idx].hash) {
        idx = child_idx;
      } else {
        break;
      }
    }

    moves
  }

  fn revert_virtual_loss(&mut self, moves: &[Pos]) {
    let mut idx = self.root_idx;
    for &pos in moves {
      let edge_idx = self.nodes[idx]
        .children
        .iter()
        .position(|edge| edge.pos == pos)
        .unwrap();
      self.nodes[idx].children[edge_idx].virtual_losses -= 1;
      if let Some(&child_idx) = self.map.get(&self.nodes[idx].children[edge_idx].hash) {
        idx = child_idx;
      } else {
        break;
      }
    }
  }

  fn add_result(&mut self, moves: &[Pos], result: N, children: Vec<Edge<N>>) {
    let mut indices = Vec::new();
    let mut idx = self.root_idx;
    for &pos in moves {
      indices.push(idx);
      let edge_idx = self.nodes[idx]
        .children
        .iter()
        .position(|edge| edge.pos == pos)
        .unwrap();
      self.nodes[idx].children[edge_idx].visits += 1;
      idx = self.add_node(self.nodes[idx].children[edge_idx].hash);
    }
    self.nodes[idx].value = result;
    self.nodes[idx].raw_value = result;
    self.nodes[idx].visits = 1;
    self.nodes[idx].children = children;
    while let Some(idx) = indices.pop() {
      Self::update_node(&self.map, &mut self.nodes, idx);
    }
  }

  const PARALLEL_READOUTS: usize = 8;

  fn make_moves(field: &mut Field, moves: &[Pos], mut player: Player) {
    for &pos in moves {
      assert!(field.put_point(pos, player), "can't put point, likely a collision");
      field.update_grounded();
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

    for pos in field.min_pos()..=field.max_pos() {
      if !field.is_putting_allowed(pos) || field.is_corner(pos) {
        continue;
      }

      assert!(field.put_point(pos, player));
      let hash = field.colored_hash(player);
      field.undo();

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

    // renormalize
    let sum: N = children.iter().map(|child| child.prior).sum();
    for child in children.iter_mut() {
      child.prior = child.prior / sum;
    }

    children.shuffle(rng);

    children
  }

  pub fn mcgs<M: Model<N>, R: Rng>(
    &mut self,
    field: &mut Field,
    player: Player,
    model: &mut M,
    rng: &mut R,
  ) -> Result<(), M::E> {
    let mut leafs = iter::repeat_with(|| self.select_path())
      .take(Self::PARALLEL_READOUTS)
      .collect::<Vec<_>>();
    for moves in &leafs {
      self.revert_virtual_loss(moves);
    }

    leafs.sort_unstable();
    leafs.dedup();

    let features_len = field_features_len(field.width(), field.height());
    let mut features = Vec::with_capacity(features_len * leafs.len());

    leafs.retain(|leaf| {
      Self::make_moves(field, leaf, player);

      let player = if leaf.len().is_multiple_of(2) {
        player
      } else {
        player.next()
      };

      let result = if field.is_game_over() {
        self.add_result(leaf, game_result(field, player), Vec::new());
        false
      } else {
        field_features_to_vec::<N>(field, player, field.width(), field.height(), 0, &mut features);
        true
      };

      for _ in 0..leaf.len() {
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

    let (policies, values) = model.predict(features)?;

    for (i, leaf) in leafs.iter().enumerate() {
      Self::make_moves(field, leaf, player);

      let player = if (leaf.len()).is_multiple_of(2) {
        player
      } else {
        player.next()
      };

      let policy = policies.slice(s![i, .., ..]);
      let value = values[(i, 0)] - values[(i, 1)];

      let children = self.create_children(field, player, &policy, rng);
      self.add_result(leaf, value, children);

      for _ in 0..leaf.len() {
        field.undo();
      }
    }

    Ok(())
  }

  /// Get the best move based on visit counts
  pub fn best_move(&self) -> Option<NonZeroPos> {
    self.nodes[self.root_idx]
      .children
      .iter()
      .max_by_key(|edge| edge.visits)
      .and_then(|edge| NonZeroPos::new(edge.pos))
  }

  /// Move the root to the best child
  pub fn next_best_root(&mut self) -> Option<NonZeroPos> {
    self.dirichlet_noise = false;
    if let Some((edge_hash, edge_pos)) = self.nodes[self.root_idx]
      .children
      .iter()
      .max_by_key(|edge| edge.visits)
      .map(|edge| (edge.hash, edge.pos))
    {
      self.root_idx = self.add_node(edge_hash);
      NonZeroPos::new(edge_pos)
    } else {
      *self = Self::new();
      None
    }
  }

  /// Move the root to the child with the given position
  pub fn next_root(&mut self, pos: Pos) {
    self.dirichlet_noise = false;
    if let Some(edge_hash) = self.nodes[self.root_idx]
      .children
      .iter()
      .find(|edge| edge.pos == pos)
      .map(|edge| edge.hash)
    {
      self.root_idx = self.add_node(edge_hash);
    } else {
      *self = Self::new();
    }
  }

  /// Compact the search tree by removing unused nodes
  pub fn compact(&mut self) {
    let mut new_search = Self {
      root_idx: 0,
      nodes: Vec::with_capacity(self.nodes.len()),
      map: HashMap::with_capacity(self.map.len()),
      dirichlet_noise: self.dirichlet_noise,
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

    *self = new_search;
  }

  /// Get the visits for each child of the root node
  pub fn visits(&self) -> impl Iterator<Item = (Pos, u64)> + '_ {
    self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| (edge.pos, edge.visits))
  }

  /// Get the value of the root node
  pub fn value(&self) -> N {
    self.nodes[self.root_idx].value
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

      let logit = N::from(edge.visits).unwrap().ln() / temperature;
      let prob = (logit - max_logit).exp();

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
      *self = Self::new();
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
  pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, epsilon: N, shape: N, temperature: N) {
    self.nodes[self.root_idx].apply_temperature(temperature);
    self.nodes[self.root_idx].add_dirichlet_noise(rng, epsilon, shape);
    self.dirichlet_noise = true;
  }
}

impl<N: Float> Default for Search<N> {
  fn default() -> Self {
    Self::new()
  }
}
