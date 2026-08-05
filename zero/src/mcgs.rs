use crate::field_features::{
  CHANNELS, GLOBAL_FEATURES, HISTORY_CHANNELS, field_features, field_features_len, field_features_to_vec,
  global as global_features, global_to_vec,
};
use crate::model::Model;
use either::Either;
use ndarray::{Array, ArrayView2, Axis, s};
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

/// Degrees of freedom of the Student's t distribution whose CDF turns a child's
/// value gap into a weight multiplier.
const VALUE_WEIGHT_DEGREES_OF_FREEDOM: f64 = 3.0;
/// Range and resolution of [`VALUE_WEIGHT_CDF`].
const VALUE_WEIGHT_CDF_RANGE: f64 = 50.0;
const VALUE_WEIGHT_CDF_BUCKETS: usize = 2000;

/// CDF of Student's t with [`VALUE_WEIGHT_DEGREES_OF_FREEDOM`] degrees of freedom, in closed
/// form: `1/2 + (u / (1 + u²) + atan(u)) / pi` for `u = x / sqrt(3)`. Exact - it
/// differentiates to `2 / (pi * sqrt(3) * (1 + x²/3)²)`.
pub(crate) fn t_cdf(x: f64) -> f64 {
  let u = x / VALUE_WEIGHT_DEGREES_OF_FREEDOM.sqrt();
  0.5 + (u / (1.0 + u * u) + u.atan()) / std::f64::consts::PI
}

/// Tabulated [`t_cdf`], sampled at the bucket edges. The closed form needs an
/// `atan` per child per backup, which measured at ~40% of total search time, so
/// it is interpolated instead (linear interpolation error here is ~6e-5).
static VALUE_WEIGHT_CDF: std::sync::LazyLock<Vec<f64>> = std::sync::LazyLock::new(|| {
  (0..=VALUE_WEIGHT_CDF_BUCKETS)
    .map(|i| {
      let t = i as f64 / VALUE_WEIGHT_CDF_BUCKETS as f64;
      t_cdf(-VALUE_WEIGHT_CDF_RANGE + 2.0 * VALUE_WEIGHT_CDF_RANGE * t)
    })
    .collect()
});

/// Interpolated lookup into [`VALUE_WEIGHT_CDF`], clamped to its range.
pub(crate) fn value_weight_cdf(x: f64) -> f64 {
  let table = &*VALUE_WEIGHT_CDF;
  let scaled = (x + VALUE_WEIGHT_CDF_RANGE) * (VALUE_WEIGHT_CDF_BUCKETS as f64 / (2.0 * VALUE_WEIGHT_CDF_RANGE));
  if scaled <= 0.0 {
    return table[0];
  }
  if scaled >= VALUE_WEIGHT_CDF_BUCKETS as f64 {
    return table[VALUE_WEIGHT_CDF_BUCKETS];
  }
  let bucket = scaled as usize;
  let frac = scaled - bucket as f64;
  table[bucket] + (table[bucket + 1] - table[bucket]) * frac
}

/// Radius of the square local pattern used as part of a [`BiasKey`].
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
    None => 1,                  // empty / neutral
    Some(p) if p == mover => 2, // owned by the mover
    Some(_) => 3,               // owned by the opponent
  }
}

/// Identifies a subtree value bias bucket.
///
/// Nodes are bucketed by the local context of the last move so that the
/// observed bias of the neural net for a given tactic can be shared across the
/// many places that tactic appears in the search tree.
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
/// `delta_sum` is `sum_n (ChildrenUtility(n) - NNUtility(n)) * ChildWeight(n)^alpha`
/// and `weight_sum` is `sum_n ChildWeight(n)^alpha`, both summed over the nodes
/// `n` currently in the bucket. The bucket's observed bias is their ratio.
#[derive(Clone, PartialEq, Debug)]
pub struct BiasEntry<N: Float> {
  pub delta_sum: N,
  pub weight_sum: N,
}

/// How much a root child is worth playing: its LCB once it carries enough
/// search weight to be trusted, otherwise its search weight and prior. `Either`
/// orders `Left < Right`, so every child with an LCB outranks every child
/// without one.
pub type PlaySelectionWeight<N> = Either<(N, N), N>;

/// A visited child as [`Search::update_node`] aggregates it: its node index, the
/// weight it contributes to the parent, its utility from the *parent's*
/// perspective, and the raw policy prior of the edge leading to it. The weight is
/// the only mutable part - both the noise pruning and the value downweighting
/// rewrite it before the children are averaged in.
type Child<N> = (usize, N, N, N);

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
  /// N(n): Total visits to this node, i.e. `own_visits + sum(edge.visits)`.
  pub visits: u64,
  /// How many times this node's own state was evaluated. One for every node that
  /// has children, since a node is expanded once and descended through
  /// afterwards. A childless one - terminal, or with every move forbidden - has
  /// no edge to descend, so every playout that reaches it evaluates it again.
  /// Tracked so that [`Search::update_node`] can rebuild `visits` and the weights
  /// below without assuming a single evaluation - assuming it would pin them at
  /// one playout while the parent edges kept counting.
  pub own_visits: u64,
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
  /// Total weight of this node's own evaluations: certain evaluations (low
  /// predicted short-term value error) weigh more, up to
  /// `UNCERTAINTY_MAX_WEIGHT`. This is the sum over all `own_visits` of them, so
  /// it is one evaluation's weight in the usual case of `own_visits == 1`.
  pub weight: N,
  /// Sum of the squares of the weights making up `weight`. Kept separately
  /// rather than derived from `weight` because a node's own evaluations need not
  /// all weigh the same: a node whose every move is forbidden is first evaluated
  /// by the net and afterwards as an exact terminal.
  pub weight_sq: N,
  /// W(n): Total weight, the weighted analog of `visits`:
  /// `weight` plus the children's edge weights.
  pub weight_sum: N,
  /// Sum of the squares of the weights making up `weight_sum`. Only used to
  /// derive the effective sample size `weight_sum² / weight_sq_sum` for the LCB:
  /// with uncertainty weighting the playouts no longer carry equal weight, so
  /// the count of them overstates how much independent evidence they are.
  pub weight_sq_sum: N,
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
      own_visits: 0,
      value: N::zero(),
      raw_value: N::zero(),
      value_sq: N::zero(),
      weight: N::zero(),
      weight_sq: N::zero(),
      weight_sum: N::zero(),
      weight_sq_sum: N::zero(),
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

/// The knobs that differ between generating training data and playing.
///
/// The exploration coefficients multiply a utility, so in principle they scale
/// with the range of one. The values here are taken from a search whose utility
/// also carries score terms, giving it a range about 1.4 times as wide as the
/// pure win/loss utility used here, which would argue for dividing them by that.
/// They are left as they are because nothing has measured which plays better.
/// Where the range enters a formula explicitly - the utility a virtual loss pulls
/// towards, the largest-possible-variance prior of the LCB - it is correctly 1.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Params {
  /// Whether to forbid apriori bad moves. Self-play wants to see them so the
  /// policy learns to reject them; play does not want to waste search on them.
  pub forbid_bad: bool,
  /// Whether an evaluation's weight follows the net's predicted short-term
  /// value error instead of being 1. Takes a net that predicts one; see
  /// [`Model::predicts_uncertainty`].
  pub uncertainty: bool,
  /// How far the priors an expanded node gets are moved from the net's trained
  /// policy towards its optimistic one, in `[0, 1]`; 0 uses the trained policy
  /// as is. See [`Model::predict`] for what the optimistic policy is.
  pub policy_optimism: f64,
  /// The optimism of the root's own priors. Deeper in the tree an inflated
  /// prior is self-correcting - playouts flow in, the values come back bad, and
  /// visits drain away - but at the root the prior shapes the very visit
  /// distribution the move is chosen by, and the refutation may not be found
  /// before the playouts run out. So the root gets its own, more conservative
  /// weight, and a root inherited from a previous search - whose priors were
  /// computed at [`Self::policy_optimism`] back when it was an ordinary leaf -
  /// has them re-predicted at this weight before the next search descends.
  pub root_policy_optimism: f64,
  /// Base exploration coefficient of the PUCT formula.
  pub cpuct_exploration: f64,
  /// How much the exploration coefficient grows with the logarithm of the total
  /// weight behind a node's children; 0 keeps it flat. See
  /// [`Search::explore_scaling`].
  pub cpuct_exploration_log: f64,
  /// How strongly the PUCT coefficient follows the node's observed utility
  /// standard deviation; 0 disables the scaling.
  pub cpuct_utility_stdev_scale: f64,
  /// How strongly a child whose value sits below its siblings is downweighted
  /// when averaging them into the parent; 0 disables it.
  pub value_weight_exponent: f64,
  /// The utility gap at which a child holding more weight than its prior
  /// justifies loses most of the excess; 0 disables the pruning. See
  /// [`Search::prune_noise_weight`].
  pub noise_prune_utility_scale: f64,
  /// FPU reduction applied at the root, where the alternative to a reduction is
  /// exploring widely.
  pub root_fpu_reduction_max: f64,
}

impl Params {
  /// Generating training data: apriori bad moves are searched so the policy
  /// learns to reject them, every playout counts the same so the weight target is
  /// a plain visit distribution, and the root explores widely rather than being
  /// pulled towards its current best.
  pub const SELF_PLAY: Self = Params {
    forbid_bad: false,
    uncertainty: false,
    // The trained policy as is: the search's own weights become the next
    // policy target, and an optimistic prior would bend that target towards
    // whatever the optimistic head favours instead of towards what the search
    // found.
    policy_optimism: 0.0,
    root_policy_optimism: 0.0,
    cpuct_exploration: 1.05,
    cpuct_exploration_log: 0.28,
    cpuct_utility_stdev_scale: 0.0,
    root_fpu_reduction_max: 0.0,
    value_weight_exponent: 0.5,
    // Off so that the weight the target is built from stays the visit count.
    // Discarding weight is a value-side correction, and letting it through would
    // teach the policy that a move is worth less than the search spent on it
    // exactly where the search was told to spend more than it wanted to.
    noise_prune_utility_scale: 0.0,
  };

  /// Playing to win: no search spent on moves that lose points outright, certain
  /// evaluations weighed more heavily than unsure ones, and exploration that
  /// follows how volatile a node has proved to be.
  pub const PLAY: Self = Params {
    forbid_bad: true,
    uncertainty: true,
    // Fully optimistic priors below the root. A move that only pays off in the
    // lines where the position turns out better than the net expects is exactly
    // what a search is there to find, and one the trained policy - an average
    // over how those positions really went - ranks far too low to ever be tried.
    policy_optimism: 1.0,
    // But mostly the trained policy at the root, where the prior picks the move
    // rather than steering exploration: the optimistic head favours overplays
    // that bank on the opponent missing the refutation, and the root is where
    // an unrefuted one stops costing playouts and becomes the move played.
    root_policy_optimism: 0.2,
    cpuct_exploration: 1.0,
    cpuct_exploration_log: 0.45,
    cpuct_utility_stdev_scale: 0.85,
    root_fpu_reduction_max: 0.1,
    value_weight_exponent: 0.25,
    noise_prune_utility_scale: 0.15,
  };
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
  /// Whether the tree was reused for a new root and so still holds values
  /// computed against subtree value bias buckets that have moved since. The next
  /// search recomputes them before it descends.
  pub stats_stale: bool,
  /// Whether the root was inherited from a previous search and so still carries
  /// the priors it got as an ordinary leaf, interpolated at
  /// [`Params::policy_optimism`]. When the root wants a different optimism, the
  /// next search re-predicts them before it descends.
  pub root_priors_stale: bool,
  /// Knobs that differ between self-play and play.
  pub params: Params,
}

impl<N: Float> Search<N> {
  pub fn new(params: Params) -> Self {
    let mut search = Search {
      root_idx: 0,
      nodes: Vec::new(),
      map: HashMap::default(),
      bias: std::collections::HashMap::default(),
      dirichlet_noise: false,
      stats_stale: false,
      root_priors_stale: false,
      params,
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
  /// node's utility; `alpha` is the exponent applied to the total weight of a
  /// node's children when weighting its contribution to the bucket. `free_prop`
  /// is the fraction of a node's bucket contribution that is removed when the
  /// node leaves the reused search tree.
  /// Setting `lambda` to zero disables the correction entirely.
  const BIAS_LAMBDA: f64 = 0.3;
  const BIAS_ALPHA: f64 = 0.8;
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

  /// A child may not hold much more weight than the policy justifies while
  /// looking worse than the children the policy ranked above it. Drops the excess
  /// and returns the reduced total weight.
  ///
  /// Root exploration noise deliberately forces weight onto moves the policy
  /// thinks little of, and that weight then drags the root's value towards
  /// whatever those moves turn out to be worth. The same happens without noise
  /// wherever exploration overshoots. Unlike
  /// [`Self::downweight_bad_children`], which only redistributes weight and so
  /// leaves the amount of evidence behind a node alone, this genuinely discards
  /// it: weight spent proving a move bad is not evidence about the position.
  ///
  /// Children are considered in descending policy order, each judged against the
  /// weighted average utility of the ones before it. A child holding more than
  /// twice the share of their weight its prior would give it, while its utility
  /// sits `gap` below their average, keeps `exp(-gap / scale)` of that excess -
  /// so a marginally worse child is left nearly alone and a clearly refuted one
  /// is cut back to its lenient share.
  fn prune_noise_weight(children: &mut [Child<N>], total_weight: N, scale: f64) -> N {
    if scale == 0.0 || children.len() < 2 || total_weight <= N::from(1e-5).unwrap() {
      return total_weight;
    }
    // The policy order is what makes "the children before it" mean "the children
    // the policy preferred"; the order the edges happen to sit in is arbitrary.
    children.sort_unstable_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    let scale = N::from(scale).unwrap();
    let leniency = N::from(2.0).unwrap();
    let mut utility_sum = N::zero();
    let mut weight_sum = N::zero();
    let mut prior_sum = N::zero();
    for &mut (_, ref mut weight, utility, prior) in children.iter_mut() {
      if weight_sum > N::zero() && prior_sum > N::zero() {
        let gap = utility_sum / weight_sum - utility;
        if gap > N::zero() {
          let share = leniency * weight_sum * prior / prior_sum;
          if *weight > share {
            *weight = *weight - (*weight - share) * (N::one() - (-gap / scale).exp());
          }
        }
      }
      utility_sum = utility_sum + utility * *weight;
      weight_sum = weight_sum + *weight;
      prior_sum = prior_sum + prior;
    }
    weight_sum
  }

  /// Reweights children by how far their utility sits below the weighted mean,
  /// then renormalizes to the original total weight.
  ///
  /// Each child's weight is multiplied by `cdf(z)^exponent`, where `z` is its
  /// utility gap in units of a spread that shrinks as the child accumulates
  /// weight (`sqrt(1 / (1.5 * sqrt(weight)))`) - so a lightly searched child is
  /// judged leniently and a heavily searched one that still looks bad is cut hard.
  fn downweight_bad_children(children: &mut [Child<N>], total_weight: N, exponent: f64) {
    if exponent == 0.0 || children.len() < 2 || total_weight <= N::zero() {
      return;
    }
    let mean_utility = children
      .iter()
      .map(|&(_, weight, utility, _)| weight * utility)
      .sum::<N>()
      / total_weight;

    // Minimum variance for stability regardless of the formula above.
    let min_variance = N::from(1e-8).unwrap();
    let precision_scale = N::from(1.5).unwrap();
    let offset = N::from(0.0001).unwrap();
    // `powf` costs more than the rest of this loop put together, so the two
    // exponents any configuration actually uses go through roots instead.
    let raise: fn(N) -> N = if exponent == 0.5 {
      |p| p.sqrt()
    } else if exponent == 0.25 {
      |p| p.sqrt().sqrt()
    } else {
      // Only reachable if a configuration picks some other exponent.
      return Self::downweight_bad_children_powf(children, total_weight, mean_utility, exponent);
    };

    let mut new_total = N::zero();
    for (_, weight, utility, _) in children.iter_mut() {
      let stdev = (min_variance + (precision_scale * weight.sqrt()).recip()).sqrt();
      let z = (*utility - mean_utility) / stdev;
      *weight = *weight * raise(N::from(value_weight_cdf(z.to_f64().unwrap())).unwrap() + offset);
      new_total = new_total + *weight;
    }
    if new_total <= N::zero() {
      return;
    }
    // Restore the original total so that only the distribution across children
    // changes and the node's own `weight_sum` is unaffected.
    let factor = total_weight / new_total;
    for (_, weight, _, _) in children.iter_mut() {
      *weight = *weight * factor;
    }
  }

  /// [`Self::downweight_bad_children`] for an exponent other than 0.5 or 0.25.
  fn downweight_bad_children_powf(children: &mut [Child<N>], total_weight: N, mean_utility: N, exponent: f64) {
    let min_variance = N::from(1e-8).unwrap();
    let precision_scale = N::from(1.5).unwrap();
    let offset = N::from(0.0001).unwrap();
    let exponent = N::from(exponent).unwrap();
    let mut new_total = N::zero();
    for (_, weight, utility, _) in children.iter_mut() {
      let stdev = (min_variance + (precision_scale * weight.sqrt()).recip()).sqrt();
      let z = (*utility - mean_utility) / stdev;
      *weight = *weight * (N::from(value_weight_cdf(z.to_f64().unwrap())).unwrap() + offset).powf(exponent);
      new_total = new_total + *weight;
    }
    if new_total <= N::zero() {
      return;
    }
    let factor = total_weight / new_total;
    for (_, weight, _, _) in children.iter_mut() {
      *weight = *weight * factor;
    }
  }

  /// Recomputes a node's visit count, total weight and MCTS utility from its
  /// children, applying value downweighting and subtree value bias correction.
  ///
  /// This is the recurrence
  /// `MCTSUtility(n) = (NodeUtility(n) * weight(n) + sum_c MCTSUtility(c) * Weight(c)) / W(n)`
  /// where `NodeUtility(n) = NNUtility(n) + lambda * ObsBias(bucket(n))`. As a
  /// side effect, the node's bucket is updated with its freshly observed error
  /// `ChildrenUtility(n) - NNUtility(n)` before that bias is retrieved back.
  pub(crate) fn update_node(
    map: &HashMap<Hash, usize>,
    nodes: &mut [Node<N>],
    bias: &mut std::collections::HashMap<BiasKey, BiasEntry<N>>,
    node_idx: usize,
    params: Params,
    // Scratch space for the children, reused across a whole backup so that the
    // pass below costs no allocation per node.
    children: &mut Vec<Child<N>>,
  ) {
    let mut sum_visits = 0;
    // Every visited child, collected first because both reweightings below need
    // to see all of them before any child's weight is final.
    children.clear();
    let mut sum_weights = N::zero();

    for edge in nodes[node_idx].children.iter() {
      // Unvisited edges contribute nothing (zero weight in both sums), so skip
      // them to avoid the hash lookup and conversion for the (often many)
      // children that have never been traversed.
      if edge.visits == 0 {
        continue;
      }
      if let Some(&child_idx) = map.get(&edge.hash) {
        let child = &nodes[child_idx];
        let edge_weight = Self::child_weight(child, edge.visits);
        sum_weights = sum_weights + edge_weight;
        children.push((child_idx, edge_weight, -child.value, edge.prior));
      }
      sum_visits += edge.visits;
    }

    // ChildWeight(n) = W(n) - weight(n), the weight of all the search below this
    // node before any of the corrections below touch it. Kept because the bias
    // bucket weights a node by how much search stands behind its observed error,
    // which is the search that happened rather than the part of it the value
    // ends up trusting.
    let searched_weight = sum_weights;

    // Discard weight that exploration put on refuted moves against the policy's
    // advice; unlike the downweighting below, this lowers the node's total.
    sum_weights = Self::prune_noise_weight(children, sum_weights, params.noise_prune_utility_scale);

    // Downweight children whose value sits far below the others before averaging
    // them in, so that a single bad line explored deeply cannot drag the node's
    // value down as much as a genuinely even position would. The weights are
    // renormalized back to the same total, so only the distribution across
    // children changes and `weight_sum` is untouched.
    Self::downweight_bad_children(children, sum_weights, params.value_weight_exponent);

    let mut sum_values = N::zero();
    let mut sum_values_sq = N::zero();
    let mut sum_weights_sq = N::zero();
    for &(child_idx, weight, _, _) in children.iter() {
      let child = &nodes[child_idx];
      sum_values = sum_values + weight * -child.value;
      sum_values_sq = sum_values_sq + weight * child.value_sq;
      // Taking only `weight / child.weight_sum` of the child's weight scales
      // every weight inside it by that factor, so the squares scale by its
      // square.
      if child.weight_sum > N::zero() {
        let scaling = weight / child.weight_sum;
        sum_weights_sq = sum_weights_sq + scaling * scaling * child.weight_sq_sum;
      }
    }

    // NodeUtility starts as the raw neural net utility and is then corrected
    // towards the observed bias of this node's bucket.
    let raw_value = nodes[node_idx].raw_value;
    let mut node_utility = raw_value;

    if let Some(key) = nodes[node_idx].bias_key {
      if sum_weights > N::zero() {
        // ChildrenUtility from this node's perspective. `sum_values` already
        // holds `-sum_c value(c) * weight(c)`, i.e. the children's utility from
        // the parent's perspective times their weights.
        let children_utility = sum_values / sum_weights;
        let weight = searched_weight.powf(N::from(Self::BIAS_ALPHA).unwrap());
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
    node.visits = node.own_visits + sum_visits;
    node.weight_sum = node.weight + sum_weights;
    node.weight_sq_sum = node.weight_sq + sum_weights_sq;
    // The bias-corrected utility enters both moments so that the implied
    // variance stays consistent with the value. The node's own evaluation
    // counts with its uncertainty weight, like every child contribution.
    node.value = (node_utility * node.weight + sum_values) / node.weight_sum;
    node.value_sq = (node_utility * node_utility * node.weight + sum_values_sq) / node.weight_sum;
  }

  /// Recomputes every reachable node from its children, bottom up, refreshing the
  /// subtree value bias corrections baked into their values.
  ///
  /// A reused tree holds values computed against the bias buckets as they stood
  /// when each node was last visited, and those buckets keep moving as the rest of
  /// the search feeds them - so on reuse the retained values are stale, and each
  /// node's recorded contribution to its bucket is stale with them.
  ///
  /// Note that one bottom-up pass is not a fixed point: recomputing a node updates
  /// its bucket, which shifts the correction of every other node sharing that
  /// bucket, including ones already visited.
  pub(crate) fn recompute_stats(&mut self) {
    self.stats_stale = false;
    if Self::BIAS_LAMBDA == 0.0 {
      return;
    }

    // Depth-first post-order, so a node is recomputed only once all of the
    // children it aggregates already have been. Transpositions make this a graph
    // rather than a tree, so nodes are emitted once; it cannot contain a cycle,
    // since every edge plays a move and so strictly grows the position.
    let mut visited = vec![false; self.nodes.len()];
    let mut stack = vec![(self.root_idx, 0usize)];
    let mut order = Vec::new();
    visited[self.root_idx] = true;
    while let Some((node_idx, cursor)) = stack.pop() {
      match self.nodes[node_idx].children.get(cursor) {
        Some(edge) => {
          stack.push((node_idx, cursor + 1));
          if edge.visits > 0
            && let Some(&child_idx) = self.map.get(&edge.hash)
            && !visited[child_idx]
          {
            visited[child_idx] = true;
            stack.push((child_idx, 0));
          }
        }
        None => order.push(node_idx),
      }
    }

    let params = self.params;
    let mut children = Vec::new();
    for node_idx in order {
      // A node the root was just moved into may not have been evaluated yet, and
      // has no value to rebuild from.
      if self.nodes[node_idx].visits == 0 {
        continue;
      }
      Self::update_node(
        &self.map,
        &mut self.nodes,
        &mut self.bias,
        node_idx,
        params,
        &mut children,
      );
    }
  }

  /// The weight an edge contributes to its parent: the child's average weight
  /// per visit times the visits that came through this edge. A child reached
  /// through several transpositions splits its weight across the edges in
  /// proportion to their visits.
  fn child_weight(child: &Node<N>, edge_visits: u64) -> N {
    if child.visits == 0 {
      return N::zero();
    }
    child.weight_sum * N::from(edge_visits).unwrap() / N::from(child.visits).unwrap()
  }

  /// The squared-weight sum an edge contributes, scaled by the edge's share of
  /// the child's visits. Note that this share enters linearly here while
  /// [`Self::update_node`] scales by its square: this is the squared-weight sum
  /// of a *subsample* of the child's playouts, so it shrinks proportionally to
  /// how many of them the edge accounts for.
  fn child_weight_sq(child: &Node<N>, edge_visits: u64) -> N {
    if child.visits == 0 {
      return N::zero();
    }
    child.weight_sq_sum * N::from(edge_visits).unwrap() / N::from(child.visits).unwrap()
  }

  /// Hyperparameter for forced playouts at the root with Dirichlet noise.
  /// nforced(c) = sqrt(k * P(c) * total_child_weight)
  /// When a root child has weight > 0 but below nforced(c), its PUCT score is
  /// set to infinity to ensure it receives enough exploration.
  const FORCED_PLAYOUTS_K: u32 = 2;

  /// How strongly evaluations count relative to their predicted short-term
  /// value error: an evaluation's weight is
  /// `coeff / (error + coeff / max_weight)`, so a perfectly certain
  /// evaluation weighs `UNCERTAINTY_MAX_WEIGHT` and uncertain ones weigh
  /// less.
  ///
  /// The coefficient is the error at which an evaluation weighs about one, so it
  /// has to sit near the error a typical evaluation actually predicts. That is
  /// what keeps the total weight behind a node comparable to its playout count,
  /// and every constant that is in weight units - the `1` the exploration term
  /// divides by, `CPUCT_EXPLORATION_BASE`, a virtual loss and its `0.25` floor,
  /// `CHOSEN_MOVE_PRUNE` - depends on that. It also fixes what
  /// `UNCERTAINTY_MAX_WEIGHT` is a multiple of: an exact terminal is meant to
  /// outweigh a typical evaluation by that factor, not by more.
  ///
  /// Measured over 143k evaluations of a trained net searching at 800 playouts a
  /// move, the median predicted error is 0.51, and it grows with the board: 0.31
  /// at 16x16, 0.53 at 20x20, 0.66 at 24x24. The value below leaves the median
  /// evaluation weighing 0.87, and between 1.33 and 0.70 across those sizes - one
  /// digit is all the precision there is to have while a single constant covers
  /// every board.
  ///
  /// Note how far this is from what a game with a narrower value swing needs: an
  /// error of 0.5 on a value spanning `[-1, 1]` is a net that is barely committed,
  /// and a coefficient tuned for a confident one would put every evaluation here
  /// at a fraction of a playout's weight.
  pub(crate) const UNCERTAINTY_COEFF: f64 = 0.5;
  pub(crate) const UNCERTAINTY_MAX_WEIGHT: f64 = 8.0;

  /// Weight of an evaluation with the given predicted short-term value error
  /// (a standard deviation in utility units).
  pub(crate) fn uncertainty_weight(uncertainty: N) -> N {
    let coeff = N::from(Self::UNCERTAINTY_COEFF).unwrap();
    let baseline = coeff / N::from(Self::UNCERTAINTY_MAX_WEIGHT).unwrap();
    coeff / (uncertainty.max(N::zero()) + baseline)
  }

  /// Whether this search weighs the playouts of `model` by their predicted
  /// error, which takes both the parameters asking for it and a model able to
  /// predict it.
  fn weigh_by_uncertainty<M: Model<N>>(&self, model: &M) -> bool {
    self.params.uncertainty && model.predicts_uncertainty()
  }

  /// Weight to count an evaluation with, or 1 when uncertainty weighting is
  /// off and every playout counts the same.
  fn eval_weight(weigh: bool, uncertainty: N) -> N {
    if weigh {
      Self::uncertainty_weight(uncertainty)
    } else {
      N::one()
    }
  }

  /// Weight of a terminal node, whose value is exact.
  fn terminal_weight(weigh: bool) -> N {
    if weigh {
      N::from(Self::UNCERTAINTY_MAX_WEIGHT).unwrap()
    } else {
      N::one()
    }
  }

  /// Typical utility standard deviation of a node; the observed stdev is
  /// measured relative to it.
  ///
  /// This only ever enters as the ratio `stdev / prior`, so what matters is not
  /// the range of the utility but whether it matches the standard deviation
  /// actually observed in this game: match it and a typical node's exploration is
  /// left alone, set it too high and exploration is damped everywhere.
  ///
  /// Measured over a trained net playing full games at 800 playouts a move, the
  /// value that leaves the median node's exploration untouched is about 0.26 on a
  /// full size board, so this damps it by under a tenth there. It depends strongly
  /// on the board, though - around 0.18 at 20x20 and 0.08 at 12x12, small boards
  /// being much the quieter - and hardly at all on how far into the game the
  /// position is.
  const CPUCT_UTILITY_STDEV_PRIOR: f64 = 0.30;
  /// Weight of the prior when blending it with the observed utility variance.
  const CPUCT_UTILITY_STDEV_PRIOR_WEIGHT: f64 = 2.0;
  /// Exploration scaling from the node's observed utility standard deviation,
  /// estimated from the value and squared-value moments blended with a prior
  /// towards `CPUCT_UTILITY_STDEV_PRIOR`. The PUCT coefficient is scaled by
  /// `1 + scale * (stdev / prior - 1)`, exploring more under volatile nodes and
  /// less under quiet ones; a scale of 0 leaves it at 1.
  pub(crate) fn utility_stdev_factor(&self, node: &Node<N>) -> N {
    let scale = N::from(self.params.cpuct_utility_stdev_scale).unwrap();
    if scale == N::zero() {
      return N::one();
    }
    let prior = N::from(Self::CPUCT_UTILITY_STDEV_PRIOR).unwrap();
    // The variance is only defined once there is more than one unit of weight
    // behind it, which is also what keeps the denominator below positive.
    let stdev = if node.weight_sum <= N::one() {
      prior
    } else {
      let prior_weight = N::from(Self::CPUCT_UTILITY_STDEV_PRIOR_WEIGHT).unwrap();
      let weight_sum = node.weight_sum;
      let utility_sq = node.value * node.value;
      // Guard against numerical imprecision producing negative variance.
      let utility_sq_avg = node.value_sq.max(utility_sq);
      (((utility_sq + prior * prior) * prior_weight + utility_sq_avg * weight_sum)
        / (prior_weight + weight_sum - N::one())
        - utility_sq)
        .max(N::zero())
        .sqrt()
    };
    N::one() + scale * (stdev / prior - N::one())
  }

  /// Weight at which the logarithmic growth of the exploration coefficient
  /// starts, i.e. the scale the total child weight is measured against.
  const CPUCT_EXPLORATION_BASE: f64 = 500.0;

  /// The whole coefficient the PUCT exploration term is scaled by:
  /// `cpuct(W) * sqrt(W + 0.01) * utility_stdev_factor`, for `W` the total weight
  /// behind the node's children.
  ///
  /// `cpuct(W) = cpuct_exploration + cpuct_exploration_log * ln((W + base) / base)`
  /// grows slowly with the search. Without that growth the `sqrt(W)` factor stops
  /// keeping up with the `1 / (1 + W(a))` in the denominator as a node is searched
  /// harder, and the node narrows onto its current best child earlier than it
  /// should.
  ///
  /// The small offset keeps the term positive when no child carries weight yet.
  ///
  /// Policy target pruning inverts this exact expression to recover the weight
  /// PUCT would have allocated, so both go through here rather than rebuilding it.
  pub(crate) fn explore_scaling(&self, total_child_weight: N, node: &Node<N>) -> N {
    let base = N::from(Self::CPUCT_EXPLORATION_BASE).unwrap();
    let cpuct = N::from(self.params.cpuct_exploration).unwrap()
      + N::from(self.params.cpuct_exploration_log).unwrap() * ((total_child_weight + base) / base).ln();
    cpuct * (total_child_weight + N::from(0.01).unwrap()).sqrt() * self.utility_stdev_factor(node)
  }

  /// Total weight currently behind a node's children, summed fresh from them.
  ///
  /// This is deliberately not the cached `weight_sum - weight`: that was computed
  /// the last time this node was recomputed, and a transposition can update one
  /// of its children through another path without touching it.
  pub(crate) fn total_child_weight(&self, node: &Node<N>) -> N {
    node.children.iter().map(|edge| self.edge_child_weight(edge)).sum()
  }

  fn select_edge(&self, node_idx: usize, is_root: bool) -> Option<usize> {
    let noise = is_root && self.dirichlet_noise;
    let node = &self.nodes[node_idx];

    // Resolve each child once, up front. The exploration coefficient needs the
    // total child weight before any child can be scored, so this has to be a
    // separate pass over the children - but the resolved child and its weight
    // are kept so that the scoring pass below needs no further map lookups.
    let mut resolved = Vec::with_capacity(node.children.len());
    let mut total_child_weight = N::zero();
    for edge in node.children.iter() {
      let child = if edge.visits > 0 {
        self.map.get(&edge.hash).map(|&child_idx| &self.nodes[child_idx])
      } else {
        None
      };
      let child_weight = child.map_or(N::zero(), |child| Self::child_weight(child, edge.visits));
      total_child_weight = total_child_weight + child_weight;
      resolved.push((child, child_weight));
    }

    let mut best_score = -N::infinity();
    let mut best = None;

    let c_fpu = N::from(if is_root {
      self.params.root_fpu_reduction_max
    } else {
      Self::FPU_REDUCTION_MAX
    })
    .unwrap();
    let forced_k = N::from(Self::FORCED_PLAYOUTS_K).unwrap();
    let puct_coeff = self.explore_scaling(total_child_weight, node);

    let prior_visited = LazyCell::new(|| {
      node
        .children
        .iter()
        .filter(|edge| edge.visits > 0)
        .map(|edge| edge.prior)
        .sum()
    });

    for ((idx, edge), &(child, child_weight)) in node.children.iter().enumerate().zip(resolved.iter()) {
      let mut q = match child {
        // Child value is from child's perspective.
        // Parent wants to maximize own value, which is -child.value
        Some(child) if child_weight > N::zero() => -child.value,
        // FPU: the parent's utility reduced by `c_fpu * sqrt(visited policy mass)`,
        // With little mass visited yet the parent's running average rests on few
        // playouts, so it is blended towards the raw net value with the weight
        // `min(1, mass²)`. An edge whose child carries no weight yet also lands
        // here.
        _ => {
          let mass: N = *prior_visited;
          let parent_weight = (mass * mass).min(N::one());
          let parent_value = node.raw_value + (node.value - node.raw_value) * parent_weight;
          parent_value - c_fpu * mass.sqrt()
        }
      };

      // Virtual losses steer concurrent playouts down different paths by
      // pulling the utility towards a loss (the utility range radius is 1) and
      // adding their weight. The floor on the child's weight keeps a single
      // uncertain evaluation of weight below 0.25 from being wiped out by one
      // virtual loss.
      let mut edge_weight = child_weight;
      if edge.virtual_losses > 0 {
        let virtual_losses = N::from(edge.virtual_losses).unwrap();
        let virtual_loss_frac = virtual_losses / (virtual_losses + child_weight.max(N::from(0.25).unwrap()));
        q = q + (-N::one() - q) * virtual_loss_frac;
        edge_weight = edge_weight + virtual_losses;
      }
      let p = edge.prior;

      // PUCT formula
      // Score = Q(a) + C * P(a) * sqrt(sum(W)) / (W(a) + 1)
      let score = q + puct_coeff * p / (edge_weight + N::one());

      // Forced playouts. The weight judged here includes the virtual losses,
      // which is what stops the whole batch from piling into one child: forcing
      // ignores the score entirely, so a virtual loss can only steer the next
      // readout elsewhere by pushing this child over its own threshold.
      let score = if noise && child_weight > N::zero() {
        let nforced = (forced_k * p * total_child_weight).sqrt();
        if edge_weight < nforced { N::infinity() } else { score }
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
    let mut is_root = true;
    let mut path = Vec::new();
    let mut terminal = self.nodes[idx].visits > 0;

    while let Some(edge_idx) = self.select_edge(idx, is_root) {
      is_root = false;
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

  fn add_result(
    &mut self,
    path: &[(usize, usize)],
    result: N,
    weight: N,
    children: Vec<Edge<N>>,
    bias_key: Option<BiasKey>,
  ) {
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
    // The root stays out of the bias table entirely; see
    // [`Self::detach_root_bias`] for why.
    let bias_key = if leaf_idx == self.root_idx { None } else { bias_key };
    // The leaf has no children yet, so it cannot update its bucket, but it does
    // immediately retrieve the bucket's current observed bias to correct its own
    // utility. The corrected utility enters both moments, as in `update_node`.
    let obs_bias = bias_key.map_or(N::zero(), |key| Self::retrieve_bias(&self.bias, &key));
    let value = result + N::from(Self::BIAS_LAMBDA).unwrap() * obs_bias;
    let leaf = &mut self.nodes[leaf_idx];
    if leaf.own_visits == 0 {
      leaf.raw_value = result;
      leaf.visits = 1;
      leaf.own_visits = 1;
      leaf.weight = weight;
      // The leaf's single evaluation is the only weight behind it, so it is also
      // the only square in its squared-weight sums.
      leaf.weight_sq = weight * weight;
      leaf.weight_sum = weight;
      leaf.weight_sq_sum = weight * weight;
      leaf.children = children;
      leaf.bias_key = bias_key;
      leaf.last_bias_delta = N::zero();
      leaf.last_bias_weight = N::zero();
      leaf.value = value;
      leaf.value_sq = value * value;
    } else {
      // This node has been evaluated before, so it is a childless one - terminal,
      // or with every move forbidden - which has no edge to descend and is
      // therefore reached as a leaf again by every playout that gets to it. A node
      // with children is expanded once and descended through afterwards, and a
      // batch never evaluates one position twice.
      //
      // This is another playout's worth of evidence about the same state, so it
      // accumulates rather than replacing what is already there. Overwriting would
      // pin `visits` and the weights at one evaluation while the parent edges kept
      // counting, leaving `weight_sq_sum` - and so the effective sample size the
      // LCB divides its variance by - inflated by the number of times the node was
      // re-entered. `children` and `bias_key` are left alone: the state is the same
      // one, so the first expansion's are already right.
      let old_weight_sum = leaf.weight_sum;
      let new_weight_sum = old_weight_sum + weight;
      leaf.raw_value = (leaf.raw_value * old_weight_sum + result * weight) / new_weight_sum;
      leaf.value = (leaf.value * old_weight_sum + value * weight) / new_weight_sum;
      leaf.value_sq = (leaf.value_sq * old_weight_sum + value * value * weight) / new_weight_sum;
      leaf.visits += 1;
      leaf.own_visits += 1;
      leaf.weight = leaf.weight + weight;
      leaf.weight_sq = leaf.weight_sq + weight * weight;
      leaf.weight_sum = new_weight_sum;
      leaf.weight_sq_sum = leaf.weight_sq_sum + weight * weight;
    }
    let params = self.params;
    let mut children = Vec::new();
    for &(node_idx, _) in path.iter().rev() {
      Self::update_node(
        &self.map,
        &mut self.nodes,
        &mut self.bias,
        node_idx,
        params,
        &mut children,
      );
    }
  }

  /// Computes the subtree value bias bucket for a leaf node from the field state
  /// at the leaf (with all of the path's moves played). Returns `None` when
  /// there is no preceding move to key on.
  pub(crate) fn bias_key(field: &Field) -> Option<BiasKey> {
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
        if self.params.forbid_bad {
          field.undo();
          continue;
        }
      } else {
        all_bad = false;
      }

      let hash = field.colored_hash(player);
      field.undo();

      if self.params.forbid_bad && field.is_corner(pos) {
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

  /// Re-predicts the priors of a root inherited from a previous search.
  ///
  /// A reused root was expanded as an ordinary leaf, so the priors its edges
  /// hold were interpolated at [`Params::policy_optimism`]. When the root wants
  /// a different optimism, ask the net about this one position again and swap
  /// the edge priors in place, keeping the children and everything the tree has
  /// learned about them. Dots are only ever added, so a position cannot repeat
  /// within a game and no path below the root can transpose back into this
  /// node: the swap touches the root and nothing else.
  async fn refresh_root_priors<M: Model<N>>(
    &mut self,
    field: &Field,
    player: Player,
    model: &mut M,
    komi_x_2: i32,
  ) -> Result<(), M::E> {
    if self.params.root_policy_optimism == self.params.policy_optimism
      || self.nodes[self.root_idx].children.is_empty()
      // A noised root deliberately does not carry the net's policy, so there is
      // nothing to refresh towards. Only self-play noises roots, and it keeps
      // the two optimisms equal, so this does not come up today.
      || self.dirichlet_noise
    {
      return Ok(());
    }

    let features = field_features::<N>(field, player, field.width(), field.height(), 0).insert_axis(Axis(0));
    let global = global_features(field, player, komi_x_2).insert_axis(Axis(0));
    let optimism = Array::from_elem(1, N::from(self.params.root_policy_optimism).unwrap());
    let (policies, _) = model.predict(features, global, optimism).await?;
    let policy = policies.slice(s![0, .., ..]);

    let stride = field.stride;
    let children = &mut self.nodes[self.root_idx].children;
    for edge in children.iter_mut() {
      let x = to_x(stride, edge.pos);
      let y = to_y(stride, edge.pos);
      edge.prior = policy[(y as usize, x as usize)];
    }
    // Renormalize over the moves the root actually has, as expansion did.
    let sum: N = children.iter().map(|edge| edge.prior).sum();
    if sum > N::zero() {
      for edge in children.iter_mut() {
        edge.prior = edge.prior / sum;
      }
    } else {
      let uniform = N::one() / N::from(children.len()).unwrap();
      for edge in children.iter_mut() {
        edge.prior = uniform;
      }
    }

    Ok(())
  }

  pub async fn mcgs<M: Model<N>, R: Rng>(
    &mut self,
    field: &mut Field,
    player: Player,
    model: &mut M,
    komi_x_2: i32,
    rng: &mut R,
  ) -> Result<(), M::E> {
    if self.stats_stale {
      self.recompute_stats();
    }
    if mem::take(&mut self.root_priors_stale) {
      self.refresh_root_priors(field, player, model, komi_x_2).await?;
    }
    let weigh = self.weigh_by_uncertainty(model);
    let mut leafs = iter::repeat_with(|| self.select_path())
      .take(Self::PARALLEL_READOUTS)
      .collect::<Vec<_>>();
    for (path, _) in &leafs {
      self.revert_virtual_loss(path);
    }

    // Keep one readout per position the batch landed on, and abandon the rest.
    // Two readouts reach the same position either by taking the same path or by
    // transposing onto it, and transpositions are the rule rather than the
    // exception here. Evaluating a position twice would count its own evaluation
    // twice against its subtree for the rest of the search and inflate its
    // squared weight, which deflates the effective sample size behind every edge
    // into it. Abandoning the extra playouts rather than merging them is what
    // keeps a node's visits equal to the playouts that reached it, and that
    // equality is what lets an edge take its share of the node's weight by its
    // share of those visits.
    //
    // Sorting first only fixes which readout of a group survives, so that the
    // batch does not depend on the order the paths were selected in.
    leafs.sort_unstable();
    let mut seen: Vec<Option<Hash>> = Vec::with_capacity(leafs.len());
    leafs.retain(|(path, _)| {
      let leaf = path
        .last()
        .map(|&(node_idx, edge_idx)| self.nodes[node_idx].children[edge_idx].hash);
      if seen.contains(&leaf) {
        false
      } else {
        seen.push(leaf);
        true
      }
    });

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
        // Terminal nodes get no bias correction. Their value is exact, so they
        // get the maximum weight.
        let weight = Self::terminal_weight(weigh);
        self.add_result(
          path,
          game_result(field, player, leaf_komi_x_2),
          weight,
          Vec::new(),
          None,
        );
        false
      } else {
        field_features_to_vec::<N>(
          field,
          player,
          field.width(),
          field.height(),
          0,
          HISTORY_CHANNELS,
          &mut features,
        );
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
    // An empty path is the root itself being expanded, and it asks for the
    // root's own optimism; everything below the root asks for the search-wide
    // one.
    let optimism = Array::from_iter(leafs.iter().map(|(path, _)| {
      N::from(if path.is_empty() {
        self.params.root_policy_optimism
      } else {
        self.params.policy_optimism
      })
      .unwrap()
    }));

    let (policies, values) = model.predict(features, global, optimism).await?;

    for (i, (path, _)) in leafs.iter().enumerate() {
      Self::make_moves(&self.nodes, field, path, player, false);

      let player = if path.len().is_multiple_of(2) {
        player
      } else {
        player.next()
      };

      let policy = policies.slice(s![i, .., ..]);
      let value = values[(i, 0)] - values[(i, 1)];
      let weight = Self::eval_weight(weigh, values[(i, 2)]);

      let children = self.create_children(field, player, &policy, rng);
      // Bucket the leaf by the local context of the move that created it. The
      // field currently has all of the path's moves played, so `field.moves`
      // ends with this node's move and the move before it.
      let bias_key = Self::bias_key(field);
      self.add_result(path, value, weight, children, bias_key);

      for _ in 0..path.len() {
        field.undo();
      }
    }

    Ok(())
  }

  /// Number of standard errors below the mean for the lower confidence bound
  /// used to select the move to play.
  const LCB_STDEVS: f64 = 5.0;

  /// A root child is only eligible for LCB selection once it carries at least
  /// this proportion of the most stably explored child's weight.
  const MIN_WEIGHT_PROP_FOR_LCB: f64 = 0.15;

  /// How far an unvisited move's assumed utility sits below its parent's, scaled
  /// by the square root of the visited policy mass.
  const FPU_REDUCTION_MAX: f64 = 0.2;

  /// Children left below this much weight are dropped from the policy target
  /// entirely, capped at a 64th of the heaviest child's weight so that the
  /// threshold stays meaningful when the whole search is small.
  ///
  /// Because [`Self::reduced_weights`] rounds up, every child that survives the
  /// reduction carries at least one, and so this only drops the children reduced
  /// to nothing - the ones PUCT would not have visited at all. Read against
  /// unrounded weights it would instead mean "under one playout's worth", which
  /// deletes the whole thin tail of the target, and with weighted playouts would
  /// not even be a playout's worth.
  const CHOSEN_MOVE_PRUNE: f64 = 1.0;

  /// Lower confidence bound of a root child's value from the root player's
  /// perspective: Q(a) minus `LCB_STDEVS` standard errors. The variance comes
  /// from the second moment propagated through the graph together with the
  /// value, so both describe the same estimate. To behave well at low playout
  /// counts a prior that the variance is the largest possible (the values span
  /// [-1, 1], a range radius of 1) is mixed in with a small weight, which
  /// diminishes as the evidence grows.
  ///
  /// Returns the bound together with the radius it was shaved by; the radius is
  /// what lets [`Self::pruned_weights`] judge how much wider a child's confidence
  /// interval would have to be before it overtook the best one.
  fn edge_lcb(&self, edge: &Edge<N>) -> Option<(N, N)> {
    let &child_idx = self.map.get(&edge.hash)?;
    let child = &self.nodes[child_idx];
    if child.visits == 0 {
      return None;
    }
    let mut weight_sum = Self::child_weight(child, edge.visits);
    let mut weight_sq_sum = Self::child_weight_sq(child, edge.visits);
    if weight_sum <= N::zero() || weight_sq_sum <= N::zero() {
      return None;
    }
    let value_sq = child.value_sq.max(child.value * child.value + N::from(1e-8).unwrap());
    // Unequally weighted playouts carry less independent evidence than their
    // count suggests, so the standard error is scaled by the effective sample
    // size `W² / sum(w²)` rather than the count. The prior gets weight
    // `W / ess³`, which shrinks as the evidence accumulates.
    let ess = weight_sum * weight_sum / weight_sq_sum;
    let prior_weight = weight_sum / (ess * ess * ess);
    let value_sq = (value_sq * weight_sum + (value_sq + N::one()) * prior_weight) / (weight_sum + prior_weight);
    weight_sum = weight_sum + prior_weight;
    weight_sq_sum = weight_sq_sum + prior_weight * prior_weight;
    let ess = weight_sum * weight_sum / weight_sq_sum;
    let variance = value_sq - child.value * child.value;
    let radius = N::from(Self::LCB_STDEVS).unwrap() * (variance / ess).sqrt();
    Some((-child.value - radius, radius))
  }

  /// How stably a root child has been explored, used to pick the one that
  /// policy target pruning and LCB eligibility measure against.
  fn stability(&self, edge: &Edge<N>) -> N {
    let visits = N::from(edge.visits).unwrap();
    let discounted = self.edge_child_weight(edge) * (visits - N::one()).max(N::zero()) / visits.max(N::one());
    discounted + N::from(2.0).unwrap() * edge.prior
  }

  /// Index and weight of the most stably explored root child, or `None` when no
  /// child has been explored at all. Note that this is not simply the heaviest
  /// child, and that a child with a single visit contributes no weight to its own
  /// stability.
  pub(crate) fn reference_child(&self) -> Option<(usize, N)> {
    let mut best: Option<(usize, N, N)> = None;
    for (idx, edge) in self.nodes[self.root_idx].children.iter().enumerate() {
      let weight = self.edge_child_weight(edge);
      if weight <= N::zero() {
        continue;
      }
      let stability = self.stability(edge);
      if best.is_none_or(|(_, best_stability, _)| stability > best_stability) {
        best = Some((idx, stability, weight));
      }
    }
    best.map(|(idx, _, weight)| (idx, weight))
  }

  /// The weight a root edge carries, or zero when it has no evaluated child.
  fn edge_child_weight(&self, edge: &Edge<N>) -> N {
    if edge.visits == 0 {
      return N::zero();
    }
    self
      .map
      .get(&edge.hash)
      .map_or(N::zero(), |&idx| Self::child_weight(&self.nodes[idx], edge.visits))
  }

  /// Play selection weight of every root child, in `children` order: the child's
  /// LCB once its weight is enough for the bound to be trusted, otherwise that
  /// weight and its prior. `Either` orders `Left < Right`, so any child with an
  /// LCB outranks all children without one, LCBs compare among themselves, and
  /// when no LCB is available the heaviest child wins with the prior as the
  /// tie-breaker.
  ///
  /// The weight judged here is the *reduced* one, the same quantity the policy
  /// target's own LCB promotion in [`Self::apply_lcb_bonus`] is gated on. Using
  /// the raw weight instead would let a child clear the bar for the move to play
  /// while failing it for the target, so that the move played and the move the
  /// target points at could differ.
  fn play_selection_weights(&self) -> Vec<PlaySelectionWeight<N>> {
    let children = &self.nodes[self.root_idx].children;
    let (reduced, reference_weight) = self
      .reduced_weights()
      .unwrap_or_else(|| (vec![N::zero(); children.len()], N::zero()));
    let min_weight = N::from(Self::MIN_WEIGHT_PROP_FOR_LCB).unwrap() * reference_weight;
    children
      .iter()
      .zip(reduced)
      .map(|(edge, weight)| {
        if weight > N::zero()
          && weight >= min_weight
          && let Some((lcb, _)) = self.edge_lcb(edge)
        {
          Either::Right(lcb)
        } else {
          Either::Left((weight, edge.prior))
        }
      })
      .collect()
  }

  /// The root child to play: the child with the best play selection weight.
  /// LCB avoids playing a move whose high value rests on too little evidence to
  /// be trusted.
  ///
  /// Ties go to the earliest child, which is how [`Self::apply_lcb_bonus`] breaks
  /// them too. Two children can easily hold the very same bound early on, and
  /// were the two to break that tie in opposite directions the move played would
  /// not be the one the policy target was promoted to peak at.
  fn best_edge(&self) -> Option<&Edge<N>> {
    let mut best: Option<(usize, PlaySelectionWeight<N>)> = None;
    for (idx, weight) in self.play_selection_weights().into_iter().enumerate() {
      if best
        .as_ref()
        .is_none_or(|(_, best_weight)| weight.partial_cmp(best_weight) == Some(std::cmp::Ordering::Greater))
      {
        best = Some((idx, weight));
      }
    }
    best.map(|(idx, _)| &self.nodes[self.root_idx].children[idx])
  }

  /// Get the best move based on LCB selection
  pub fn best_move(&self) -> Option<NonZeroPos> {
    self.best_edge().and_then(|edge| NonZeroPos::new(edge.pos))
  }

  /// Takes the node that has just become the root out of the subtree value bias
  /// table.
  ///
  /// The root does not belong in it. Its bucket is keyed on its own last move, and
  /// no node of its subtree can repeat that move, so the bucket only ever holds
  /// the root itself: the correction it reads back is then its own observed error,
  /// and mixing that into its utility is pure self-feedback on the value the search
  /// reports and on the root's own exploration through
  /// [`Self::utility_stdev_factor`].
  ///
  /// Its contribution is not simply erased but decayed by `BIAS_FREE_PROP`, the
  /// same way [`Self::compact`] releases the contribution of a node that leaves the
  /// tree: the bias of a tactic should carry over partially rather than vanish the
  /// moment the search moves into it.
  fn detach_root_bias(&mut self) {
    let root = &mut self.nodes[self.root_idx];
    let Some(key) = root.bias_key.take() else {
      return;
    };
    let (delta, weight) = (root.last_bias_delta, root.last_bias_weight);
    root.last_bias_delta = N::zero();
    root.last_bias_weight = N::zero();
    if let Some(entry) = self.bias.get_mut(&key) {
      let free_prop = N::from(Self::BIAS_FREE_PROP).unwrap();
      entry.delta_sum = entry.delta_sum - delta * free_prop;
      entry.weight_sum = entry.weight_sum - weight * free_prop;
    }
  }

  /// Move the root to the best child
  pub fn next_best_root(&mut self) -> Option<NonZeroPos> {
    self.dirichlet_noise = false;
    if let Some((edge_hash, edge_pos)) = self.best_edge().map(|edge| (edge.hash, edge.pos)) {
      self.root_idx = self.add_node(edge_hash);
      self.detach_root_bias();
      self.stats_stale = true;
      self.root_priors_stale = true;
      NonZeroPos::new(edge_pos)
    } else {
      *self = Self::new(self.params);
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
      self.detach_root_bias();
      self.stats_stale = true;
      self.root_priors_stale = true;
      true
    } else {
      *self = Self::new(self.params);
      false
    }
  }

  /// Reset the search to a fresh empty tree, dropping all reused state
  /// including the subtree value bias buckets.
  pub fn clear(&mut self) {
    *self = Self::new(self.params);
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
      stats_stale: self.stats_stale,
      root_priors_stale: self.root_priors_stale,
      params: self.params,
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

    // A bucket is only ever read or updated through the `bias_key` of a live
    // node, so entries whose key no surviving node carries are garbage: drop
    // them instead of letting the map grow forever.
    let live_keys = new_search
      .nodes
      .iter()
      .filter_map(|node| node.bias_key)
      .collect::<std::collections::HashSet<_>>();
    new_search.bias.retain(|key, _| live_keys.contains(key));

    // Surviving nodes were removed from `self.map` above; whatever remains are
    // the nodes being dropped. Decay their contribution to their buckets, so
    // that the bias of a reused tactic carries over only partially rather than
    // lingering at full strength forever.
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

  /// The value the search settled on for every explored root child, from the
  /// root player's perspective, together with the weight of search behind it:
  /// `(pos, weight, q)`.
  ///
  /// This is the per-move q training target: what a single forward pass should
  /// say the search would conclude about each reply. The weights are the raw
  /// ones - unlike the policy target, which prunes the weight forced
  /// exploration spent, a q value is not a preference, so all of the evidence
  /// behind it is wanted.
  pub fn q_values(&self) -> impl Iterator<Item = (Pos, N, N)> + '_ {
    self.nodes[self.root_idx].children.iter().filter_map(|edge| {
      if edge.visits == 0 {
        return None;
      }
      let &child_idx = self.map.get(&edge.hash)?;
      let child = &self.nodes[child_idx];
      let weight = Self::child_weight(child, edge.visits);
      if weight > N::zero() {
        Some((edge.pos, weight, -child.value))
      } else {
        None
      }
    })
  }

  /// Get the weight and prior for each child of the root node
  pub fn weights_with_prior(&self) -> impl Iterator<Item = (Pos, (N, N))> + '_ {
    self.nodes[self.root_idx]
      .children
      .iter()
      .map(|edge| (edge.pos, (self.edge_child_weight(edge), edge.prior)))
  }

  /// Get the play selection weight for each child of the root node: the LCB
  /// when available, otherwise weight and prior. Consumers playing the
  /// max-weight move pick the same child as `best_move`.
  pub fn play_selection(&self) -> Vec<(Pos, PlaySelectionWeight<N>)> {
    self.nodes[self.root_idx]
      .children
      .iter()
      .zip(self.play_selection_weights())
      .map(|(edge, weight)| (edge.pos, weight))
      .collect()
  }

  /// Get the weight of each child of the root node, the policy training target
  /// of a search without forced playouts to remove.
  pub fn weights(&self) -> impl Iterator<Item = (Pos, N)> + '_ {
    self
      .weights_with_prior()
      .map(|(pos, (weight, _))| (pos, weight))
      .filter(|(_, weight)| *weight > N::zero())
  }

  /// Reduced weight of every root child, in `children` order and zero for the
  /// children no playout reached, together with the weight of the reference
  /// child that the reduction is measured against.
  ///
  /// This is policy target pruning steps 1-3:
  /// 1. Find the reference child c* (the most stably explored one).
  /// 2. Compute PUCT(c*) using final utility estimates.
  /// 3. For each other child c, reduce its weight so that PUCT(c) does not
  ///    exceed PUCT(c*).
  ///
  /// The reduction is unbounded: a child is taken all the way down to the weight
  /// PUCT would have given it, which can be less than the forced playouts added.
  /// Capping the removal at that many - "subtract up to nforced" - only undoes
  /// the forcing, and leaves behind the playouts the search spent before it
  /// learned that a child was bad. Removing those too is the point of measuring
  /// against the *final* utilities.
  ///
  /// The result is rounded up, which is what makes a reduced weight of zero mean
  /// something: [`Self::CHOSEN_MOVE_PRUNE`] then drops exactly the children PUCT
  /// would not have visited at all, rather than every child left under one
  /// playout's worth of weight.
  ///
  /// Returns `None` when no child has been explored, leaving nothing to measure
  /// against.
  fn reduced_weights(&self) -> Option<(Vec<N>, N)> {
    let root = &self.nodes[self.root_idx];
    let children = &root.children;

    let (best_idx, best_weight) = match self.reference_child() {
      Some(best) if best.1 > N::zero() => best,
      _ => return None,
    };
    let best_edge = &children[best_idx];

    let total_child_weight = self.total_child_weight(root);
    // Invert the very coefficient selection used, so pruning matches what PUCT
    // actually allocated.
    let puct_coeff = self.explore_scaling(total_child_weight, root);

    let best_q = self
      .map
      .get(&best_edge.hash)
      .map_or(N::zero(), |&child_idx| -self.nodes[child_idx].value);

    // Compute PUCT(c*) for the best child
    let best_puct = best_q + puct_coeff * best_edge.prior / (best_weight + N::one());

    let reduced = children
      .iter()
      .enumerate()
      .map(|(idx, edge)| {
        let child_weight = self.edge_child_weight(edge);
        // An unexplored child has nothing to reduce, and the reference child is
        // the yardstick, so it keeps its weight by construction.
        if child_weight <= N::zero() || idx == best_idx {
          return child_weight;
        }

        let child_q = self
          .map
          .get(&edge.hash)
          .map_or(N::zero(), |&child_idx| -self.nodes[child_idx].value);

        // Compute the weight PUCT would have naturally allocated to this child
        // by inverting the PUCT formula:
        //   best_puct = child_q + puct_coeff * P(c) / (W(c) + 1)
        //   W(c) = puct_coeff * P(c) / (best_puct - child_q) - 1
        let explore_component = best_puct - child_q;

        let retrospective_weight = if explore_component <= N::zero() {
          child_weight
        } else {
          (puct_coeff * edge.prior / explore_component - N::one()).max(N::zero())
        };

        // Cap the weight at what PUCT would have allocated, rounded up so that a
        // child it would have visited at all keeps a whole playout's worth of
        // target rather than a fraction. The reference child is exempt, above, so
        // the yardstick the pruning measures against stays exact.
        retrospective_weight.min(child_weight).ceil()
      })
      .collect::<Vec<_>>();

    Some((reduced, best_weight))
  }

  /// The play selection value of every root child left with a share of the
  /// policy target, paired with the child's index and position.
  ///
  /// This is [`Self::reduced_weights`] followed by policy target pruning steps 4
  /// and 5: promote the best lower confidence bound, then drop the children left
  /// with a negligible share.
  fn play_selection_values(&self) -> Vec<(usize, Pos, N)> {
    let Some((mut reduced, reference_weight)) = self.reduced_weights() else {
      return Vec::new();
    };

    // Promote the child with the best lower confidence bound, so that the move
    // the search would actually play is also the one the target points at.
    self.apply_lcb_bonus(&mut reduced, reference_weight);

    // Drop the children left with a negligible share of the target, which is
    // mostly those that only forced exploration ever visited. The threshold is
    // relative to the heaviest remaining child so that it stays meaningful at
    // any number of playouts.
    let max_weight = reduced.iter().copied().fold(N::zero(), N::max);
    let prune_below = N::from(Self::CHOSEN_MOVE_PRUNE)
      .unwrap()
      .min(max_weight / N::from(64.0).unwrap());
    let children = &self.nodes[self.root_idx].children;
    reduced
      .into_iter()
      .enumerate()
      .filter(|&(_, weight)| weight > N::zero() && weight >= prune_below)
      .map(|(idx, weight)| (idx, children[idx].pos, weight))
      .collect()
  }

  /// Get pruned weights for the policy target: the play selection values, which
  /// decouple the target from the forced exploration playouts the search spent -
  /// and which the move to play is drawn from, so the two agree.
  pub fn pruned_weights(&self) -> impl Iterator<Item = (Pos, N)> {
    self
      .play_selection_values()
      .into_iter()
      .map(|(_, pos, weight)| (pos, weight))
  }

  /// Raises the weight of the root child with the best lower confidence bound
  /// until it outranks every other child.
  ///
  /// For each rival, `radius_factor` asks how many times wider that rival's
  /// confidence interval would have to be before its bound reached the best
  /// one's. Squaring it converts a ratio of standard errors into a ratio of
  /// weights, since a standard error shrinks with the square root of weight. The
  /// `0.20 * excess` in the denominator caps the factor near 5, bounding the
  /// bonus at ~25x however hopeless the rival looks.
  ///
  /// `reduced` holds the reduced weight of every root child in `children` order.
  /// `reference_weight` is the weight of the most stably explored child, from
  /// before the reduction - eligibility is judged against that, while the bonus
  /// itself is computed from the reduced weights.
  fn apply_lcb_bonus(&self, reduced: &mut [N], reference_weight: N) {
    let children = &self.nodes[self.root_idx].children;
    let min_weight = N::from(Self::MIN_WEIGHT_PROP_FOR_LCB).unwrap() * reference_weight;
    let bounds = children.iter().map(|edge| self.edge_lcb(edge)).collect::<Vec<_>>();

    // Only children carrying enough weight to be trusted may claim the bonus.
    let mut best: Option<(usize, N)> = None;
    for (i, (&weight, bound)) in reduced.iter().zip(bounds.iter()).enumerate() {
      if let Some((lcb, _)) = bound
        && weight > N::zero()
        && weight >= min_weight
        && best.is_none_or(|(_, best_lcb)| *lcb > best_lcb)
      {
        best = Some((i, *lcb));
      }
    }
    let Some((best_i, best_lcb)) = best else {
      return;
    };

    let cap = N::from(0.20).unwrap();
    let mut adjusted = reduced[best_i];
    for (i, (&weight, bound)) in reduced.iter().zip(bounds.iter()).enumerate() {
      if i == best_i {
        continue;
      }
      let Some((lcb, radius)) = bound else {
        continue;
      };
      let excess = best_lcb - *lcb;
      // A rival with a better bound that merely failed the weight check is not
      // actually worse, so there is nothing to out-rank it by.
      if excess < N::zero() {
        continue;
      }
      let radius_factor = (*radius + excess) / (*radius + cap * excess);
      adjusted = adjusted.max(radius_factor * radius_factor * weight);
    }
    reduced[best_i] = adjusted;
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
  pub fn policy_surprise(target: &[(Pos, N)], priors: &[N]) -> N {
    let total = target.iter().map(|&(_, weight)| weight).sum::<N>();
    if total <= N::zero() {
      return N::zero();
    }
    // Floor on the prior to avoid `ln(0)` for targets on moves the prior gave a
    // zero probability (and to bound the surprise of such moves).
    let offset = N::from(1e-30).unwrap();
    let mut surprise = N::zero();
    for &(pos, weight) in target {
      if weight <= N::zero() {
        continue;
      }
      let t = weight / total;
      surprise = surprise + t * (t.ln() - (priors[pos] + offset).ln());
    }
    // Guard against tiny negative values from floating point imprecision.
    surprise.max(N::zero())
  }
}

impl<N: Float + Sum + SampleUniform> Search<N> {
  /// Move the root to a random child, sampled from the play selection values -
  /// the same quantity that becomes the policy target, so the forced playouts that
  /// widen the search are taken back out and the LCB bonus is applied before the
  /// move is drawn.
  pub fn next_root_with_temperature<R: Rng>(&mut self, temperature: N, rng: &mut R) -> Option<NonZeroPos> {
    // Pruning leaves only children with a positive value, so every entry here can
    // be sampled and the heaviest defines the logit offset.
    let values = self.play_selection_values();
    let max_weight = values.iter().map(|&(_, _, weight)| weight).fold(N::zero(), N::max);
    if max_weight <= N::zero() {
      return None;
    }
    let max_logit = max_weight.ln();
    let probs = values
      .iter()
      .map(|&(_, _, weight)| ((weight.ln() - max_logit) / temperature).exp())
      .collect::<Vec<_>>();
    let sum_exp: N = probs.iter().copied().sum();

    let mut sample = rng.random_range(N::zero()..sum_exp);
    let mut chosen_edge = None;

    for (&(idx, pos, _), prob) in values.iter().zip(probs) {
      if prob >= sample {
        chosen_edge = Some((self.nodes[self.root_idx].children[idx].hash, pos));
        break;
      } else {
        sample = sample - prob;
      }
    }

    self.dirichlet_noise = false;
    if let Some((hash, pos)) = chosen_edge {
      self.root_idx = self.add_node(hash);
      self.detach_root_bias();
      self.stats_stale = true;
      self.root_priors_stale = true;
      NonZeroPos::new(pos)
    } else {
      *self = Self::new(self.params);
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
