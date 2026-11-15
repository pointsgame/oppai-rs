use crate::field_features::{add_unrotated_policy, field_features, global, rotation_sizes};
use crate::mcgs::{Params, Search};
use crate::model::Model;
use log::info;
use ndarray::{Array1, Array2, Array3, Axis, stack};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use oppai_rotate::rotate::{ROTATIONS, rotate};
use rand::distr::uniform::SampleUniform;
use rand::{Rng, RngExt, seq::SliceRandom};
use rand_distr::{Distribution, Exp, Exp1, Open01, StandardNormal};
use std::fmt::{Debug, Display};
use std::iter::{self, Sum};

/// Root visit budgets of a self-play search. A search runs until its root
/// holds this many visits, counting the subtree reused from previous turns -
/// so a cheap search whose inherited subtree already meets the budget does a
/// single batch and moves on, and the budget is a number of playouts rather
/// than a number of batches of them.
pub(crate) const MCTS_VISITS: u32 = 200;
pub(crate) const MCTS_FULL_VISITS: u32 = 1000;

/// Once the search value has stayed beyond this, towards the same winner, for
/// [`REDUCE_VISITS_LOOKBACK`] consecutive turns, the game is all but decided:
/// full searches ramp their visit budget down towards the cheap budget and
/// their training weight towards [`REDUCED_VISITS_WEIGHT`], both by the square
/// of how far past the threshold the value sits. Playing the tail out cheaply
/// keeps the true final score (and with it every score target) while spending
/// almost nothing on positions whose outcome the net already calls, instead of
/// resigning and corrupting those targets.
const REDUCE_VISITS_THRESHOLD: f64 = 0.9;
const REDUCE_VISITS_LOOKBACK: usize = 3;
pub(crate) const REDUCED_VISITS_WEIGHT: f64 = 0.1;

/// Probability that a turn of a self-play game also searches a side position:
/// the position after a move the game did not play, so that the net learns how
/// to answer moves self-play never makes - it otherwise sees almost no
/// positions right after a mistake, and so learns little about punishing one.
const SIDE_POSITION_PROB: f64 = 0.02;

/// Probability that a searched side position goes on: the move its search
/// chose is played, then another move off the beaten track, and the result is
/// searched in turn - so that some side positions follow several odd moves
/// rather than just one.
const SIDE_POSITION_CONTINUE_PROB: f64 = 0.25;

/// Probability that a finished self-play game forks a new game near its start:
/// the opening gets a move the net would not have played, and the game is
/// played out anew from there. Self-play otherwise follows the net's own
/// preferences, and its games crowd into the openings the net already knows.
const EARLY_FORK_PROB: f64 = 0.04;

/// Probability that a finished self-play game that does not fork near its
/// start forks a new game at a random turn instead, for variety in the middle
/// of the game, where nothing else gets any.
const LATE_FORK_PROB: f64 = 0.01;

/// How far into the game, after the set opening, an early fork goes on
/// average, as a fraction of the area: 1.25% of it, the counterpart of the
/// raw policy opening's 2% - a fork happens about as early as the openings
/// the raw policy plays end.
const EARLY_FORK_AREA_PROP: f64 = 0.0125;

/// The fewest random moves a fork picks the best of, so that the move it plays
/// is a reasonable one rather than any.
const FORK_MIN_CHOICES: usize = 3;

/// The most random moves an early fork picks the best of. Few, so that the
/// opening it plays tends to be an unusual one.
const EARLY_FORK_MAX_CHOICES: usize = 12;

/// The most random moves a late fork picks the best of. More than an early
/// fork, because a random move later in the game is far more likely to be a
/// plain blunder that decides it.
const LATE_FORK_MAX_CHOICES: usize = 36;

/// Visit budget and training weight of a full search, ramped down once the
/// recent search values say the game is decided. `values` holds every turn's
/// search value from Red's perspective.
pub(crate) fn reduced_search(values: &[f64]) -> (u32, f64) {
  let mut visits = f64::from(MCTS_FULL_VISITS);
  let mut weight = 1.0;
  if let Some(window) = values.len().checked_sub(REDUCE_VISITS_LOOKBACK).map(|s| &values[s..]) {
    let min = window.iter().copied().fold(f64::INFINITY, f64::min);
    let max = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Large only when every turn of the window points at the same winner: one
    // turn leaning the other way pulls it right down.
    let extreme = min.max(-max).min(1.0);
    if extreme > REDUCE_VISITS_THRESHOLD {
      let prop = (extreme - REDUCE_VISITS_THRESHOLD) / (1.0 - REDUCE_VISITS_THRESHOLD);
      let prop = prop * prop;
      visits += prop * (f64::from(MCTS_VISITS) - visits);
      weight += prop * (REDUCED_VISITS_WEIGHT - weight);
    }
  }
  (visits.round() as u32, weight)
}

/// Search statistics for a single move played in a self-play game.
///
/// * `.0` - search weight of each explored child of the root (the policy
///   target). Uncertainty weighting makes playouts count unequally, so this is
///   the total weight the search accumulated behind each move rather than the
///   number of playouts. Data recorded before weighting was introduced stores
///   plain visit counts, which is the same thing with every playout weighing
///   one and normalizes to the same target.
///
/// * `.1` - the training weight of the row: 1 for a full search, ramped down
///   towards [`REDUCED_VISITS_WEIGHT`] once the game is all but decided, and 0
///   for a cheap search. A positive weight is what makes the row a training
///   sample; data recorded when this was a boolean full-search flag loads as
///   weight 1 or 0.
/// * `.2` - policy surprise: the KL divergence from the raw root policy prior
///   to the policy training target, used for policy surprise weighting. For
///   cheap searches it decides whether the row earns training weight despite
///   the shallow search; `0` in data recorded before it was stored.
/// * `.3` - the search's value estimate of the position (root Q), in `[-1, 1]`
///   from the perspective of the player to move. Used for value surprise
///   weighting.
/// * `.4` - the raw neural net value of the position, without any search, in
///   `[-1, 1]` from the perspective of the player to move. Used for value
///   surprise weighting.
/// * `.5` - `(pos, weight, q, score)` for every explored child of the root: the
///   value the search settled on for that reply, in `[-1, 1]` from the
///   perspective of the player to move, the score it settled on in points from
///   the same perspective, and the search weight behind them. The per-move q
///   training targets. Empty in data recorded before they were stored, which
///   the loss masks out by the zero weight.
#[derive(Clone, PartialEq, Default, Debug)]
pub struct Visits(
  pub Vec<(Pos, f64)>,
  pub f64,
  pub f64,
  pub f64,
  pub f64,
  pub Vec<(Pos, f64, f64, f64)>,
);

impl Visits {
  pub fn total(&self) -> f64 {
    self.0.iter().map(|&(_, v)| v).sum()
  }

  pub fn max(&self) -> f64 {
    self.0.iter().map(|&(_, v)| v).fold(0.0, f64::max)
  }

  /// Improved stochastic policy values, pushed into an existing vector.
  ///
  /// A target with no weight anywhere is left all zeros rather than smoothed
  /// into some distribution: a zero target contributes no cross-entropy loss,
  /// so a row with nothing recorded teaches the policy nothing instead of
  /// teaching it noise. No healthy data has one - a move is only recorded when
  /// the search chose it, which takes a positively weighted child.
  pub fn policies_to_vec<N: Float + Copy>(
    &self,
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    rotation: u8,
    policies: &mut Vec<N>,
  ) {
    let total = self.total();
    let start_idx = policies.len();

    policies.extend(iter::repeat_n(N::zero(), (width * height) as usize));

    if total > 0.0 {
      for &(pos, weight) in &self.0 {
        let x = to_x(field_width + 1, pos);
        let y = to_y(field_width + 1, pos);
        let (x, y) = rotate(field_width, field_height, x, y, rotation);

        let idx = start_idx + (y as usize) * (width as usize) + (x as usize);
        policies[idx] = N::from(weight).unwrap() / N::from(total).unwrap();
      }
    }
  }

  /// Per-move q targets, pushed as three planes: the q value the search settled
  /// on for each explored child, the score it settled on, then the search
  /// weight behind them. Cells no explored child covers stay zero in every
  /// plane, and the zero weight is what keeps them out of the loss - including
  /// the entire plane of a position recorded before q values were stored.
  pub fn q_values_to_vec<N: Float + Copy>(
    &self,
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    rotation: u8,
    planes: &mut Vec<N>,
  ) {
    let start_idx = planes.len();
    let plane = (width * height) as usize;

    planes.extend(iter::repeat_n(N::zero(), 3 * plane));

    for &(pos, weight, q, score) in &self.5 {
      let x = to_x(field_width + 1, pos);
      let y = to_y(field_width + 1, pos);
      let (x, y) = rotate(field_width, field_height, x, y, rotation);

      let idx = start_idx + (y as usize) * (width as usize) + (x as usize);
      planes[idx] = N::from(q).unwrap();
      planes[idx + plane] = N::from(score).unwrap();
      planes[idx + 2 * plane] = N::from(weight).unwrap();
    }
  }

  /// Improved stochastic policy values.
  pub fn policies<N: Float>(
    &self,
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    rotation: u8,
  ) -> Array2<N> {
    let mut vec = Vec::with_capacity((width * height) as usize);

    self.policies_to_vec(width, height, field_width, field_height, rotation, &mut vec);

    Array2::from_shape_vec((height as usize, width as usize), vec).unwrap()
  }
}

/// A searched position off the main line of a self-play game.
///
/// The game is never played out from it, so it has no result, no final score
/// and no final captures: its only targets are those of its own search, see
/// [`crate::examples::ExampleGame::played_out`].
#[derive(Clone, PartialEq, Debug)]
pub struct SidePosition {
  /// The number of the game's moves before the side line branches off.
  pub branch: usize,
  /// The player making the first move of the side line. The players alternate
  /// from there.
  pub player: Player,
  /// The moves of the side line. The last one is the move the search of the
  /// position before it chose, the way a move of the game follows the search
  /// recorded for it.
  pub moves: Vec<Pos>,
  /// The search of the position before the last move.
  pub visits: Visits,
}

/// A side position still to be searched.
pub(crate) struct SideStart {
  /// The field after the moves of the side line so far.
  pub(crate) field: Field,
  /// The player to move there.
  pub(crate) player: Player,
  /// Komi of the player to move, multiplied by 2.
  pub(crate) komi_x_2: i32,
  pub(crate) branch: usize,
  pub(crate) first_player: Player,
  pub(crate) moves: Vec<Pos>,
}

/// Komi multiplied by 2 from Red's perspective, which is what
/// [`Field::is_game_over`] takes, given that of the player to move - or the
/// other way round, since turning the perspective is its own inverse.
fn red_komi_x_2(player: Player, komi_x_2: i32) -> i32 {
  if player == Player::Red { komi_x_2 } else { -komi_x_2 }
}

/// Picks the move a side line goes on with: a sample of the policy `priors`
/// (indexed by position) at temperature 1 most of the time, so that the side
/// positions are mostly ones a plausible opponent could reach, at temperature 2
/// a quarter of the time and a uniformly random move once in twenty, so that
/// some follow the blunders the policy all but rules out. `banned` is the move the game itself played, which would make the
/// side position one of the game's own.
pub(crate) fn side_move<N: Float, R: Rng>(
  field: &Field,
  priors: &[N],
  banned: Option<Pos>,
  rng: &mut R,
) -> Option<Pos> {
  let r = rng.random::<f64>();
  let temperature = if r < 0.7 {
    Some(1.0)
  } else if r < 0.95 {
    Some(2.0)
  } else {
    None
  };
  let moves = (field.min_pos()..=field.max_pos())
    .filter(|&pos| field.is_putting_allowed(pos) && Some(pos) != banned)
    .collect::<Vec<_>>();
  let weights = moves
    .iter()
    .map(|&pos| temperature.map_or(1.0, |t| priors[pos].to_f64().unwrap().max(0.0).powf(1.0 / t)))
    .collect::<Vec<_>>();
  let total = weights.iter().sum::<f64>();
  // A policy with no mass on any of the moves left leaves only the uniform choice.
  if total <= 0.0 {
    return (!moves.is_empty()).then(|| moves[rng.random_range(0..moves.len())]);
  }
  let mut sample = rng.random_range(0.0..total);
  for (&pos, &weight) in moves.iter().zip(&weights) {
    if weight > sample {
      return Some(pos);
    }
    sample -= weight;
  }
  // Rounding can leave the sample a hair past the last weight.
  moves
    .iter()
    .zip(&weights)
    .rev()
    .find(|&(_, &weight)| weight > 0.0)
    .map(|(&pos, _)| pos)
}

/// Starts a full search of the position: from a fresh tree, with the root
/// expanded, its raw priors saved into `raw_priors` and Dirichlet noise added
/// to them.
async fn start_full_search<N, M, R>(
  search: &mut Search<N>,
  field: &mut Field,
  player: Player,
  model: &M,
  komi_x_2: i32,
  raw_priors: &mut [N],
  rng: &mut R,
) -> Result<(), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  // Recorded searches start from a fresh tree: the Dirichlet noise and
  // forced playouts have to shape the entire visit distribution, and visits
  // inherited from previous searches would leak into the policy target and
  // inflate the policy surprise.
  search.clear();
  // The root has to be expanded before the noise can be applied to its children priors.
  search.mcgs(field, player, model, komi_x_2, rng).await?;
  search.root_priors(raw_priors);
  // Total Dirichlet alpha, matching AlphaZero's 0.03 per move on an empty 19x19 board
  // (0.03 * 361 = 10.83). Kept constant across board sizes and through the game, with
  // the shaping in `add_dirichlet_noise` deciding how it is spread across the moves.
  let total_concentration = N::from(0.03 * 19.0.powi(2)).unwrap();
  let temperature = interpolate_early(field, N::from(1.25).unwrap(), N::from(1.1).unwrap());
  search.add_dirichlet_noise(rng, N::from(0.25).unwrap(), total_concentration, temperature);
  Ok(())
}

/// Runs the search until its root holds `target_visits` visits, and at least
/// one batch even when the reused subtree already meets the budget, so every
/// move rests on a search of the actual root.
async fn run_search<N, M, R>(
  search: &mut Search<N>,
  field: &mut Field,
  player: Player,
  model: &M,
  komi_x_2: i32,
  target_visits: u32,
  rng: &mut R,
) -> Result<(), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  loop {
    search.mcgs(field, player, model, komi_x_2, rng).await?;
    if search.root_visits() >= u64::from(target_visits) {
      return Ok(());
    }
  }
}

/// The search's value of the root, clamped like the per-move q targets: the
/// winloss averaging can drift a hair past the `[-1, 1]` a value is defined on,
/// and the training loss reads the value target as a probability that has to
/// stay in `[0, 1]`.
fn search_value<N: Float + Sum>(search: &Search<N>) -> f64 {
  search.winloss().to_f64().unwrap().clamp(-1.0, 1.0)
}

/// The per-move q targets of the search's root.
fn q_values<N: Float + Sum>(search: &Search<N>) -> Vec<(Pos, f64, f64, f64)> {
  search
    .q_values()
    .map(|(pos, weight, q, score)| {
      (
        pos,
        weight.to_f64().unwrap(),
        q.to_f64().unwrap(),
        score.to_f64().unwrap(),
      )
    })
    .collect()
}

/// Searches the side positions queued in `queue`, and those that they lead to
/// in turn, with full searches.
pub(crate) async fn search_side_positions<N, M, R>(
  mut queue: Vec<SideStart>,
  model: &M,
  readouts: usize,
  rng: &mut R,
) -> Result<Vec<SidePosition>, M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  let mut side_positions = Vec::new();
  while let Some(mut side) = queue.pop() {
    if side.field.is_game_over(red_komi_x_2(side.player, side.komi_x_2)) {
      continue;
    }
    // Searched like a full search of the game, at the full budget: the row is
    // a training sample of weight 1.
    let mut search = Search::new(Params {
      readouts,
      ..Params::SELF_PLAY
    });
    let mut raw_priors = vec![N::zero(); side.field.length()];
    start_full_search(
      &mut search,
      &mut side.field,
      side.player,
      model,
      side.komi_x_2,
      &mut raw_priors,
      rng,
    )
    .await?;
    run_search(
      &mut search,
      &mut side.field,
      side.player,
      model,
      side.komi_x_2,
      MCTS_FULL_VISITS,
      rng,
    )
    .await?;
    let value = search_value(&search);
    let target: Vec<(Pos, N)> = search.pruned_weights().collect();
    let surprise = Search::policy_surprise(&target, &raw_priors).to_f64().unwrap();
    let visits = Visits(
      target
        .into_iter()
        .map(|(pos, weight)| (pos, weight.to_f64().unwrap()))
        .collect(),
      1.0,
      surprise,
      value,
      search.raw_winloss().to_f64().unwrap(),
      q_values(&search),
    );
    let Some(response) = search.next_root_with_temperature(
      interpolate_early(&side.field, N::from(0.75).unwrap(), N::from(0.15).unwrap()),
      rng,
    ) else {
      continue;
    };
    let mut moves = side.moves;
    moves.push(response.get());
    side_positions.push(SidePosition {
      branch: side.branch,
      player: side.first_player,
      moves: moves.clone(),
      visits,
    });

    if rng.random::<f64>() >= SIDE_POSITION_CONTINUE_PROB {
      continue;
    }
    let mut field = side.field;
    assert!(field.put_point(response.get(), side.player));
    field.update_grounded();
    let player = side.player.next();
    let komi_x_2 = -side.komi_x_2;
    if field.is_game_over(red_komi_x_2(player, komi_x_2)) {
      continue;
    }
    // The next odd move is drawn from the raw policy of the position, which a
    // single step of a fresh search expands the root with.
    let mut search = Search::new(Params {
      readouts,
      ..Params::SELF_PLAY
    });
    search.mcgs(&mut field, player, model, komi_x_2, rng).await?;
    let mut priors = vec![N::zero(); field.length()];
    search.root_priors(&mut priors);
    if let Some(pos) = side_move(&field, &priors, None, rng) {
      assert!(field.put_point(pos, player));
      field.update_grounded();
      moves.push(pos);
      queue.push(SideStart {
        field,
        player: player.next(),
        komi_x_2: -komi_x_2,
        branch: side.branch,
        first_player: side.first_player,
        moves,
      });
    }
  }
  Ok(side_positions)
}

/// Interpolates between an early-game and a rest-of-game value, decaying
/// exponentially with the number of moves played. KataGo's interpolateEarly
/// uses a halflife of `sqrt(area)` moves, spending ~7.6% of a typical go game
/// at the early value; dots games fill only ~0.32 of the field, so half that
/// halflife keeps the same fraction of a typical game.
fn interpolate_early<N: Float>(field: &Field, early_value: N, value: N) -> N {
  let halflives = N::from(2 * field.moves_count()).unwrap() / N::from(field.width() * field.height()).unwrap().sqrt();
  value + (early_value - value) * N::from(0.5).unwrap().powf(halflives)
}

/// Samples a move from `policy`, indexed by `(y, x)` of the field.
fn select_policy_move<N, R>(field: &Field, policy: &Array2<N>, rng: &mut R) -> Option<NonZeroPos>
where
  N: Float + Sum + SampleUniform,
  R: Rng,
{
  let mut sum = N::zero();
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      sum = sum + policy[(y as usize, x as usize)];
    }
  }
  let mut sample = rng.random_range(N::zero()..sum);
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      let policy = policy[(y as usize, x as usize)];
      if policy >= sample {
        return NonZeroPos::new(pos);
      } else {
        sample = sample - policy;
      }
    }
  }
  None
}

/// Plays a self-play game out from `field`, opening with a few moves of the raw
/// policy, and returns the searches of its turns and the side positions
/// searched off its main line. `readouts` overrides [`Params::SELF_PLAY`]'s
/// readouts per search call, see [`Params::readouts`].
pub async fn episode<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  mut komi_x_2: i32,
  readouts: usize,
  rng: &mut R,
) -> Result<(Vec<Visits>, Vec<SidePosition>), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  // The number of raw policy opening moves: an exponential with mean
  // `area / 50`, i.e. 2% of the area (the argument is the rate, the inverse of
  // the mean). KataGo's initializeGameUsingPolicy plays 4% of the area,
  // targeting ~6% of a typical go game; dots games fill only ~0.32 of the
  // field, so half of that keeps the same fraction of a typical game.
  let exp = Exp::new(N::from(50).unwrap() / N::from(field.width() * field.height()).unwrap()).unwrap();
  let raw_policy_moves = exp.sample(rng).floor().to_u32().unwrap();

  info!("Playing {} raw policy moves", raw_policy_moves);

  for _ in 0..raw_policy_moves {
    // Under a random symmetry, like every evaluation of the search, so that the
    // openings do not inherit the orientation-specific preferences of the net.
    let rotation = rng.random_range(0..ROTATIONS);
    let (width, height) = rotation_sizes(field.width(), field.height());
    let features = field_features(field, player, width, height, rotation);
    let global = global(field, player, komi_x_2);
    // The opening is sampled from the trained policy: these moves are meant to
    // spread the training positions over the openings the net actually plays.
    let (predicted, _) = model
      .predict(
        features.insert_axis(Axis(0)),
        global.insert_axis(Axis(0)),
        Array1::zeros(1),
      )
      .await?;
    let mut policy = Array2::zeros((field.height() as usize, field.width() as usize));
    add_unrotated_policy(
      predicted.index_axis(Axis(0), 0),
      field.width(),
      field.height(),
      rotation,
      N::one(),
      &mut policy,
    );
    if let Some(pos) = select_policy_move(field, &policy, rng) {
      assert!(field.put_point(pos.get(), player));
      field.update_grounded();
      player = player.next();
      komi_x_2 = -komi_x_2;
    } else {
      break;
    }
  }

  play_out(field, player, model, komi_x_2, readouts, rng).await
}

/// Plays a self-play game out from `field` with searches, see [`episode`].
pub async fn play_out<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  mut komi_x_2: i32,
  readouts: usize,
  rng: &mut R,
) -> Result<(Vec<Visits>, Vec<SidePosition>), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  let mut search = Search::new(Params {
    readouts,
    ..Params::SELF_PLAY
  });
  let mut visits = Vec::new();

  // Raw network policy priors of the root, captured before temperature and
  // Dirichlet noise overwrite them, so the policy surprise is measured against
  // the network's true prior rather than the noised one.
  let mut raw_priors = vec![N::zero(); field.length()];

  // Every turn's search value from Red's perspective, driving the visit
  // reduction of full searches once the game looks decided.
  let mut search_values: Vec<f64> = Vec::new();

  // Side positions to search once the game is over.
  let mut side_starts = Vec::new();

  while !field.is_game_over(red_komi_x_2(player, komi_x_2)) {
    let full_search = rng.random::<f64>() <= 0.25;

    let (target_visits, target_weight) = if full_search {
      start_full_search(&mut search, field, player, model, komi_x_2, &mut raw_priors, rng).await?;
      reduced_search(&search_values)
    } else {
      // Cheap searches keep reusing the tree - they only pick a move.
      (MCTS_VISITS, 0.0)
    };

    run_search(&mut search, field, player, model, komi_x_2, target_visits, rng).await?;

    let value = search_value(&search);
    search_values.push(if player == Player::Red { value } else { -value });

    let target: Vec<(Pos, N)> = if full_search {
      // Use pruned weights for full searches with Dirichlet noise.
      // This removes the extra forced playouts from the policy target,
      // producing a cleaner training signal.
      search.pruned_weights().collect()
    } else {
      search.weights().collect()
    };
    // Policy surprise (KL divergence from the raw policy prior to the policy
    // target) is computed for cheap searches too: one whose surprise stands
    // far above the game's full-search average earns training weight despite
    // the shallow search. Cheap search roots are never noised, so their priors
    // can be snapshotted after the search.
    if !full_search {
      search.root_priors(&mut raw_priors);
    }
    let surprise = Search::policy_surprise(&target, &raw_priors).to_f64().unwrap();
    let current_visits = Visits(
      target
        .into_iter()
        .map(|(pos, weight)| (pos, weight.to_f64().unwrap()))
        .collect(),
      target_weight,
      surprise,
      value,
      search.raw_winloss().to_f64().unwrap(),
      q_values(&search),
    );

    let pos = if let Some(pos) = search.next_root_with_temperature(
      interpolate_early(field, N::from(0.75).unwrap(), N::from(0.15).unwrap()),
      rng,
    ) {
      pos
    } else {
      break;
    };

    // A move other than the one played, drawn from the raw policy of the
    // position, leads to a side position.
    if rng.random::<f64>() < SIDE_POSITION_PROB
      && let Some(side) = side_move(field, &raw_priors, Some(pos.get()), rng)
    {
      let mut side_field = field.clone();
      assert!(side_field.put_point(side, player));
      side_field.update_grounded();
      side_starts.push(SideStart {
        field: side_field,
        player: player.next(),
        komi_x_2: -komi_x_2,
        branch: field.moves_count(),
        first_player: player,
        moves: vec![side],
      });
    }

    visits.push(current_visits);
    search.compact();
    assert!(field.put_point(pos.get(), player));
    field.update_grounded();
    player = player.next();
    komi_x_2 = -komi_x_2;
  }

  let side_positions = search_side_positions(side_starts, model, readouts, rng).await?;

  Ok((visits, side_positions))
}

/// The position a new self-play game starts from when it forks a finished one.
pub struct Fork {
  /// The field after the moves the game is forked from and the fork's move.
  pub field: Field,
  /// The player to move.
  pub player: Player,
  /// Komi of the player to move, multiplied by 2.
  pub komi_x_2: i32,
}

/// Decides whether the finished self-play game `field`, whose first `opening`
/// moves were a set opening and whose komi multiplied by 2 is `komi_x_2` from
/// Red's perspective, forks a new game, and where.
///
/// A fork goes back to a turn of the game - one near its start mostly, see
/// [`EARLY_FORK_PROB`], or any - and plays a move the game did not: the best,
/// by the net's value of the position after it, of a few random legal moves.
/// The fork's game is played out from there with searches, see [`play_out`];
/// the moves before the fork are the original game's, which its own positions
/// already train on.
pub async fn fork<N, M, R>(
  field: &Field,
  opening: usize,
  komi_x_2: i32,
  model: &M,
  rng: &mut R,
) -> Result<Option<Fork>, M::E>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
  R: Rng,
{
  let moves = field.colored_moves().collect::<Vec<_>>();
  if moves.len() <= opening {
    return Ok(None);
  }
  let r = rng.random::<f64>();
  let (turn, max_choices) = if r < EARLY_FORK_PROB {
    // An exponential with the mean of `EARLY_FORK_AREA_PROP` of the area: the
    // argument is the rate, the inverse of the mean.
    let area = f64::from(field.width() * field.height());
    let exp = Exp::<f64>::new(1.0 / (EARLY_FORK_AREA_PROP * area)).unwrap();
    (opening + exp.sample(rng).floor() as usize, EARLY_FORK_MAX_CHOICES)
  } else if r < EARLY_FORK_PROB + LATE_FORK_PROB {
    (rng.random_range(opening..moves.len()), LATE_FORK_MAX_CHOICES)
  } else {
    return Ok(None);
  };
  // The game ended before the turn: its final position has no turn to fork.
  if turn >= moves.len() {
    return Ok(None);
  }

  let mut start = Field::new(field.width(), field.height(), field.zobrist_arc());
  for &(pos, player) in &moves[..turn] {
    assert!(start.put_point(pos, player));
    start.update_grounded();
  }
  let player = moves[turn].1;
  let red_komi = komi_x_2;
  let komi_x_2 = red_komi_x_2(player, red_komi);

  // The candidates, as the fields after them: a move that ends the game would
  // leave nothing to play out.
  let mut legal = (start.min_pos()..=start.max_pos())
    .filter(|&pos| start.is_putting_allowed(pos))
    .collect::<Vec<_>>();
  let choices = rng.random_range(FORK_MIN_CHOICES..=max_choices).min(legal.len());
  let (chosen, _) = legal.partial_shuffle(rng, choices);
  let candidates = chosen
    .iter()
    .filter_map(|&pos| {
      let mut candidate = start.clone();
      assert!(candidate.put_point(pos, player));
      candidate.update_grounded();
      (!candidate.is_game_over(red_komi)).then_some(candidate)
    })
    .collect::<Vec<_>>();
  if candidates.is_empty() {
    return Ok(None);
  }

  // All of them in one forward pass, each under a random symmetry, valued for
  // the opponent, who moves next.
  let (width, height) = rotation_sizes(field.width(), field.height());
  let features = candidates
    .iter()
    .map(|candidate| field_features(candidate, player.next(), width, height, rng.random_range(0..ROTATIONS)))
    .collect::<Vec<Array3<N>>>();
  let globals = candidates
    .iter()
    .map(|candidate| global(candidate, player.next(), -komi_x_2))
    .collect::<Vec<Array1<N>>>();
  let features = stack(Axis(0), &features.iter().map(|f| f.view()).collect::<Vec<_>>()).unwrap();
  let globals = stack(Axis(0), &globals.iter().map(|g| g.view()).collect::<Vec<_>>()).unwrap();
  let (_, values) = model
    .predict(features, globals, Array1::zeros(candidates.len()))
    .await?;
  let best = (0..candidates.len())
    .min_by(|&a, &b| {
      let value = |i: usize| values[(i, 0)] - values[(i, 1)];
      value(a).partial_cmp(&value(b)).unwrap_or(std::cmp::Ordering::Equal)
    })
    .unwrap();

  info!("Forking the game at move {} of {}", turn, moves.len());

  Ok(Some(Fork {
    field: candidates.into_iter().nth(best).unwrap(),
    player: player.next(),
    komi_x_2: -komi_x_2,
  }))
}
