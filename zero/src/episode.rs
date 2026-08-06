use crate::field_features::{field_features, global};
use crate::mcgs::{Params, Search};
use crate::model::Model;
use log::info;
use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use oppai_rotate::rotate::rotate;
use rand::distr::uniform::SampleUniform;
use rand::{Rng, RngExt};
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

/// Interpolates between an early-game and a rest-of-game value, decaying
/// exponentially with the number of moves played. KataGo's interpolateEarly
/// uses a halflife of `sqrt(area)` moves, spending ~7.6% of a typical go game
/// at the early value; dots games fill only ~0.32 of the field, so half that
/// halflife keeps the same fraction of a typical game.
fn interpolate_early<N: Float>(field: &Field, early_value: N, value: N) -> N {
  let halflives = N::from(2 * field.moves_count()).unwrap() / N::from(field.width() * field.height()).unwrap().sqrt();
  value + (early_value - value) * N::from(0.5).unwrap().powf(halflives)
}

fn select_policy_move<N, R>(field: &Field, policy: Array3<N>, rng: &mut R) -> Option<NonZeroPos>
where
  N: Float + Sum + SampleUniform,
  R: Rng,
{
  // TODO: KataGo also makes random moves with small probability, see PlayUtils::getGameInitializationMove
  let mut sum = N::zero();
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      sum = sum + policy[(0, y as usize, x as usize)];
    }
  }
  let mut sample = rng.random_range(N::zero()..sum);
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      let policy = policy[(0, y as usize, x as usize)];
      if policy >= sample {
        return NonZeroPos::new(pos);
      } else {
        sample = sample - policy;
      }
    }
  }
  None
}

pub async fn episode<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  mut komi_x_2: i32,
  rng: &mut R,
) -> Result<Vec<Visits>, M::E>
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
    let features = field_features(field, player, field.width(), field.height(), 0);
    let global = global(field, player, komi_x_2);
    // The opening is sampled from the trained policy: these moves are meant to
    // spread the training positions over the openings the net actually plays.
    let (policy, _) = model
      .predict(
        features.insert_axis(Axis(0)),
        global.insert_axis(Axis(0)),
        Array1::zeros(1),
      )
      .await?;
    if let Some(pos) = select_policy_move(field, policy, rng) {
      assert!(field.put_point(pos.get(), player));
      field.update_grounded();
      player = player.next();
      komi_x_2 = -komi_x_2;
    } else {
      break;
    }
  }

  let mut search = Search::new(Params::SELF_PLAY);
  let mut visits = Vec::new();

  // Raw network policy priors of the root, captured before temperature and
  // Dirichlet noise overwrite them, so the policy surprise is measured against
  // the network's true prior rather than the noised one.
  let mut raw_priors = vec![N::zero(); field.length()];

  // Every turn's search value from Red's perspective, driving the visit
  // reduction of full searches once the game looks decided.
  let mut search_values: Vec<f64> = Vec::new();

  while !field.is_game_over(if player == Player::Red { komi_x_2 } else { -komi_x_2 }) {
    let full_search = rng.random::<f64>() <= 0.25;

    let (target_visits, target_weight) = if full_search {
      // Recorded searches start from a fresh tree: the Dirichlet noise and
      // forced playouts have to shape the entire visit distribution, and visits
      // inherited from previous searches would leak into the policy target and
      // inflate the policy surprise. Cheap searches keep reusing the tree -
      // they only pick a move.
      search.clear();
      // The root has to be expanded before the noise can be applied to its children priors.
      search.mcgs(field, player, model, komi_x_2, rng).await?;
      search.root_priors(&mut raw_priors);
      // Total Dirichlet alpha, matching AlphaZero's 0.03 per move on an empty 19x19 board
      // (0.03 * 361 = 10.83). Kept constant across board sizes and through the game, with
      // the shaping in `add_dirichlet_noise` deciding how it is spread across the moves.
      let total_concentration = N::from(0.03 * 19.0.powi(2)).unwrap();
      let temperature = interpolate_early(field, N::from(1.25).unwrap(), N::from(1.1).unwrap());
      search.add_dirichlet_noise(rng, N::from(0.25).unwrap(), total_concentration, temperature);
      reduced_search(&search_values)
    } else {
      (MCTS_VISITS, 0.0)
    };

    // At least one batch even when the reused subtree already meets the
    // budget, so every move rests on a search of the actual root.
    loop {
      search.mcgs(field, player, model, komi_x_2, rng).await?;
      if search.root_visits() >= u64::from(target_visits) {
        break;
      }
    }

    // Clamped like the per-move q targets: the winloss averaging can drift a
    // hair past the `[-1, 1]` a value is defined on, and the training loss
    // reads the value target as a probability that has to stay in `[0, 1]`.
    let value = search.winloss().to_f64().unwrap().clamp(-1.0, 1.0);
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
        .collect(),
    );

    let pos = if let Some(pos) = search.next_root_with_temperature(
      interpolate_early(field, N::from(0.75).unwrap(), N::from(0.15).unwrap()),
      rng,
    ) {
      pos
    } else {
      break;
    };

    visits.push(current_visits);
    search.compact();
    assert!(field.put_point(pos.get(), player));
    field.update_grounded();
    player = player.next();
    komi_x_2 = -komi_x_2;
  }

  Ok(visits)
}
