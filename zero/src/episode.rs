use crate::examples::Examples;
use crate::field_features::{field_features, score_one_hop};
use crate::mcgs::Search;
use crate::model::Model;
use log::info;
use ndarray::{Array1, Array2, Array3, Axis, array};
use num_traits::{Float, One, Zero};
use oppai_field::field::{to_x, to_y};
use oppai_field::zobrist::Zobrist;
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use oppai_rotate::rotate::{MIRRORS, ROTATIONS, rotate, rotate_sizes};
use rand::Rng;
use rand::distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp, Exp1, Open01, StandardNormal};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::{self, Sum};
use std::sync::Arc;

const MCTS_SIMS: u32 = 200;
const MCTS_FULL_SIMS: u32 = 1000;

#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct Visits(pub Vec<(Pos, u64)>, pub bool);

impl Visits {
  pub fn total(&self) -> u64 {
    self.0.iter().map(|&(_, v)| v).sum()
  }

  pub fn max(&self) -> u64 {
    self.0.iter().map(|&(_, v)| v).max().unwrap_or_default()
  }

  /// Improved stochastic policy values.
  pub fn policies<N: Float>(&self, width: u32, height: u32, rotation: u8) -> Array2<N> {
    let total = self.total();
    let (width, height) = rotate_sizes(width, height, rotation);
    let mut policies = Array2::zeros((height as usize, width as usize));

    for &(pos, visits) in &self.0 {
      let x = to_x(width + 1, pos);
      let y = to_y(width + 1, pos);
      let (x, y) = rotate(width, height, x, y, rotation);
      policies[(y as usize, x as usize)] = N::from(visits).unwrap() / N::from(total).unwrap();
    }

    policies
  }
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
  let r = rng.random_range(N::zero()..sum);
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      let policy = policy[(0, y as usize, x as usize)];
      if policy > r {
        return NonZeroPos::new(pos);
      } else {
        sum = sum - policy;
      }
    }
  }
  None
}

pub fn episode<N, M, R>(field: &mut Field, mut player: Player, model: &mut M, rng: &mut R) -> Result<Vec<Visits>, M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  // TODO: does it scale to big sizes?
  let exp = Exp::new(N::from(25).unwrap() / N::from(field.width() * field.height()).unwrap()).unwrap();
  let raw_policy_moves = exp.sample(rng).floor().to_u32().unwrap();

  info!("Playing {} raw policy moves", raw_policy_moves);

  for _ in 0..raw_policy_moves {
    let features = field_features(field, player, field.width(), field.height(), 0);
    let (policy, _) = model.predict(features.insert_axis(Axis(0)))?;
    if let Some(pos) = select_policy_move(field, policy, rng) {
      assert!(field.put_point(pos.get(), player));
      field.update_grounded();
      player = player.next();
    } else {
      break;
    }
  }

  let mut search = Search::new();
  let mut visits = Vec::new();

  while !field.is_game_over() {
    let full_search = rng.random::<f64>() <= 0.25;

    let sims = if full_search {
      // TODO: does it scale to big sizes?
      let shape = N::from(0.03 * 19.0.powi(2)).unwrap() / N::from(field.width() * field.height()).unwrap();
      // TODO: dynamically adjust temperature from 1.25 to 1.1?
      // see Search::maybeAddPolicyNoiseAndTemp and interpolateEarly in KataGo
      let temperature = N::from(1.1).unwrap();
      search.add_dirichlet_noise(rng, N::from(0.25).unwrap(), shape, temperature);
      MCTS_FULL_SIMS
    } else {
      MCTS_SIMS
    };

    for _ in 0..sims {
      search.mcgs(field, player, model, rng)?;
    }

    visits.push(Visits(
      search.visits().filter(|(_, visits)| *visits > 0).collect(),
      full_search,
    ));

    let pos = search.next_best_root().unwrap();
    search.compact();
    assert!(field.put_point(pos.get(), player));
    field.update_grounded();
    player = player.next();
  }

  Ok(visits)
}

fn game_result<N: Float>(field: &Field, player: Player) -> Array1<N> {
  match field.score(player).cmp(&0) {
    Ordering::Less => array![N::zero(), N::one()],
    Ordering::Equal => array![N::one() / (N::one() + N::one()), N::one() / (N::one() + N::one())],
    Ordering::Greater => array![N::one(), N::zero()],
  }
}

pub fn examples<N: Float + Zero + One>(
  width: u32,
  height: u32,
  zobrist: Arc<Zobrist<u64>>,
  visits: &[Visits],
  moves: &[(Pos, Player)],
) -> Examples<N> {
  let mut examples = Examples::<N>::default();
  let mut field = Field::new(width, height, zobrist);

  let initial_moves = moves.len() - visits.len();
  let rotations = if width == height { ROTATIONS } else { MIRRORS };

  for &(pos, player) in &moves[0..initial_moves] {
    assert!(field.put_point(pos, player), "invalid moves sequence");
    field.update_grounded();
  }

  for (&(pos, player), visits) in moves[initial_moves..].iter().zip(visits.iter()) {
    if visits.1 {
      for rotation in 0..rotations {
        examples
          .inputs
          .push(field_features(&field, player, field.width(), field.height(), rotation));
        examples.policies.push(visits.policies(width, height, rotation));
      }
    }

    assert!(field.put_point(pos, player), "invalid moves sequence");
    field.update_grounded();
  }

  for (&(_, player), visits) in moves[initial_moves..].iter().zip(visits.iter()) {
    if visits.1 {
      let value = game_result::<N>(&field, player);
      examples.values.extend(iter::repeat_n(value, rotations as usize));
      let score = score_one_hop(&field, player, 0);
      examples.scores.extend(iter::repeat_n(score, rotations as usize));
    }
  }

  examples
}
