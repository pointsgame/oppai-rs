use crate::field_features::field_features;
use crate::mcts::{game_result, mcts};
use crate::model::Model;
use crate::{examples::Examples, mcts_node::MctsNode};
use ndarray::Array2;
use num_traits::{Float, One, Zero};
use oppai_field::field::{to_x, to_y};
use oppai_field::zobrist::Zobrist;
use oppai_field::{
  field::{Field, Pos},
  player::Player,
};
use oppai_rotate::rotate::{MIRRORS, ROTATIONS, rotate, rotate_sizes};
use rand::Rng;
use rand::distr::uniform::SampleUniform;
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
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

pub fn episode<N, M, R>(field: &mut Field, mut player: Player, model: &mut M, rng: &mut R) -> Result<Vec<Visits>, M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  let mut node = MctsNode::default();
  let mut visits = Vec::new();

  while !field.is_game_over() {
    if node.children.is_empty() {
      mcts(field, player, &mut node, model, rng)?;
    }

    let full_search = field.moves_count() > 40 && rng.random::<f64>() <= 0.25;

    let sims = if full_search {
      node.add_dirichlet_noise(rng, N::from(0.25).unwrap(), N::from(0.03).unwrap());
      MCTS_FULL_SIMS
    } else {
      MCTS_SIMS
    };

    for _ in 0..sims {
      mcts(field, player, &mut node, model, rng)?;
    }

    visits.push(Visits(
      node
        .children
        .iter()
        .filter(|child| child.visits > 0)
        .map(|child| (child.pos, child.visits))
        .collect(),
      full_search,
    ));

    node = node.best_child().unwrap();
    field.put_point(node.pos, player);
    player = player.next();

    log::debug!(
      "Score: {}, visits: {}, policy: {}, wins: {}\n{:?}",
      field.score(Player::Red),
      node.visits,
      node.policy,
      node.wins,
      field
    );
  }

  Ok(visits)
}

pub fn examples<N: Float + Zero + One>(
  width: u32,
  height: u32,
  zobrist: Arc<Zobrist>,
  visits: &[Visits],
  moves: &[(Pos, Player)],
) -> Examples<N> {
  let mut examples = Examples::<N>::default();
  let mut field = Field::new(width, height, zobrist);

  let initial_moves = moves.len() - visits.len();
  let rotations = if width == height { ROTATIONS } else { MIRRORS };

  for &(pos, player) in &moves[0..initial_moves] {
    assert!(field.put_point(pos, player), "invalid moves sequence");
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
  }

  let value = game_result::<N>(&field, Player::Red);
  for (&(_, player), visits) in moves[initial_moves..].iter().zip(visits.iter()) {
    if visits.1 {
      let value = match player {
        Player::Red => value,
        Player::Black => -value,
      };
      examples.values.extend(iter::repeat_n(value, rotations as usize));
    }
  }

  examples
}
