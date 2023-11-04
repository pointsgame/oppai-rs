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
use oppai_rotate::rotate::{rotate, rotate_sizes, MIRRORS, ROTATIONS};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use std::fmt::{Debug, Display};
use std::iter::{self, Sum};
use std::sync::Arc;

const MCTS_SIMS: u32 = 256;

const EXPLORATION_THRESHOLD: u32 = 30;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Visits(pub Vec<(Pos, u64)>);

impl Visits {
  pub fn total(&self) -> u64 {
    self.0.iter().map(|&(_, v)| v).sum()
  }

  /// Improved stochastic policy values.
  pub fn policies<N: Float>(&self, width: u32, height: u32, rotation: u8) -> Array2<N> {
    let total = self.total();
    let (width, height) = rotate_sizes(width, height, rotation);
    let mut policies = Array2::zeros((height as usize, width as usize));

    for &(pos, visits) in &self.0 {
      let x = to_x(width, pos);
      let y = to_y(width, pos);
      let (x, y) = rotate(width, height, x, y, rotation);
      policies[(y as usize, x as usize)] = N::from(visits).unwrap() / N::from(total).unwrap();
    }

    policies
  }
}

fn select<N: Float + Sum + SampleUniform, R: Rng>(mut nodes: Vec<MctsNode<N>>, rng: &mut R) -> MctsNode<N> {
  let r = rng.gen_range(N::zero()..nodes.iter().map(|child| child.probability()).sum::<N>());
  let mut sum = N::zero();
  while let Some(node) = nodes.pop() {
    sum = sum + node.probability();
    if sum > r {
      return node;
    }
  }
  unreachable!()
}

pub fn episode<N, M, R>(field: &mut Field, mut player: Player, model: &M, rng: &mut R) -> Result<Vec<Visits>, M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  let mut node = MctsNode::default();
  let mut moves_count = 0;
  let mut visits = Vec::new();

  while !field.is_game_over() {
    for _ in 0..MCTS_SIMS {
      mcts(field, player, &mut node, model, rng)?;
    }

    visits.push(Visits(
      node
        .children
        .iter()
        .filter(|child| child.visits > 0)
        .map(|child| (child.pos, child.visits))
        .collect(),
    ));

    node = if moves_count < EXPLORATION_THRESHOLD {
      select(node.children, rng)
    } else {
      node.best_child().unwrap()
    };
    node.add_dirichlet_noise(rng, N::from(0.25).unwrap(), N::from(0.03).unwrap());
    field.put_point(node.pos, player);
    player = player.next();
    moves_count += 1;

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
    assert!(field.put_point(pos, player), "invalid moves siequence");
  }

  for (&(pos, player), visits) in moves[initial_moves..].iter().zip(visits.iter()) {
    for rotation in 0..rotations {
      examples.inputs.push(field_features(&field, player, rotation));
      examples.policies.push(visits.policies(width, height, rotation));
    }

    assert!(field.put_point(pos, player), "invalid moves siequence");
  }

  let value = game_result::<N>(&field, Player::Red);
  for &(_, player) in &moves[initial_moves..] {
    let value = match player {
      Player::Red => value,
      Player::Black => -value,
    };
    examples.values.extend(iter::repeat(value).take(rotations as usize));
  }

  examples
}
