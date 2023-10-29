use crate::field_features::field_features;
use crate::mcts::{game_result, mcts};
use crate::model::Model;
use crate::{examples::Examples, mcts_node::MctsNode};
use num_traits::Float;
use oppai_field::{
  field::{Field, Pos},
  player::Player,
};
use oppai_rotate::rotate::{MIRRORS, ROTATIONS};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::iter::Sum;

const MCTS_SIMS: u32 = 256;

const EXPLORATION_THRESHOLD: u32 = 30;

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

pub fn episode<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  rng: &mut R,
  examples: &mut Examples<N>,
) -> Result<(), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
{
  let mut node = MctsNode::default();
  let mut moves_count = 0;
  let rotations = if field.width() == field.height() {
    ROTATIONS
  } else {
    MIRRORS
  };

  while !field.is_game_over() {
    for _ in 0..MCTS_SIMS {
      mcts(field, player, &mut node, model, rng)?;
    }

    for rotation in 0..rotations {
      examples.inputs.push(field_features(field, player, rotation));
      examples
        .policies
        .push(node.policies(field.width(), field.height(), rotation));
    }

    node = if moves_count < EXPLORATION_THRESHOLD {
      select(node.children, rng)
    } else {
      node.best_child().unwrap()
    };
    field.put_point(node.pos, player);
    player = player.next();
    moves_count += 1;

    log::debug!(
      "Score: {}, visits: {}, prior probability: {}, wins: {}\n{:?}",
      field.score(Player::Red),
      node.visits,
      node.prior_probability,
      node.wins,
      field
    );
  }

  let mut value = game_result(field, if moves_count % 2 == 0 { player } else { player.next() });
  for _ in 0..moves_count {
    for _ in 0..rotations {
      examples.values.push(value);
    }
    value = -value;
  }

  Ok(())
}
