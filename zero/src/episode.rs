use crate::field_features::field_features;
use crate::mcts::MctsNode;
use crate::model::Model;
use ndarray::{s, Array2, Array3, ArrayView2, Axis};
use num_traits::Float;
use oppai_common::common::is_last_move_stupid;
use oppai_field::field::{to_x, to_y, Field, Pos};
use oppai_field::player::Player;
use oppai_rotate::rotate::{MIRRORS, ROTATIONS};
use rand::distributions::uniform::SampleUniform;
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt::Display;
use std::iter::{self, Sum};

#[inline]
pub fn logistic<N: Float>(p: N) -> N {
  let l = N::one() + N::one();
  let k = N::one();
  l / ((-p * k).exp() + N::one()) - N::one()
}

#[inline]
fn game_result<N: Float>(field: &Field, player: Player) -> N {
  logistic(N::from(field.score(player)).unwrap())
}

#[inline]
fn make_moves(initial: &Field, moves: &[Pos], mut player: Player) -> Field {
  let mut field = initial.clone();
  for &pos in moves {
    field.put_point(pos, player);
    player = player.next();
  }
  field
}

const MCTS_SIMS: u32 = 256;

const PARALLEL_READOUTS: usize = 8;

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

fn create_children<N: Float + Sum, R: Rng>(
  field: &mut Field,
  player: Player,
  policy: &ArrayView2<N>,
  value: N,
  rng: &mut R,
) -> Vec<MctsNode<N>> {
  let width = field.width();
  let mut children = (field.min_pos()..=field.max_pos())
    .filter(|&pos| {
      field.is_putting_allowed(pos) && {
        field.put_point(pos, player);
        let is_stupid = is_last_move_stupid(field, pos, player);
        field.undo();
        !is_stupid
      }
    })
    .map(|pos| {
      let x = to_x(width, pos);
      let y = to_y(width, pos);
      let p = policy[(y as usize, x as usize)];
      MctsNode::new(pos, p, value)
    })
    .collect::<Vec<_>>();
  children.shuffle(rng);
  // renormalize
  let sum: N = children.iter().map(|child| child.p).sum();
  for child in children.iter_mut() {
    child.p = child.p / sum;
  }
  children
}

pub fn mcts<N, M, R>(
  field: &mut Field,
  player: Player,
  node: &mut MctsNode<N>,
  model: &M,
  rng: &mut R,
) -> Result<(), M::E>
where
  M: Model<N>,
  N: Float + Sum,
  R: Rng,
{
  let mut leafs = iter::repeat_with(|| node.select())
    .take(PARALLEL_READOUTS)
    .collect::<Vec<_>>();
  for moves in &leafs {
    node.revert_virtual_loss(moves);
  }

  leafs.sort_unstable();
  leafs.dedup();

  let mut fields = leafs
    .iter()
    .map(|leaf| make_moves(field, leaf, player))
    .collect::<Vec<_>>();

  fields.retain_mut(|cur_field| {
    if cur_field.is_game_over() {
      node.add_result(
        &cur_field.points_seq()[field.moves_count()..],
        game_result(cur_field, player),
        Vec::new(),
      );
      false
    } else {
      true
    }
  });

  if fields.is_empty() {
    return Ok(());
  }

  let feautures = fields
    .iter()
    .map(|cur_field| {
      field_features(
        cur_field,
        if (cur_field.moves_count() - field.moves_count()) % 2 == 0 {
          player
        } else {
          player.next()
        },
        0,
      )
    })
    .collect::<Vec<_>>();
  let features = ndarray::stack(
    Axis(0),
    feautures.iter().map(|f| f.view()).collect::<Vec<_>>().as_slice(),
  )
  .unwrap();

  let (policies, values) = model.predict(features)?;

  for (i, mut cur_field) in fields.into_iter().enumerate() {
    let policy = policies.slice(s![i, .., ..]);
    let value = values[i];
    let even = (cur_field.moves_count() - field.moves_count()) % 2 == 0;
    let player = if even { player } else { player.next() };
    let children = create_children(&mut cur_field, player, &policy, value, rng);
    let value = if even { value } else { -value };
    node.add_result(&cur_field.points_seq()[field.moves_count()..], value, children);
  }

  Ok(())
}

pub fn episode<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  rng: &mut R,
  inputs: &mut Vec<Array3<N>>,
  policies: &mut Vec<Array2<N>>,
  values: &mut Vec<N>,
) -> Result<(), M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display,
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

    if node.children.is_empty() {
      // no good moves left, but game is not over yet
      break;
    }

    for rotation in 0..rotations {
      inputs.push(field_features(field, player, rotation));
      policies.push(node.policies(field.width(), field.height(), rotation));
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
      "Score: {}, n: {}, p: {}, w: {}\n{:?}",
      field.score(Player::Red),
      node.n,
      node.p,
      node.w,
      field
    );
  }

  let mut value = game_result(field, if moves_count % 2 == 0 { player } else { player.next() });
  for _ in 0..moves_count {
    for _ in 0..rotations {
      values.push(value);
    }
    value = -value;
  }

  Ok(())
}
