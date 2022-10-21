use crate::field_features::field_features;
use crate::mcts::MctsNode;
use crate::model::Model;
use ndarray::{s, Array2, Array3, ArrayView2, Axis};
use oppai_common::common::is_last_move_stupid;
use oppai_field::field::{to_x, to_y, Field, Pos};
use oppai_field::player::Player;
use oppai_rotate::rotate::ROTATIONS;
use rand::Rng;
use std::cmp::Ordering;
use std::iter;

fn winner(field: &Field, player: Player) -> i64 {
  match field.score(player).cmp(&0) {
    Ordering::Less => -1,
    Ordering::Equal => 0,
    Ordering::Greater => 1,
  }
}

fn make_moves(initial: &Field, moves: &[Pos], mut player: Player) -> Field {
  let mut field = initial.clone();
  for &pos in moves {
    field.put_point(pos, player);
    player = player.next();
  }
  field
}

const MCTS_SIMS: u32 = 128;

const PARALLEL_READOUTS: usize = 8;

fn select<R: Rng>(mut nodes: Vec<MctsNode>, rng: &mut R) -> MctsNode {
  let r = rng.gen_range(0f64..nodes.iter().map(|child| child.probability()).sum::<f64>());
  let mut node = nodes.pop().unwrap();
  let mut sum = node.probability();
  while sum < r {
    node = nodes.pop().unwrap();
    sum += node.probability();
  }
  node
}

fn create_children(field: &mut Field, player: Player, policy: &ArrayView2<f64>, value: f64) -> Vec<MctsNode> {
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
  // renormalize
  let sum: f64 = children.iter().map(|child| child.p).sum();
  for child in children.iter_mut() {
    child.p /= sum;
  }
  children
}

pub fn mcts<E, M>(field: &mut Field, player: Player, node: &mut MctsNode, model: &M) -> Result<(), E>
where
  M: Model<E = E>,
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
        winner(cur_field, player) as f64,
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
    let children = create_children(&mut cur_field, player, &policy, value);
    let value = if even { value } else { -value };
    node.add_result(&cur_field.points_seq()[field.moves_count()..], value, children);
  }

  Ok(())
}

pub fn episode<E, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &M,
  rng: &mut R,
  inputs: &mut Vec<Array3<f64>>,
  policies: &mut Vec<Array2<f64>>,
  values: &mut Vec<f64>,
) -> Result<(), E>
where
  M: Model<E = E>,
  R: Rng,
{
  let mut node = MctsNode::new(0, 0f64, 0f64);
  let mut moves_count = 0;

  while !field.is_game_over() {
    for _ in 0..MCTS_SIMS {
      mcts(field, player, &mut node, model)?;
    }

    if node.children.is_empty() {
      // no good moves left, but game is not over yet
      break;
    }

    // TODO: check dimensions
    for rotation in 0..ROTATIONS {
      inputs.push(field_features(field, player, rotation));
      policies.push(node.policies(field.width(), field.height(), rotation));
    }

    node = select(node.children, rng);
    field.put_point(node.pos, player);
    player = player.next();
    moves_count += 1;

    log::debug!("Score: {}\n{:?}", field.score(Player::Red), field);
  }

  let mut value = winner(field, if moves_count % 2 == 0 { player } else { player.next() });
  for _ in 0..moves_count {
    for _ in 0..ROTATIONS {
      values.push(value as f64);
    }
    value = -value;
  }

  Ok(())
}
