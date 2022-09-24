use crate::field_features::field_features;
use crate::mcts::MctsNode;
use crate::model::TrainableModel;
use ndarray::{s, ArrayView2, Axis};
use oppai_field::field::{manhattan, to_x, to_y, wave, Field, Pos};
use oppai_field::player::Player;
use rand::Rng;
use std::iter;

fn is_game_ended(field: &Field) -> bool {
  field.points_seq().len() > 50
}

fn winner(field: &Field, player: Player) -> i64 {
  use std::cmp::Ordering;
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

const MCTS_SIMS: u32 = 100;

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

fn find_children(field: &Field, max_distance: u32) -> Vec<Pos> {
  let mut result = Vec::new();
  let mut values = vec![u32::max_value(); field.length()];
  for &pos in field.points_seq() {
    values[pos] = 0;
    wave(field.width(), pos, |next_pos| {
      let distance = manhattan(field.width(), next_pos, pos);
      if field.cell(next_pos).is_putting_allowed() && values[next_pos] > distance {
        if values[next_pos] == u32::max_value() {
          result.push(next_pos);
        }
        values[next_pos] = distance;
        distance < max_distance
      } else {
        false
      }
    });
  }
  result
}

fn create_children(field: &Field, policy: &ArrayView2<f64>, value: f64) -> Vec<MctsNode> {
  let mut children = find_children(field, 3)
    .into_iter()
    .map(|pos| {
      let x = to_x(field.width(), pos);
      let y = to_y(field.width(), pos);
      let p = policy[(x as usize, y as usize)];
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

pub fn episode<E, M, R>(field: &mut Field, mut player: Player, model: &M, rng: &mut R) -> Result<(), E>
where
  M: TrainableModel<E = E>,
  R: Rng,
{
  let mut node = MctsNode::new(0, 0f64, 0f64);

  while !is_game_ended(field) {
    for _ in 0..MCTS_SIMS {
      let leafs = iter::repeat_with(|| node.select())
        .take(PARALLEL_READOUTS)
        .collect::<Vec<_>>();
      for moves in &leafs {
        node.revert_virtual_loss(moves);
      }

      let mut fields = leafs
        .iter()
        .map(|leaf| make_moves(field, leaf, player))
        .collect::<Vec<_>>();

      for cur_field in fields.iter().filter(|field| is_game_ended(field)) {
        node.add_result(
          &field.points_seq()[field.moves_count()..],
          winner(cur_field, player) as f64,
          Vec::new(),
        );
      }

      fields.retain(|field| !is_game_ended(field));
      fields.sort_by_key(|field| field.hash());
      fields.dedup_by_key(|field| field.hash());

      // TODO: rotations
      let feautures = fields
        .iter()
        .map(|field| field_features(field, player, 0))
        .collect::<Vec<_>>();
      let features = ndarray::stack(
        Axis(0),
        feautures.iter().map(|f| f.view()).collect::<Vec<_>>().as_slice(),
      )
      .unwrap();

      let (policies, values) = model.predict(features)?;

      for (i, cur_field) in fields.into_iter().enumerate() {
        let policy = policies.slice(s![i, .., ..]);
        let value = values[i];
        let children = create_children(&cur_field, &policy, value);
        node.add_result(&cur_field.points_seq()[field.moves_count()..], value, children);
      }
    }

    node = select(node.children, rng);
    field.put_point(node.pos, player);
    player = player.next();
  }

  Ok(())
}
