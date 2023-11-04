use crate::field_features::{field_features_len, field_features_to_vec, CHANNELS};
use crate::mcts_node::MctsNode;
use crate::model::Model;
use ndarray::{s, Array, ArrayView2};
use num_traits::Float;
use oppai_field::field::{to_x, to_y, Field, Pos};
use oppai_field::player::Player;
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt::{Debug, Display};
use std::iter::{self, Sum};

#[inline]
pub fn game_result<N: Float>(field: &Field, player: Player) -> N {
  N::from(field.score(player).signum()).unwrap()
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

const PARALLEL_READOUTS: usize = 8;

fn create_children<N: Float + Sum, R: Rng>(
  field: &mut Field,
  policy: &ArrayView2<N>,
  value: N,
  rng: &mut R,
) -> Vec<MctsNode<N>> {
  let width = field.width();
  let mut children = (field.min_pos()..=field.max_pos())
    .filter(|&pos| field.is_putting_allowed(pos) && !field.is_corner(pos))
    .map(|pos| {
      let x = to_x(width, pos);
      let y = to_y(width, pos);
      let p = policy[(y as usize, x as usize)];
      MctsNode::new(pos, p, value)
    })
    .collect::<Vec<_>>();
  children.shuffle(rng);
  // renormalize
  let sum: N = children.iter().map(|child| child.prior_probability).sum();
  for child in children.iter_mut() {
    child.prior_probability = child.prior_probability / sum;
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
  N: Float + Sum + Display + Debug,
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
        &cur_field.moves()[field.moves_count()..],
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

  let mut features = Vec::with_capacity(field_features_len(field.width(), field.height()) * fields.len());
  for cur_field in &fields {
    field_features_to_vec::<N>(
      cur_field,
      if (cur_field.moves_count() - field.moves_count()) % 2 == 0 {
        player
      } else {
        player.next()
      },
      0,
      &mut features,
    )
  }
  let features = Array::from_shape_vec(
    (fields.len(), CHANNELS, field.height() as usize, field.width() as usize),
    features,
  )
  .unwrap();

  let (policies, values) = model.predict(features)?;

  for (i, mut cur_field) in fields.into_iter().enumerate() {
    let policy = policies.slice(s![i, .., ..]);
    let value = values[i];
    let children = create_children(&mut cur_field, &policy, value, rng);
    let value = if (cur_field.moves_count() - field.moves_count()) % 2 == 0 {
      value
    } else {
      -value
    };
    node.add_result(&cur_field.moves()[field.moves_count()..], value, children);
  }

  Ok(())
}
