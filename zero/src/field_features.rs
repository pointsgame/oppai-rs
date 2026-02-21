use std::iter;

use ndarray::{Array, Array2, Array3};
use num_traits::{Float, One, Zero};
use oppai_field::cell::Cell;
use oppai_field::field::{Field, NonZeroPos};
use oppai_field::player::Player;
use oppai_rotate::rotate::{rotate, rotate_back};

pub const CHANNELS: usize = 14;

#[inline]
pub fn field_features_len(width: u32, height: u32) -> usize {
  (width * height) as usize * CHANNELS
}

fn push_history<N: Zero + One + Copy>(
  field: &Field,
  pos: Option<NonZeroPos>,
  features: &mut Vec<N>,
  width: u32,
  height: u32,
  rotation: u8,
) {
  let field_width = field.width();
  let field_height = field.height();
  let len = features.len();
  features.extend(iter::repeat_n(N::zero(), (width * height) as usize));
  if let Some(pos) = pos {
    let (x, y) = field.to_xy(pos.get());
    let (x, y) = rotate(field_width, field_height, x, y, rotation);
    features[len + (y * width + x) as usize] = N::one();
  }
}

fn push_features<N: Zero, F: Fn(Cell) -> N + Copy>(
  field: &Field,
  f: F,
  features: &mut Vec<N>,
  width: u32,
  height: u32,
  rotation: u8,
) {
  let field_width = field.width();
  let field_height = field.height();
  features.extend((0..height).flat_map(|y| {
    (0..width).map(move |x| {
      if x >= field_width || y >= field_height {
        return N::zero();
      }
      let (x, y) = rotate_back(field_width, field_height, x, y, rotation);
      let pos = field.to_pos(x, y);
      f(field.cell(pos))
    })
  }));
}

fn push_score<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  features: &mut Vec<N>,
  width: u32,
  height: u32,
  komi_x_2: i32,
) {
  let len = features.len();
  features.extend(iter::repeat_n(N::zero(), (width * height) as usize));

  let field_width = field.width();
  let score_len = (field_width * field.height()) as usize;
  let center = (score_len / 2) as i32;

  let score = field.score(player) + komi_x_2.div_euclid(2);
  let lower_idx = center + score;
  let upper_idx = center + score + 1;

  let to_index = |idx: i32| -> usize {
    let idx = idx as u32;
    let x = idx % field_width;
    let y = idx / field_width;
    (y * width + x) as usize
  };

  if upper_idx <= 0 {
    features[len + to_index(0)] = N::one();
  } else if lower_idx >= score_len as i32 - 1 {
    features[len + to_index(score_len as i32 - 1)] = N::one();
  } else {
    let lambda = N::from(komi_x_2.rem_euclid(2)).unwrap() / (N::one() + N::one());
    features[len + to_index(lower_idx)] = N::one() - lambda;
    features[len + to_index(upper_idx)] = lambda;
  }
}

// fn push_score<N: Float + Zero + One + Copy>(
//   field: &Field,
//   player: Player,
//   features: &mut Vec<N>,
//   width: u32,
//   height: u32,
//   komi_x_2: i32,
// ) {
//   let len = features.len();
//   let score_len = (width * height) as usize;
//   features.extend(iter::repeat_n(N::zero(), score_len));
//   let center = (score_len / 2) as i32;
//   let score = field.score(player) + komi_x_2.div_euclid(2);
//   let lower_idx = center + score;
//   let upper_idx = center + score + 1;
//   if upper_idx <= 0 {
//     features[len] = N::one();
//   } else if lower_idx >= score_len as i32 - 1 {
//     features[len + score_len - 1] = N::one();
//   } else {
//     let lambda = N::from(komi_x_2.rem_euclid(2)).unwrap() / (N::one() + N::one());
//     features[len + lower_idx as usize] = N::one() - lambda;
//     features[len + upper_idx as usize] = lambda;
//   }
// }

pub fn field_features_to_vec<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
  features: &mut Vec<N>,
  komi_x_2: i32,
) {
  let enemy = player.next();
  push_features(field, |_| N::one(), features, width, height, rotation);
  push_features(
    field,
    |cell| {
      if cell.is_players_point(player) {
        N::one()
      } else {
        N::zero()
      }
    },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| {
      if cell.is_players_point(enemy) {
        N::one()
      } else {
        N::zero()
      }
    },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_owner(player) { N::one() } else { N::zero() },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_owner(enemy) { N::one() } else { N::zero() },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| {
      if cell.is_players_empty_base(player) {
        N::one()
      } else {
        N::zero()
      }
    },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| {
      if cell.is_players_empty_base(enemy) {
        N::one()
      } else {
        N::zero()
      }
    },
    features,
    width,
    height,
    rotation,
  );
  push_features(
    field,
    |cell| {
      if cell.is_grounded() { N::one() } else { N::zero() }
    },
    features,
    width,
    height,
    rotation,
  );
  push_history(
    field,
    field.moves.last().copied().and_then(NonZeroPos::new),
    features,
    width,
    height,
    rotation,
  );
  push_history(
    field,
    field
      .moves
      .get(field.moves.len().overflowing_sub(2).0)
      .copied()
      .and_then(NonZeroPos::new),
    features,
    width,
    height,
    rotation,
  );
  push_history(
    field,
    field
      .moves
      .get(field.moves.len().overflowing_sub(3).0)
      .copied()
      .and_then(NonZeroPos::new),
    features,
    width,
    height,
    rotation,
  );
  push_history(
    field,
    field
      .moves
      .get(field.moves.len().overflowing_sub(4).0)
      .copied()
      .and_then(NonZeroPos::new),
    features,
    width,
    height,
    rotation,
  );
  push_history(
    field,
    field
      .moves
      .get(field.moves.len().overflowing_sub(5).0)
      .copied()
      .and_then(NonZeroPos::new),
    features,
    width,
    height,
    rotation,
  );
  push_score(field, player, features, width, height, komi_x_2);
}

pub fn field_features<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
  komi_x_2: i32,
) -> Array3<N> {
  let mut features = Vec::with_capacity(field_features_len(width, height));
  field_features_to_vec::<N>(field, player, width, height, rotation, &mut features, komi_x_2);
  Array::from(features)
    .into_shape_with_order((CHANNELS, height as usize, width as usize))
    .unwrap()
}

pub fn score_features<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  komi_x_2: i32,
) -> Array2<N> {
  let mut features = Vec::with_capacity((width * height) as usize);
  push_score::<N>(field, player, &mut features, width, height, komi_x_2);
  Array::from(features)
    .into_shape_with_order((height as usize, width as usize))
    .unwrap()
}
