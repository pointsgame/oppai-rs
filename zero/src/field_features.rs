use std::iter;

use ndarray::{Array, Array1, Array3};
use num_traits::{Float, One, Zero};
use oppai_field::cell::Cell;
use oppai_field::field::{Field, NonZeroPos};
use oppai_field::player::Player;
use oppai_rotate::rotate::{rotate, rotate_back};

pub const CHANNELS: usize = 13;

#[inline]
pub fn field_features_len(width: u32, height: u32) -> usize {
  (width * height) as usize * CHANNELS
}

// TODO: KataGo has 2% chance of zeroing history during training
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

pub fn field_features_to_vec<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
  features: &mut Vec<N>,
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
}

pub fn field_features<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
) -> Array3<N> {
  let mut features = Vec::with_capacity(field_features_len(width, height));
  field_features_to_vec::<N>(field, player, width, height, rotation, &mut features);
  Array::from(features)
    .into_shape_with_order((CHANNELS, height as usize, width as usize))
    .unwrap()
}

pub const SCORE_ONE_HOT_SIZE: usize = 1001;

pub fn score_one_hot<N: Float + Zero + One + Copy>(field: &Field, player: Player, komi_x_2: i32) -> Array1<N> {
  let mut score_one_hot = vec![N::zero(); SCORE_ONE_HOT_SIZE];
  const CENTER: i32 = (SCORE_ONE_HOT_SIZE / 2) as i32;
  let score = field.score(player) + komi_x_2.div_euclid(2);
  let lower_idx = CENTER + score;
  let upper_idx = CENTER + score + 1;
  if upper_idx <= 0 {
    score_one_hot[0] = N::one();
  } else if lower_idx >= SCORE_ONE_HOT_SIZE as i32 - 1 {
    score_one_hot[SCORE_ONE_HOT_SIZE - 1] = N::one();
  } else {
    let lambda = N::from(komi_x_2.rem_euclid(2)).unwrap() / (N::one() + N::one());
    score_one_hot[lower_idx as usize] = N::one() - lambda;
    score_one_hot[upper_idx as usize] = lambda;
  }
  Array::from(score_one_hot)
}
