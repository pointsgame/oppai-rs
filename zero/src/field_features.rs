use std::iter;

use ndarray::{Array, Array1, Array2, Array3, ArrayView2};
use num_traits::{Float, One, Zero};
use oppai_field::cell::Cell;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use oppai_rotate::rotate::{rotate, rotate_back, rotate_sizes};

use crate::tactics::{TACTICAL_CHANNELS, Tactics};

/// Number of history planes, each marking the location of one past move.
pub const HISTORY_CHANNELS: usize = 5;

/// The number of input planes, see [`field_features_to_vec`].
pub const CHANNELS: usize = 8 + HISTORY_CHANNELS + TACTICAL_CHANNELS;

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
  push_plane(field, |pos| f(field.cell(pos)), features, width, height, rotation);
}

/// Pushes a plane holding `f` of every position of the field under `rotation`,
/// padded with zeros to `width` and `height`.
fn push_plane<N: Zero, F: Fn(Pos) -> N + Copy>(
  field: &Field,
  f: F,
  features: &mut Vec<N>,
  width: u32,
  height: u32,
  rotation: u8,
) {
  let (field_width, field_height) = rotate_sizes(field.width(), field.height(), rotation);
  features.extend((0..height).flat_map(|y| {
    (0..width).map(move |x| {
      if x >= field_width || y >= field_height {
        return N::zero();
      }
      let (x, y) = rotate_back(field_width, field_height, x, y, rotation);
      f(field.to_pos(x, y))
    })
  }));
}

/// Pushes the input planes of the position `field` holds with `player` to
/// move, under `rotation` and padded to `width` and `height`:
///
/// 0. the board itself;
/// 1. and 2. the dots of the player to move and of the opponent;
/// 3. and 4. the cells each of them owns;
/// 5. and 6. their empty bases;
/// 7. the grounded cells;
/// 8. to 12. the last [`HISTORY_CHANNELS`] moves, the latest first;
/// 13. to 16. the tactical planes of `tactics`, see [`Tactics`], which has to
///     be built for this field and `player`.
///
/// `history` is the number of history planes to fill; the remaining ones are
/// left empty. Inference uses the full `HISTORY_CHANNELS` while training
/// truncates it randomly (history dropout).
#[allow(clippy::too_many_arguments)]
pub fn field_features_to_vec<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  tactics: &Tactics,
  width: u32,
  height: u32,
  rotation: u8,
  history: usize,
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
  for i in 0..HISTORY_CHANNELS {
    push_history(
      field,
      if i < history {
        field
          .moves
          .get(field.moves.len().overflowing_sub(i + 1).0)
          .copied()
          .and_then(NonZeroPos::new)
      } else {
        None
      },
      features,
      width,
      height,
      rotation,
    );
  }
  for plane in 0..TACTICAL_CHANNELS {
    push_plane(
      field,
      |pos| {
        if tactics.plane(plane, pos) { N::one() } else { N::zero() }
      },
      features,
      width,
      height,
      rotation,
    );
  }
}

/// Two planes marking captured cells in the given field state, with or without
/// a dot: first the cells captured by `player`, then the ones captured by the enemy.
pub fn captured_features_to_vec<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
  features: &mut Vec<N>,
) {
  let enemy = player.next();
  push_features(
    field,
    |cell| {
      if cell.is_captured() && cell.is_owner(player) {
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
      if cell.is_captured() && cell.is_owner(enemy) {
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
}

/// The input planes of [`field_features_to_vec`] as an array, with every
/// history plane filled. Builds the [`Tactics`] on a copy of `field`, so it
/// is for the places that need the planes of few positions.
pub fn field_features<N: Float + Zero + One + Copy>(
  field: &Field,
  player: Player,
  width: u32,
  height: u32,
  rotation: u8,
) -> Array3<N> {
  let tactics = Tactics::new(&mut field.clone(), player);
  let mut features = Vec::with_capacity(field_features_len(width, height));
  field_features_to_vec::<N>(
    field,
    player,
    &tactics,
    width,
    height,
    rotation,
    HISTORY_CHANNELS,
    &mut features,
  );
  Array::from(features)
    .into_shape_with_order((CHANNELS, height as usize, width as usize))
    .unwrap()
}

/// Input sizes that fit the field under every rotation: a square of its longer
/// side. The rotations that transpose a rectangular field swap its sides, and
/// padding to a square lets positions evaluated under different rotations
/// share one batch.
#[inline]
pub fn rotation_sizes(width: u32, height: u32) -> (u32, u32) {
  let side = width.max(height);
  (side, side)
}

/// Adds `scale` times a policy predicted for the field evaluated under
/// `rotation` to `policy`, read back into field coordinates: `policy` is
/// indexed by `(y, x)` of the field itself, whatever the rotation was. Called
/// once per rotation with `scale` the reciprocal of their number, it averages
/// the predictions of several rotations.
pub fn add_unrotated_policy<N: Float>(
  predicted: ArrayView2<N>,
  width: u32,
  height: u32,
  rotation: u8,
  scale: N,
  policy: &mut Array2<N>,
) {
  for y in 0..height {
    for x in 0..width {
      let (rx, ry) = rotate(width, height, x, y, rotation);
      let cell = &mut policy[(y as usize, x as usize)];
      *cell = *cell + scale * predicted[(ry as usize, rx as usize)];
    }
  }
}

pub const SCORE_ONE_HOT_SIZE: usize = 1001;

pub fn score_one_hot_to_vec<N: Float + Zero + One + Copy>(score: i32, komi_x_2: i32, scores: &mut Vec<N>) {
  let start_idx = scores.len();
  scores.extend(iter::repeat_n(N::zero(), SCORE_ONE_HOT_SIZE));
  const CENTER: i32 = (SCORE_ONE_HOT_SIZE / 2) as i32;
  let score = score + komi_x_2.div_euclid(2);
  let lower_idx = CENTER + score;
  let upper_idx = CENTER + score + 1;
  if upper_idx <= 0 {
    scores[start_idx] = N::one();
  } else if lower_idx >= SCORE_ONE_HOT_SIZE as i32 - 1 {
    scores[start_idx + SCORE_ONE_HOT_SIZE - 1] = N::one();
  } else {
    let lambda = N::from(komi_x_2.rem_euclid(2)).unwrap() / (N::one() + N::one());
    scores[start_idx + lower_idx as usize] = N::one() - lambda;
    scores[start_idx + upper_idx as usize] = lambda;
  }
}

pub fn score_one_hot<N: Float + Zero + One + Copy>(score: i32, komi_x_2: i32) -> Array1<N> {
  let mut score_one_hot = Vec::with_capacity(SCORE_ONE_HOT_SIZE);
  score_one_hot_to_vec(score, komi_x_2, &mut score_one_hot);
  Array::from(score_one_hot)
}

/// The features of the position as a whole, see [`global_to_vec`].
pub const GLOBAL_FEATURES: usize = 3;

/// The index of the komi among the [`GLOBAL_FEATURES`], in points from the
/// view of the player to move.
pub const GLOBAL_KOMI: usize = 2;

/// The features of the position as a whole, from the view of `player`, the
/// player to move, with `komi_x_2` from that view too:
///
/// 0. the lead with the komi, in units of 20 points;
/// 1. 1 if the komi is whole, so that the game can end in a draw, and 0 if it
///    is a half point;
/// 2. the komi itself, in points.
///
/// The lead alone hides the komi: a lead of one point under a komi of -0.5 is
/// the same number as no lead under a komi of 0.5, and a whole komi differs
/// from a half one by 1/40 of it, too little for the network to read whether
/// a draw is possible. The last two features tell both apart exactly.
pub fn global_to_vec<N: Float + Zero + One + Copy>(field: &Field, player: Player, komi_x_2: i32, global: &mut Vec<N>) {
  let score = N::from(field.score(player) * 2 + komi_x_2).unwrap() / N::from(40).unwrap();
  global.push(score);
  global.push(if komi_x_2 % 2 == 0 { N::one() } else { N::zero() });
  global.push(N::from(komi_x_2).unwrap() / (N::one() + N::one()));
}

pub fn global<N: Float + Zero + One>(field: &Field, player: Player, komi_x_2: i32) -> Array1<N> {
  let mut features = Vec::with_capacity(GLOBAL_FEATURES);
  global_to_vec(field, player, komi_x_2, &mut features);
  Array::from(features)
}
