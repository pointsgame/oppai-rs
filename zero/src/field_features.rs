use ndarray::{Array, Array3};
use num_traits::{One, Zero};
use oppai_field::cell::Cell;
use oppai_field::field::Field;
use oppai_field::player::Player;
use oppai_rotate::rotate::rotate;

fn push_features<N, F: Fn(Cell) -> N + Copy>(field: &Field, f: F, features: &mut Vec<N>, rotation: u8) {
  features.extend((0..field.height()).flat_map(|y| {
    (0..field.width()).map(move |x| {
      let (x, y) = rotate(field.width(), field.height(), x, y, rotation);
      let pos = field.to_pos(x, y);
      f(field.cell(pos))
    })
  }));
}

pub fn field_features<N: Zero + One>(field: &Field, player: Player, rotation: u8) -> Array3<N> {
  let mut features = Vec::with_capacity((field.width() * field.height() * 2) as usize);
  let enemy = player.next();
  push_features(
    field,
    |cell| if cell.is_owner(player) { N::one() } else { N::zero() },
    &mut features,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_owner(enemy) { N::one() } else { N::zero() },
    &mut features,
    rotation,
  );
  push_features(
    field,
    |cell| {
      if cell.is_players_point(player) {
        N::one()
      } else {
        N::zero()
      }
    },
    &mut features,
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
    &mut features,
    rotation,
  );
  Array::from(features)
    .into_shape((4, field.height() as usize, field.width() as usize))
    .unwrap()
}
