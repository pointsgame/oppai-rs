use ndarray::{Array, Array3};
use oppai_field::cell::Cell;
use oppai_field::field::Field;
use oppai_field::player::Player;
use oppai_rotate::rotate::rotate;

fn push_features<F: Fn(Cell) -> f64 + Copy>(field: &Field, f: F, features: &mut Vec<f64>, rotation: u8) {
  features.extend((0..field.height()).flat_map(|y| {
    (0..field.width()).map(move |x| {
      let (x, y) = rotate(field.width(), field.height(), x, y, rotation);
      let pos = field.to_pos(x, y);
      f(field.cell(pos))
    })
  }));
}

pub fn field_features(field: &Field, player: Player, rotation: u8) -> Array3<f64> {
  let mut features = Vec::with_capacity((field.width() * field.height() * 2) as usize);
  let enemy = player.next();
  push_features(
    field,
    |cell| if cell.is_owner(player) { 1f64 } else { 0f64 },
    &mut features,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_owner(enemy) { 1f64 } else { 0f64 },
    &mut features,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_players_point(player) { 1f64 } else { 0f64 },
    &mut features,
    rotation,
  );
  push_features(
    field,
    |cell| if cell.is_players_point(enemy) { 1f64 } else { 0f64 },
    &mut features,
    rotation,
  );
  Array::from(features)
    .into_shape((4, field.height() as usize, field.width() as usize))
    .unwrap()
    .permuted_axes((1, 2, 0))
}
