use ndarray::{Array, Array3};
use oppai_field::field::Field;
use oppai_field::player::Player;

fn push_features(field: &Field, player: Player, features: &mut Vec<f64>) {
  features.extend(
    (field.min_pos()..=field.max_pos())
      .filter(|&pos| !field.cell(pos).is_bad())
      .map(|pos| if field.cell(pos).is_owner(player) { 1f64 } else { 0f64 }),
  );
}

pub fn field_features(field: &Field, player: Player) -> Array3<f64> {
  // TODO: rotations, shifts
  let mut features = Vec::with_capacity((field.width() * field.height() * 2) as usize);
  push_features(field, player, &mut features);
  push_features(field, player.next(), &mut features);
  Array::from(features).into_shape((2, field.height() as usize, field.width() as usize)).unwrap().permuted_axes((1, 2, 0))
}
