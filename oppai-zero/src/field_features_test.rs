use rand_xorshift::XorShiftRng;
use rand::SeedableRng;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use ndarray::prelude::{array, s};
use crate::field_features::field_features;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[test]
fn field_features_square() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    a.a
    Aa.
    .aA
    ",
  );

  let red = array![
    [1., 0., 1.],
    [0., 1., 0.],
    [0., 1., 0.]
  ];
  let black = array![
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.]
  ];

  let features = field_features(&field, Player::Red);
  assert_eq!(features.slice(s![.., .., 0]), red);
  assert_eq!(features.slice(s![.., .., 1]), black);

  let features = field_features(&field, Player::Black);
  assert_eq!(features.slice(s![.., .., 0]), black);
  assert_eq!(features.slice(s![.., .., 1]), red);
}

#[test]
fn field_features_rectangle() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    aA
    Aa
    aA
    ",
  );

  let red = array![
    [1., 0.],
    [0., 1.],
    [1., 0.]
  ];
  let black = array![
    [0., 1.],
    [1., 0.],
    [0., 1.]
  ];

  let features = field_features(&field, Player::Red);
  assert_eq!(features.slice(s![.., .., 0]), red);
  assert_eq!(features.slice(s![.., .., 1]), black);

  let features = field_features(&field, Player::Black);
  assert_eq!(features.slice(s![.., .., 0]), black);
  assert_eq!(features.slice(s![.., .., 1]), red);
}

#[test]
fn field_features_capture() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a.
    aAa
    .a.
    ",
  );

  let red = array![
    [0., 1., 0.],
    [1., 1., 1.],
    [0., 1., 0.]
  ];
  let black = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.]
  ];

  let features = field_features(&field, Player::Red);
  assert_eq!(features.slice(s![.., .., 0]), red);
  assert_eq!(features.slice(s![.., .., 1]), black);

  let features = field_features(&field, Player::Black);
  assert_eq!(features.slice(s![.., .., 0]), black);
  assert_eq!(features.slice(s![.., .., 1]), red);
}
