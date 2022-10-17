use crate::field_features::field_features;
use ndarray::prelude::{array, s};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

#[test]
fn field_features_square() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    a.a
    Aa.
    .aA
    ",
  );

  #[rustfmt::skip]
  let red = array![
    [1., 0., 1.],
    [0., 1., 0.],
    [0., 1., 0.]
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.]
  ];

  let features = field_features(&field, Player::Red, 0);
  assert_eq!(features.slice(s![0, .., ..]), red);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);

  let features = field_features(&field, Player::Black, 0);
  assert_eq!(features.slice(s![0, .., ..]), black);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
}

#[test]
fn field_features_rectangle() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    aA
    Aa
    aA
    ",
  );

  #[rustfmt::skip]
  let red = array![
    [1., 0.],
    [0., 1.],
    [1., 0.]
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 1.],
    [1., 0.],
    [0., 1.]
  ];

  let features = field_features(&field, Player::Red, 0);
  assert_eq!(features.slice(s![0, .., ..]), red);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);

  let features = field_features(&field, Player::Black, 0);
  assert_eq!(features.slice(s![0, .., ..]), black);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
}

#[test]
fn field_features_capture() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aAa
    .a.
    ",
  );

  #[rustfmt::skip]
  let red_owner = array![
    [0., 1., 0.],
    [1., 1., 1.],
    [0., 1., 0.]
  ];
  #[rustfmt::skip]
  let black_owner = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.]
  ];
  #[rustfmt::skip]
  let red = array![
    [0., 1., 0.],
    [1., 0., 1.],
    [0., 1., 0.]
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]
  ];

  let features = field_features(&field, Player::Red, 0);
  assert_eq!(features.slice(s![0, .., ..]), red_owner);
  assert_eq!(features.slice(s![1, .., ..]), black_owner);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);

  let features = field_features(&field, Player::Black, 0);
  assert_eq!(features.slice(s![0, .., ..]), black_owner);
  assert_eq!(features.slice(s![1, .., ..]), red_owner);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
}
