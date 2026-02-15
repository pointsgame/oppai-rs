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
  let mask = array![
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.],
  ];
  #[rustfmt::skip]
  let red = array![
    [1., 0., 1.],
    [0., 1., 0.],
    [0., 1., 0.],
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
  ];
  #[rustfmt::skip]
  let grounded = array![
    [1., 0., 1.],
    [1., 1., 0.],
    [0., 1., 1.],
  ];
  #[rustfmt::skip]
  let empty = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
  assert_eq!(features.slice(s![4, .., ..]), black);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), grounded);

  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);
  assert_eq!(features.slice(s![4, .., ..]), red);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), grounded);
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
  let mask = array![
    [1., 1.],
    [1., 1.],
    [1., 1.],
  ];
  #[rustfmt::skip]
  let red = array![
    [1., 0.],
    [0., 1.],
    [1., 0.],
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 1.],
    [1., 0.],
    [0., 1.],
  ];
  #[rustfmt::skip]
  let empty = array![
    [0., 0.],
    [0., 0.],
    [0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
  assert_eq!(features.slice(s![4, .., ..]), black);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), mask);

  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);
  assert_eq!(features.slice(s![4, .., ..]), red);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), mask);
}

#[test]
fn field_features_wide_rectangle() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    aA
    Aa
    aA
    ",
  );

  #[rustfmt::skip]
  let mask = array![
    [1., 1., 0.],
    [1., 1., 0.],
    [1., 1., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let red = array![
    [1., 0., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let empty = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width() + 1, field.height() + 1, 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red);
  assert_eq!(features.slice(s![4, .., ..]), black);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), mask);

  let features = field_features::<f64>(&field, Player::Black, field.width() + 1, field.height() + 1, 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);
  assert_eq!(features.slice(s![4, .., ..]), red);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), mask);
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
  let mask = array![
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.],
  ];
  #[rustfmt::skip]
  let red = array![
    [0., 1., 0.],
    [1., 0., 1.],
    [0., 1., 0.],
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let red_owner = array![
    [0., 1., 0.],
    [1., 1., 1.],
    [0., 1., 0.],
  ];
  #[rustfmt::skip]
  let black_owner = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let empty = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), black);
  assert_eq!(features.slice(s![3, .., ..]), red_owner);
  assert_eq!(features.slice(s![4, .., ..]), black_owner);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), red_owner);

  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black_owner);
  assert_eq!(features.slice(s![4, .., ..]), red_owner);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), red_owner);
}

#[test]
fn field_features_empty_base() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    a.a
    .a.
    ",
  );

  #[rustfmt::skip]
  let mask = array![
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.],
  ];
  #[rustfmt::skip]
  let red = array![
    [0., 1., 0.],
    [1., 0., 1.],
    [0., 1., 0.],
  ];
  #[rustfmt::skip]
  let empty_base = array![
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let empty = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), red);
  assert_eq!(features.slice(s![2, .., ..]), empty);
  assert_eq!(features.slice(s![3, .., ..]), red);
  assert_eq!(features.slice(s![4, .., ..]), empty);
  assert_eq!(features.slice(s![5, .., ..]), empty_base);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), red);

  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), empty);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), empty);
  assert_eq!(features.slice(s![4, .., ..]), red);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty_base);
  assert_eq!(features.slice(s![7, .., ..]), red);
}
