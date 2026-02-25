use crate::field_features::{SCORE_ONE_HOP_SIZE, field_features, score_one_hop};
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
    d.c
    Eb.
    .aF
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
  #[rustfmt::skip]
  let history_1 = array![
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 1.],
  ];
  #[rustfmt::skip]
  let history_2 = array![
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let history_3 = array![
    [1., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let history_4 = array![
    [0., 0., 1.],
    [0., 0., 0.],
    [0., 0., 0.],
  ];
  #[rustfmt::skip]
  let history_5 = array![
    [0., 0., 0.],
    [0., 1., 0.],
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
  assert_eq!(features.slice(s![8, .., ..]), history_1);
  assert_eq!(features.slice(s![9, .., ..]), history_2);
  assert_eq!(features.slice(s![10, .., ..]), history_3);
  assert_eq!(features.slice(s![11, .., ..]), history_4);
  assert_eq!(features.slice(s![12, .., ..]), history_5);

  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(features.slice(s![0, .., ..]), mask);
  assert_eq!(features.slice(s![1, .., ..]), black);
  assert_eq!(features.slice(s![2, .., ..]), red);
  assert_eq!(features.slice(s![3, .., ..]), black);
  assert_eq!(features.slice(s![4, .., ..]), red);
  assert_eq!(features.slice(s![5, .., ..]), empty);
  assert_eq!(features.slice(s![6, .., ..]), empty);
  assert_eq!(features.slice(s![7, .., ..]), grounded);
  assert_eq!(features.slice(s![8, .., ..]), history_1);
  assert_eq!(features.slice(s![9, .., ..]), history_2);
  assert_eq!(features.slice(s![10, .., ..]), history_3);
  assert_eq!(features.slice(s![11, .., ..]), history_4);
  assert_eq!(features.slice(s![12, .., ..]), history_5);

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 4);
  assert_eq!(features.slice(s![0, .., ..]), mask.t());
  assert_eq!(features.slice(s![1, .., ..]), red.t());
  assert_eq!(features.slice(s![2, .., ..]), black.t());
  assert_eq!(features.slice(s![3, .., ..]), red.t());
  assert_eq!(features.slice(s![4, .., ..]), black.t());
  assert_eq!(features.slice(s![5, .., ..]), empty.t());
  assert_eq!(features.slice(s![6, .., ..]), empty.t());
  assert_eq!(features.slice(s![7, .., ..]), grounded.t());
  assert_eq!(features.slice(s![8, .., ..]), history_1.t());
  assert_eq!(features.slice(s![9, .., ..]), history_2.t());
  assert_eq!(features.slice(s![10, .., ..]), history_3.t());
  assert_eq!(features.slice(s![11, .., ..]), history_4.t());
  assert_eq!(features.slice(s![12, .., ..]), history_5.t());

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 5);
  assert_eq!(features.slice(s![0, .., ..]), mask.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![1, .., ..]), red.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![2, .., ..]), black.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![3, .., ..]), red.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![4, .., ..]), black.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![5, .., ..]), empty.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![6, .., ..]), empty.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![7, .., ..]), grounded.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![8, .., ..]), history_1.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![9, .., ..]), history_2.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![10, .., ..]), history_3.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![11, .., ..]), history_4.slice(s![..; -1, ..]).t());
  assert_eq!(features.slice(s![12, .., ..]), history_5.slice(s![..; -1, ..]).t());
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

#[test]
fn score_one_hop_center() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...
    ...
    ...
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, 0);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2], 1.0);
}

#[test]
fn score_one_hop_max() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...
    ...
    ...
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, SCORE_ONE_HOP_SIZE as i32 + 1);

  assert_eq!(*score.last().unwrap(), 1.0);
}

#[test]
fn score_one_hop_min() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...
    ...
    ...
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, -(SCORE_ONE_HOP_SIZE as i32) - 1);

  assert_eq!(score[0], 1.0);
}

#[test]
fn score_one_hop_one() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aAa
    .a.
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, 0);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 + 1], 1.0);
}

#[test]
fn score_one_hop_minus_one() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .A.
    AaA
    .A.
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, 0);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 - 1], 1.0);
}

#[test]
fn score_one_hop_one_opposite() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aAa
    .a.
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Black, 0);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 - 1], 1.0);
}

#[test]
fn score_one_hop_fractional_komi() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...
    ...
    ...
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, 3);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 + 1], 0.5);
  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 + 2], 0.5);
}

#[test]
fn score_one_hop_fractional_negative_komi() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...
    ...
    ...
    ",
  );

  let score = score_one_hop::<f64>(&field, Player::Red, -3);

  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 - 1], 0.5);
  assert_eq!(score[SCORE_ONE_HOP_SIZE / 2 - 2], 0.5);
}
