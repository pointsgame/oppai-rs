use crate::field_features::{
  CHANNELS, FEATURE_CHANNELS, GLOBAL_FEATURES, GLOBAL_SHORTFALL, SCORE_ONE_HOT_SIZE, add_unrotated_policy,
  field_features, global, rotation_sizes, score_one_hot,
};
use crate::tactics::{TACTICAL_CHANNELS, Tactics};
use ndarray::prelude::{Array2, array, s};
use oppai_field::construct_field::construct_field;
use oppai_field::field::Field;
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
fn score_one_hot_center() {
  let score = score_one_hot::<f64>(0, 0);

  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2], 1.0);
}

#[test]
fn score_one_hot_max() {
  let score = score_one_hot::<f64>(0, SCORE_ONE_HOT_SIZE as i32 + 1);

  assert_eq!(*score.last().unwrap(), 1.0);
}

#[test]
fn score_one_hot_min() {
  let score = score_one_hot::<f64>(0, -(SCORE_ONE_HOT_SIZE as i32) - 1);

  assert_eq!(score[0], 1.0);
}

#[test]
fn score_one_hot_one() {
  let score = score_one_hot::<f64>(1, 0);

  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 + 1], 1.0);
}

#[test]
fn score_one_hot_minus_one() {
  let score = score_one_hot::<f64>(-1, 0);

  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 - 1], 1.0);
}

#[test]
fn score_one_hot_fractional_komi() {
  let score = score_one_hot::<f64>(0, 3);

  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 + 1], 0.5);
  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 + 2], 0.5);
}

#[test]
fn score_one_hot_fractional_negative_komi() {
  let score = score_one_hot::<f64>(0, -3);

  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 - 1], 0.5);
  assert_eq!(score[SCORE_ONE_HOT_SIZE / 2 - 2], 0.5);
}

// A policy predicted for a rotated position reads back into the field's own
// coordinates: a net that just points at the mover's dots points at them on the
// field, whichever rotation it saw the field under - the transposing ones on a
// rectangle included, which only fit the padded square.
#[test]
fn unrotated_policy_points_back_at_the_field() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    aa.A.
    ..a..
    .A...
    ",
  );
  let (width, height) = rotation_sizes(field.width(), field.height());
  assert_eq!((width, height), (5, 5));
  let expected = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0)
    .slice(s![1, .., ..])
    .to_owned();
  for rotation in 0..oppai_rotate::rotate::ROTATIONS {
    let features = field_features::<f64>(&field, Player::Red, width, height, rotation);
    let mut policy = Array2::zeros((field.height() as usize, field.width() as usize));
    add_unrotated_policy(
      features.slice(s![1, .., ..]),
      field.width(),
      field.height(),
      rotation,
      1.0,
      &mut policy,
    );
    assert_eq!(policy, expected, "rotation {rotation}");
  }
}

/// The tactical planes, from channel 13 on: the cells where the player to move
/// and the opponent capture with one dot, then with two.
fn tactical_planes(field: &Field, player: Player) -> Vec<Array2<f64>> {
  let features = field_features::<f64>(field, player, field.width(), field.height(), 0);
  (0..TACTICAL_CHANNELS)
    .map(|plane| {
      features
        .slice(s![FEATURE_CHANNELS - TACTICAL_CHANNELS + plane, .., ..])
        .to_owned()
    })
    .collect()
}

#[test]
fn tactical_planes_mark_a_one_dot_capture() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .....
    ..a..
    .aA..
    ..a..
    .....
    ",
  );
  #[rustfmt::skip]
  let capture = array![
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
  ];
  let empty = Array2::<f64>::zeros((5, 5));

  assert_eq!(
    tactical_planes(&field, Player::Red),
    vec![capture.clone(), empty.clone(), empty.clone(), empty.clone()]
  );
  // The same capture, read from the side of the opponent.
  assert_eq!(
    tactical_planes(&field, Player::Black),
    vec![empty.clone(), capture, empty.clone(), empty]
  );
}

#[test]
fn tactical_planes_mark_a_two_dot_capture() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .....
    ..a..
    .aA..
    .....
    .....
    ",
  );
  #[rustfmt::skip]
  let pair = array![
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0.],
  ];
  let empty = Array2::<f64>::zeros((5, 5));

  assert_eq!(
    tactical_planes(&field, Player::Red),
    vec![empty.clone(), empty.clone(), pair.clone(), empty.clone()]
  );
  assert_eq!(
    tactical_planes(&field, Player::Black),
    vec![empty.clone(), empty.clone(), empty, pair]
  );
}

/// The tactical planes are read under the rotation like every other plane, so
/// pointing them back at the field gives the same planes whatever the
/// rotation, the transposing ones on a rectangle included.
#[test]
fn tactical_planes_rotate_with_the_board() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..a...
    .aA...
    ..a.A.
    ...Aa.
    ",
  );
  let (width, height) = rotation_sizes(field.width(), field.height());
  let expected = tactical_planes(&field, Player::Red);
  assert!(expected.iter().any(|plane| plane.sum() > 0.0));
  for rotation in 0..oppai_rotate::rotate::ROTATIONS {
    let features = field_features::<f64>(&field, Player::Red, width, height, rotation);
    for (plane, expected) in expected.iter().enumerate() {
      let mut unrotated = Array2::zeros((field.height() as usize, field.width() as usize));
      add_unrotated_policy(
        features.slice(s![FEATURE_CHANNELS - TACTICAL_CHANNELS + plane, .., ..]),
        field.width(),
        field.height(),
        rotation,
        1.0,
        &mut unrotated,
      );
      assert_eq!(&unrotated, expected, "rotation {rotation}, plane {plane}");
    }
  }
}

/// Building the trajectories plays and undoes moves, and leaves the field as
/// it was.
#[test]
fn tactics_leave_the_field_unchanged() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..a...
    .aA...
    ..a.A.
    ...Aa.
    ",
  );
  let state = |field: &Field| {
    (
      field.hash(),
      field.moves.clone(),
      field.score(Player::Red),
      field.non_grounded_red,
      field.non_grounded_black,
      (0..field.length()).map(|pos| field.cell(pos)).collect::<Vec<_>>(),
    )
  };
  let before = state(&field);
  let tactics = Tactics::new(&mut field, Player::Black);
  assert!((0..field.length()).any(|pos| tactics.plane(1, pos)));
  assert_eq!(state(&field), before);
}

// The shortfalls are each side's ungrounded dots minus its lead, komi
// included, from the view of the player to move first, and never negative.
#[test]
fn global_shortfalls() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .......
    .a.a...
    .......
    ....A..
    .......
    ",
  );
  assert_eq!(field.non_grounded_red, 2);
  assert_eq!(field.non_grounded_black, 1);

  let shortfalls = |player: Player, komi_x_2: i32| {
    let features = global::<f64>(&field, player, komi_x_2);
    assert_eq!(features.len(), GLOBAL_FEATURES);
    (features[GLOBAL_SHORTFALL] * 20.0, features[GLOBAL_SHORTFALL + 1] * 20.0)
  };
  assert_eq!(shortfalls(Player::Red, 0), (2.0, 1.0));
  assert_eq!(shortfalls(Player::Black, 0), (1.0, 2.0));
  // Half a point of komi for Red takes half a point off Red's shortfall and
  // adds it to Black's.
  assert_eq!(shortfalls(Player::Red, 1), (1.5, 1.5));
  assert_eq!(shortfalls(Player::Black, -1), (1.5, 1.5));
  // A lead beyond the ungrounded dots leaves no shortfall rather than a
  // negative one.
  assert_eq!(shortfalls(Player::Black, 6), (0.0, 5.0));
}

/// The chain plane of the player to move, or of the opponent, see
/// `push_chains`.
fn chain_plane(features: &ndarray::Array3<f64>, own: bool) -> Array2<f64> {
  features
    .slice(s![FEATURE_CHANNELS + if own { 0 } else { 1 }, .., ..])
    .to_owned()
}

#[test]
fn chain_planes_rank_the_chains_by_size() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    a.b..
    .c...
    ....d
    .A.e.
    ",
  );
  assert_eq!(CHANNELS, FEATURE_CHANNELS + 2);

  #[rustfmt::skip]
  let red = array![
    [1., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 2.],
    [0., 0., 0., 2., 0.],
  ];
  #[rustfmt::skip]
  let black = array![
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
  ];

  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(chain_plane(&features, true), red);
  assert_eq!(chain_plane(&features, false), black);
  let features = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  assert_eq!(chain_plane(&features, true), black);
  assert_eq!(chain_plane(&features, false), red);
}

// Chains of the same size go by their first cell in row order, of the board
// as rotated.
#[test]
fn chain_planes_order_equal_chains_by_their_first_cell() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...a
    b...
    ",
  );
  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  assert_eq!(chain_plane(&features, true), array![[0., 0., 0., 1.], [2., 0., 0., 0.]]);
  // Rotation 1 mirrors the columns.
  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 1);
  assert_eq!(chain_plane(&features, true), array![[1., 0., 0., 0.], [0., 0., 0., 2.]]);
}

#[test]
fn chain_planes_skip_captured_dots() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .b.
    cAd
    .e.
    ",
  );
  let features = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  #[rustfmt::skip]
  let red = array![
    [0., 1., 0.],
    [1., 0., 1.],
    [0., 1., 0.],
  ];
  assert_eq!(chain_plane(&features, true), red);
  assert_eq!(chain_plane(&features, false).sum(), 0.0);
}

/// Under every rotation, and padded to a square, the chain planes are the
/// planes of the board as it is, rotated: the chains here all differ in size,
/// so their ranks do not depend on the rotation.
#[test]
fn chain_planes_follow_the_rotation() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    a.b..
    .c...
    ....d
    .A.e.
    ",
  );
  let (width, height) = rotation_sizes(field.width(), field.height());
  let expected = field_features::<f64>(&field, Player::Red, field.width(), field.height(), 0);
  for rotation in 0..oppai_rotate::rotate::ROTATIONS {
    let features = field_features::<f64>(&field, Player::Red, width, height, rotation);
    for own in [true, false] {
      let mut unrotated = Array2::zeros((field.height() as usize, field.width() as usize));
      add_unrotated_policy(
        chain_plane(&features, own).view(),
        field.width(),
        field.height(),
        rotation,
        1.0,
        &mut unrotated,
      );
      assert_eq!(unrotated, chain_plane(&expected, own), "rotation {rotation}, own {own}");
    }
  }
}
