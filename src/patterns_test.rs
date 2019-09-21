use crate::patterns::Patterns;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

fn construct_patterns(strings: &[&str]) -> Patterns {
  let mut p = Patterns::empty();
  for s in strings {
    p = p.union(&Patterns::from_str("<none>", s));
  }
  p
}

#[test]
#[should_panic]
fn pattern_moves_discrepancy() {
  construct_patterns(&["
    4 4 1.0
    ....
    .RB.
    .BR.
    .+..
    2 3 1.0
    "]);
}

#[test]
#[should_panic]
fn pattern_without_moves_on_image() {
  construct_patterns(&["
    4 4 1.0
    ....
    .RB.
    .BR.
    ....
    2 3 1.0
    "]);
}

#[test]
#[should_panic]
fn pattern_with_less_moves_than_on_image() {
  construct_patterns(&["
    4 4 1.0
    ....
    .RB.
    .BR.
    .++.
    2 3 1.0
    "]);
}

#[test]
fn pattern_empty_doesnt_match() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    #B..
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_borders_matches() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    #BR.
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_borders_doesnt_match() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    #BR.
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ....
    .aA.
    .Aa.
    ....
    ",
  );
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_any_matches() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    ****
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_any_except_border_matches() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    #???
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_any_except_border_doesnt_match() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #RB.
    ????
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_red_black_or_none_matches() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #Rbb
    #Brr
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_red_black_or_none_doesnt_match() {
  let p = construct_patterns(&["
    4 4 1.0
    #...
    #bbb
    #rrr
    #.+.
    2 3 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_rotation_0() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(0, 1), 1f64)]);
}

#[test]
fn pattern_rotation_1() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    ...a.
    .Aa..
    .aAA.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 4), 1f64)]);
}

#[test]
fn pattern_rotation_2() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .a.A.
    ..aA.
    ..Aa.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(4, 3), 1f64)]);
}

#[test]
fn pattern_rotation_3() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .AAa.
    ..aA.
    .a...
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(3, 0), 1f64)]);
}

#[test]
fn pattern_rotation_4() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    ..Aa.
    ..aA.
    .a.A.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(4, 1), 1f64)]);
}

#[test]
fn pattern_rotation_5() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .aAA.
    .Aa..
    ...a.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 0), 1f64)]);
}

#[test]
fn pattern_rotation_6() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .A.a.
    .Aa..
    .aA..
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(0, 3), 1f64)]);
}

#[test]
fn pattern_rotation_7() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .a...
    ..aA.
    .AAa.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(3, 4), 1f64)]);
}

#[test]
fn pattern_inversion_doesnt_match() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  assert!(p.find(&field, Player::Black, false).is_empty());
}

#[test]
fn pattern_inversion_matches() {
  let p = construct_patterns(&["
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .Aa..
    .aA..
    .a.A.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Black, false), vec![(field.to_pos(0, 1), 1f64)]);
}

#[test]
fn pattern_multiple_moves() {
  let p = construct_patterns(&["
    5 5 5.0
    .+...
    +RB..
    .BR..
    .B.R.
    .....
    0 1 3.0
    1 0 1.0
    "]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  assert_eq!(
    p.find_sorted(&field, Player::Red, false),
    vec![(field.to_pos(0, 1), 0.75f64), (field.to_pos(1, 0), 0.25f64)]
  );
  assert_eq!(p.find_foreground(&field, Player::Red, false), Some(field.to_pos(0, 1)));
}

#[test]
fn multiple_patterns() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .+...
    +RB..
    .BR..
    .B.R.
    .....
    0 1 3.0
    1 0 1.0
    ",
    "
    5 5 4.0
    ???..
    ?rb..
    .br..
    .B.R+
    ...+.
    4 3 1.0
    3 4 3.0
    ",
  ]);
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  assert_eq!(
    p.find_sorted(&field, Player::Red, false),
    vec![
      (field.to_pos(3, 4), 0.6f64),
      (field.to_pos(4, 3), 0.2f64),
      (field.to_pos(0, 1), 0.15f64),
      (field.to_pos(1, 0), 0.05f64)
    ]
  );
  assert_eq!(p.find_foreground(&field, Player::Red, false), Some(field.to_pos(3, 4)));
}
