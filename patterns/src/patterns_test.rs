use crate::patterns::Patterns;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::rngs::SmallRng;
use rand::SeedableRng;

const SEED: u64 = 99991;

fn construct_patterns(strings: &[&str]) -> Patterns {
  let mut p = Patterns::empty();
  for s in strings {
    p = p.union(&Patterns::from_str(s).unwrap());
  }
  p
}

#[test]
#[should_panic]
fn pattern_without_moves_on_image() {
  construct_patterns(&["
    ....
    .XO.
    .OX.
    ....
    "]);
}

#[test]
fn pattern_empty_doesnt_match() {
  let p = construct_patterns(&["
    #...
    #XO.
    #O..
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
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
    #...
    #XO.
    #OX.
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 3)]);
}

#[test]
fn pattern_borders_doesnt_match() {
  let p = construct_patterns(&["
    #...
    #XO.
    #OX.
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
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
    #...
    #XO.
    ****
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 3)]);
}

#[test]
fn pattern_any_except_border_matches() {
  let p = construct_patterns(&["
    #...
    #XO.
    #???
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 3)]);
}

#[test]
fn pattern_any_except_border_doesnt_match() {
  let p = construct_patterns(&["
    #...
    #XO.
    ????
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
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
    #...
    #Xoo
    #Oxx
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    ...
    aA.
    Aa.
    ...
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 3)]);
}

#[test]
fn pattern_red_black_or_none_doesnt_match() {
  let p = construct_patterns(&["
    #...
    #ooo
    #xxx
    #.+.
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
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
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(0, 1)]);
}

#[test]
fn pattern_rotation_1() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    ...a.
    .Aa..
    .aAA.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 4)]);
}

#[test]
fn pattern_rotation_2() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .a.A.
    ..aA.
    ..Aa.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(4, 3)]);
}

#[test]
fn pattern_rotation_3() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .AAa.
    ..aA.
    .a...
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(3, 0)]);
}

#[test]
fn pattern_rotation_4() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    ..Aa.
    ..aA.
    .a.A.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(4, 1)]);
}

#[test]
fn pattern_rotation_5() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .aAA.
    .Aa..
    ...a.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(1, 0)]);
}

#[test]
fn pattern_rotation_6() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .A.a.
    .Aa..
    .aA..
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(0, 3)]);
}

#[test]
fn pattern_rotation_7() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .a...
    ..aA.
    .AAa.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![field.to_pos(3, 4)]);
}

#[test]
fn pattern_inversion_doesnt_match() {
  let p = construct_patterns(&["
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
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
    .....
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .Aa..
    .aA..
    .a.A.
    .....
    ",
  );
  assert_eq!(p.find(&field, Player::Black, false), vec![field.to_pos(0, 1)]);
}

#[test]
fn pattern_multiple_moves() {
  let p = construct_patterns(&["
    .+...
    +XO..
    .OX..
    .O.X.
    .....
    "]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  let mut moves = p.find(&field, Player::Red, false);
  moves.sort();
  assert_eq!(moves, vec![field.to_pos(1, 0), field.to_pos(0, 1)]);
}

#[test]
fn multiple_patterns() {
  let p = construct_patterns(&[
    "
    .+...
    +XO..
    .OX..
    .O.X.
    .....
    ",
    "
    ???..
    ?xo..
    .ox..
    .O.X+
    ...+.
    ",
  ]);
  let field = construct_field(
    &mut SmallRng::seed_from_u64(SEED),
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    ",
  );
  let mut moves = p.find(&field, Player::Red, false);
  moves.sort();
  assert_eq!(
    moves,
    vec![
      field.to_pos(1, 0),
      field.to_pos(0, 1),
      field.to_pos(4, 3),
      field.to_pos(3, 4),
    ]
  );
}
