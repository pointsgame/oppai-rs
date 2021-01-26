use crate::common::{is_last_move_stupid, is_penult_move_stupid};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

#[test]
fn is_last_move_stupid_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .....
    ..A..
    .AbA.
    ..A..
    .....
    ",
  );

  let pos = field.to_pos(2, 2);
  assert!(is_last_move_stupid(&field, pos, Player::Red));
}

#[test]
fn is_last_move_stupid_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .....
    .AbA.
    ..A..
    .....
    ",
  );

  let pos = field.to_pos(2, 1);
  assert!(is_last_move_stupid(&field, pos, Player::Red));
}

#[test]
fn is_last_move_not_stupid() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .....
    .Aba.
    ..A..
    .....
    ",
  );

  let pos = field.to_pos(2, 1);
  assert!(!is_last_move_stupid(&field, pos, Player::Red));
}

#[test]
fn is_penult_move_stupid_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .....
    ..C..
    .AbA.
    ..A..
    .....
    ",
  );

  assert!(is_penult_move_stupid(&field));
}

#[test]
fn is_penult_move_stupid_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .......
    ..ACA..
    .A...A.
    .A.b.A.
    ..AAA..
    .......
    ",
  );

  assert!(is_penult_move_stupid(&field));
}
