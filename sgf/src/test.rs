use crate::{from_sgf_str, to_sgf_str};
use oppai_field::{any_field::AnyField, construct_field::construct_field, field::Field, player::Player};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

#[test]
fn cross() {
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    ....
    .aB.
    .Dc.
    ....
    ",
  )
  .into();
  let sgf = to_sgf_str(&field).unwrap();
  assert_eq!(sgf, "(;GM[40]SZ[4:4]RU[russian];W[bb];B[cb];W[cc];B[bc])");
  let field_from_sgf: Field = from_sgf_str(sgf.as_ref(), &mut rng).unwrap();
  assert_eq!(field_from_sgf.moves, field.field().moves);
}

#[test]
fn simple_surround() {
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .a.
    cBa
    .a.
    ",
  )
  .into();
  let sgf = to_sgf_str(&field).unwrap();
  assert_eq!(
    sgf,
    "(;GM[40]SZ[3:3]RU[russian];W[ba];W[cb];W[bc];B[bb];W[ab.abbccbba])"
  );
  let field_from_sgf: Field = from_sgf_str(sgf.as_ref(), &mut rng).unwrap();
  assert_eq!(field_from_sgf.moves, field.field().moves);
}

#[test]
fn apply_control_surrounding_in_same_turn() {
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .a...
    aBa.c
    .a...
    ",
  )
  .into();
  let sgf = to_sgf_str(&field).unwrap();
  assert_eq!(
    sgf,
    "(;GM[40]SZ[5:3]RU[russian];W[ba];W[ab];W[cb];W[bc];B[bb];W[eb.abbccbba])"
  );
  let field_from_sgf: Field = from_sgf_str(sgf.as_ref(), &mut rng).unwrap();
  assert_eq!(field_from_sgf.moves, field.field().moves);
}

#[test]
fn apply_control_surrounding_in_same_turn_followed_by_simple_surrounding() {
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .a...a.
    aBa.cAa
    .a...a.
    ",
  )
  .into();
  let sgf = to_sgf_str(&field).unwrap();
  assert_eq!(
    sgf,
    "(;GM[40]SZ[7:3]RU[russian];B[fb];W[ba];W[fa];W[ab];W[cb];W[gb];W[bc];W[fc];B[bb];W[eb.ebfcgbfa.abbccbba])"
  );
  let field_from_sgf: Field = from_sgf_str(sgf.as_ref(), &mut rng).unwrap();
  assert_eq!(field_from_sgf.moves, field.field().moves);
}

#[test]
fn zagram352562() {
  // http://eidokropki.reaktywni.pl/index.phtml#url:zagram352562
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = from_sgf_str::<Field, _>(include_str!("tests/zagram352562.txt"), &mut rng).unwrap();
  assert_eq!(field.width, 39);
  assert_eq!(field.height, 32);
  assert_eq!(field.moves_count(), 260);
  assert_eq!(field.captured_count(Player::Red), 60);
  assert_eq!(field.captured_count(Player::Black), 3);
}
