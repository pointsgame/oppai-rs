use crate::{from_sgf, to_sgf};
use oppai_field::{construct_field::construct_field, field::Field, player::Player};
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
  );
  let sgf = to_sgf(&field).unwrap();
  assert_eq!(sgf, "(;GM[40]SZ[4:4];W[bb];B[cb];W[cc];B[bc])");
  let field_from_sgf: Field = from_sgf(sgf.as_ref(), &mut rng).unwrap();
  assert_eq!(field_from_sgf.points_seq(), field.points_seq());
}

#[test]
fn zagram352562() {
  // http://eidokropki.reaktywni.pl/index.phtml#url:zagram352562
  env_logger::try_init().ok();
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = from_sgf::<Field, _>(include_str!("tests/zagram352562.txt"), &mut rng).unwrap();
  assert_eq!(field.width(), 39);
  assert_eq!(field.height(), 32);
  assert_eq!(field.moves_count(), 260);
  assert_eq!(field.captured_count(Player::Red), 60);
  assert_eq!(field.captured_count(Player::Black), 3);
}
