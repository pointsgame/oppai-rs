use crate::Config;
use oppai_field::construct_field::construct_field;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

#[test]
fn square() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
      .......
      .ab.HG.
      .dc.FE.
      .......
      ",
  );
  let config = Config {
    width: 128,
    height: 128,
    ..Default::default()
  };
  let document = super::field_to_svg(&config, &field.into());
  assert_eq!(format!("{}", document), include_str!("tests/square.svg"));
}

#[test]
fn nested() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
      ...X...
      ..VUT..
      .WkjiS.
      LdCABhR
      .MefgQ.
      ..NOP..
      ...Y...
      ",
  );
  let config = Config {
    width: 128,
    height: 128,
    ..Default::default()
  };
  let document = super::field_to_svg(&config, &field.into());
  assert_eq!(format!("{}", document), include_str!("tests/nested.svg"));
}
