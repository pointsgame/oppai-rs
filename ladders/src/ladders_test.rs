use crate::ladders::ladders;
use oppai_field::construct_field::construct_field;
use oppai_field::field::NonZeroPos;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

#[test]
fn ladders_escape() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    .........
    .........
    .........
    ..aA.....
    .aAAa....
    ..aa.....
    .........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);
}

#[test]
fn ladders_capture_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    ......a..
    .........
    .........
    ..aA.....
    .aAAa....
    ..aa.....
    .........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(3, 3)));
  assert_eq!(score, 3);
  assert_eq!(depth, 5);
}

#[test]
fn ladders_capture_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    ......a..
    .........
    .........
    .........
    .aAAa....
    ..aa.....
    .........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 4)));
  assert_eq!(score, 2);
  assert_eq!(depth, 6);
}

#[test]
fn ladders_capture_3() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    .......a.
    .........
    .........
    .........
    .aAAa....
    ..aa.....
    .........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 4)));
  assert_eq!(score, 2);
  assert_eq!(depth, 7);
}

#[test]
fn ladders_capture_4() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    .........
    ........a
    ........a
    .........
    .aAAa....
    ..aa.....
    .........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 4)));
  assert_eq!(score, 2);
  assert_eq!(depth, 8);
}

#[test]
fn ladders_side_capture_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..............
    ...........aa.
    .............a
    .............a
    ...........Aa.
    .......a..AAa.
    ........aaaa..
    .aAAa.........
    ..aa..........
    ..............
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 6)));
  assert_eq!(score, 2);
  assert_eq!(depth, 7);
}

#[test]
fn ladders_side_capture_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..............
    ..............
    ............a.
    ..............
    ..............
    ..............
    .....aa.......
    .aa.aAAA......
    .AA..AAa......
    .aa.aAa.......
    .....a........
    ..............
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(3, 8)));
  assert_eq!(score, 2);
  assert_eq!(depth, 10);
}

#[test]
fn ladders_shift() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ...........
    .........a.
    ...........
    .......a...
    ...........
    .aAAa......
    ..aa.......
    ...........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 4)));
  assert_eq!(score, 2);
  assert_eq!(depth, 9);
}

#[test]
fn ladders_rotate() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    .........a..
    .......a....
    ............
    .aAAa.......
    ..aa......a.
    ............
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 3)));
  assert_eq!(score, 2);
  assert_eq!(depth, 11);
}

#[test]
fn ladders_fork() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ......
    .aa...
    .AA...
    .aaAa.
    ......
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(3, 2)));
  assert_eq!(score, 1);
  assert_eq!(depth, 1);
}

#[test]
fn ladders_fork_deep() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..a...
    ..A...
    .a....
    .aAAa.
    ..aa..
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(2, 2)));
  assert_eq!(score, 1);
  assert_eq!(depth, 2);
}

#[test]
fn ladders_fork_stupid() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .......
    .aa.aa.
    .AA.AA.
    .aaAaa.
    .......
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);
}

#[test]
fn ladders_stupid() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    ........a.
    ..........
    .Aaa......
    ..AAAa....
    .Aaaa.....
    ..........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);
}

#[test]
fn ladders_not_viable_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    ........a.
    ..........
    .AaaA.....
    .aAAAa....
    .Aaaa.....
    ..........
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);
  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);
}

#[test]
fn ladders_viable() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    ......a..
    .........
    .........
    ..aa.....
    .aAAA....
    ..aAAa...
    .aAAa....
    ..aa.....
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);
  assert_eq!(pos, NonZeroPos::new(field.to_pos(5, 5)));
  assert_eq!(score, 7);
  assert_eq!(depth, 5);
}

#[test]
fn ladders_not_viable_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    .......a..
    ..........
    ..........
    ...aa.....
    .AaAAA....
    ...aAAa...
    .AaAAa....
    ...aa.....
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);
  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);
}

#[test]
fn ladders_viable_multi() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .................
    .......aaaa......
    ......a....aAA...
    .....aA.....aAA..
    .....AA......aAA.
    .....AA.....aAA..
    .....aa....aAA...
    .a.....aaaa......
    .................
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(4, 4)));
  assert_eq!(score, 5);
  assert_eq!(depth, 5);
}

#[test]
fn ladders_viable_complex() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    ...AA.a..
    ..A.a....
    .Aaa.....
    .aA..Aa..
    .aA..aA..
    .aA..A...
    .aAAAaa..
    ..aaa....
    .........
    ",
  );

  let (_, score, _) = ladders(&mut field, Player::Red, &|| false);
  // It's possible to capture 8 points here but current method is
  // limited - it doesn't consider ladders after captures.
  assert_eq!(score, 6);
}

#[test]
fn ladders_depth_choice() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .............
    .............
    .............
    .............
    .............
    .a.aaa.......
    ...AAA.......
    ..aaaaa......
    .............
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);
  assert_eq!(pos, NonZeroPos::new(field.to_pos(6, 6)));
  assert_eq!(score, 3);
  assert_eq!(depth, 2);
}
