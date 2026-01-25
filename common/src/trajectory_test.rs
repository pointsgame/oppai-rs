use crate::trajectory::{build_trajectories, build_trajectories_from};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use smallvec::SmallVec;

const SEED: u64 = 7;

#[test]
fn build_trajectories_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .......
    ..a.a..
    .a...a.
    .a.A.a.
    ..aaa..
    .......
    ",
  );

  let trajectories = build_trajectories::<1, _, SmallVec<[_; 3]>>(&mut field, Player::Red, 1, &|| false);

  assert_eq!(trajectories.len(), 3);
}

#[test]
fn build_trajectories_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ..a..a..
    .a....a.
    .a.AA.a.
    ..aaaa..
    ........
    ",
  );

  let trajectories = build_trajectories::<2, _, SmallVec<[_; 7]>>(&mut field, Player::Red, 2, &|| false);

  assert_eq!(trajectories.len(), 7);
}

#[test]
fn build_trajectories_3() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    .........
    ..a...a..
    .a.....a.
    .a.....a.
    .a.AAA.a.
    ..aaaaa..
    .........
    ",
  );

  let trajectories = build_trajectories::<3, _, SmallVec<[_; 19]>>(&mut field, Player::Red, 3, &|| false);

  assert_eq!(trajectories.len(), 19);
}

#[test]
fn build_trajectories_with_no_extra_points() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .......
    .a.....
    .a.....
    .a.aa..
    .aAAAa.
    .aAAAa.
    ..aaa..
    .......
    ",
  );

  let trajectories = build_trajectories::<2, _, SmallVec<[_; 3]>>(&mut field, Player::Red, 2, &|| false);

  assert_eq!(trajectories.len(), 3);
}

#[test]
fn build_trajectories_through_empty_base() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ....aa..
    ..A...a.
    .A.A..a.
    .aA...a.
    .a....a.
    ..aaaa..
    ........
    ",
  );

  let trajectories = build_trajectories::<2, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 2, &|| false);

  assert_eq!(trajectories.len(), 1);
}

#[test]
fn build_trajectories_crankle_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .A.AAA.AAA.AAA.
    ...............
    aAAA.AAA.AAA.A.
    .A.AAA.AAA.AAA.
    ...............
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 512]>>(&mut field, Player::Red, 29, &|| false);

  assert_eq!(trajectories.len(), 512);
}

#[test]
fn build_trajectories_crankle_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .AAAAAAAAAAAAA.
    .A...A...A...A.
    .A.A.A.A.A.A.A.
    ...A...A...A...
    aAAAAAAAAAAAAA.
    ...............
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 27, &|| false);

  assert_eq!(trajectories.len(), 1);
}

#[test]
fn build_trajectories_crankle_3() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    aAAAAAAAAAAA.
    a..........A.
    aAAAAAAAAA.A.
    .Aa........A.
    .AaAAAAAAAAA.
    .Aa.......aA.
    .AAAAAAAAAaA.
    .A........aA.
    .A.AAAAAAAAA.
    .A...........
    .AAAAAAAAAAA.
    .............
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 61, &|| false);

  assert_eq!(trajectories.len(), 1);
}

#[test]
fn build_trajectories_maze_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    aAAAAAAAAAAAAAAAAAAAAAa
    a....A...A.A.A.A...A.Aa
    aAAA.A.A.A.A.A.AAA.A.Aa
    aA.A.A.A.....A.....A.Aa
    aA.A.A.AAA.AAAAA.AAA.Aa
    aA...A.A.A.....A.A...Aa
    aAAA.AAA.AAA.A.A.A.AAAa
    aA...A.A.....A.......Aa
    aA.AAA.A.AAA.AAAAA.AAAa
    aA.A.....A.A.A.......Aa
    aA.A.AAA.A.A.AAA.A.A.Aa
    aA.....A...A.A...A.A.Aa
    aAAAAAAA.AAA.A.AAA.AAAa
    aA.A...A.A.A.A.A.A...Aa
    aA.A.AAA.A.AAAAA.AAA.Aa
    aA.....A.A.....A.A.A.Aa
    aA.AAAAAAA.AAAAA.A.A.Aa
    aA.....A...A...A.....Aa
    aAAA.A.A.AAA.A.A.AAA.Aa
    aA...A.......A...A....a
    aAAAAAAAAAAAAAAAAAAAAAa
    aaaaaaaaaaaaaaaaaaaaaaa
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 39, &|| false);

  assert!(!trajectories.is_empty());
}

#[test]
fn build_trajectories_maze_2() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
    a....A.......A...A.............Aa
    aAAA.A.AAAAAAAAA.AAAAAAAAA.AAA.Aa
    aA...A...A.....A...A.A.....A...Aa
    aA.AAA.A.AAAAA.A.AAA.A.AAA.A.A.Aa
    aA.....A.........A.....A.A.A.A.Aa
    aA.AAA.A.AAAAAAAAAAAAA.A.A.AAA.Aa
    aA...A.A.A.A...........A...A...Aa
    aAAA.A.A.A.A.AAA.AAAAAAAAAAAAAAAa
    aA...A.A...A.A.A...A.......A...Aa
    aA.AAAAA.A.A.A.AAA.AAA.AAAAAAA.Aa
    aA...A...A.A.A...A.A...........Aa
    aAAAAAAA.A.A.A.AAA.AAAAA.A.AAAAAa
    aA.A...A.A.......A.A.A...A.A...Aa
    aA.A.AAAAAAA.A.A.A.A.AAAAA.A.AAAa
    aA.A.....A...A.A.A.............Aa
    aA.A.A.AAAAAAAAA.A.A.A.A.A.AAA.Aa
    aA...A.A.........A.A.A.A.A.A.A.Aa
    aA.AAA.AAAAA.A.A.AAAAAAA.A.A.AAAa
    aA...A.A.A.A.A.A...A.A.A.A.....Aa
    aAAA.AAA.A.AAA.A.AAA.A.A.A.AAA.Aa
    aA...........A.A...A.....A.A...Aa
    aA.AAAAA.AAAAA.A.A.A.A.A.A.A.A.Aa
    aA.A...........A.A.A.A.A.A.A.A.Aa
    aA.A.AAAAA.A.AAAAAAAAAAAAAAA.AAAa
    aA.A...A...A...A...A.A.A.......Aa
    aA.AAA.AAAAAAA.AAA.A.A.A.A.AAAAAa
    aA.A.A.......A...A.......A.A.A.Aa
    aA.A.AAAAA.A.AAA.AAA.AAA.AAA.A.Aa
    aA.....A...A...A.A.....A........a
    aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 67, &|| false);

  assert!(!trajectories.is_empty());
}

#[test]
#[ignore]
fn build_trajectories_maze_3() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    a..A.A.....A.......A.......A...A...A.A...A
    aA.A.A.AAA.A.AAA.AAA.AAAAA.A.AAA.AAA.A.AAA
    aA.......A.....A...A.A.A.A.....A...A.A.A.A
    aA.AAA.AAA.AAA.A.A.AAA.A.A.AAAAA.A.A.A.A.A
    aA.A.A...A...A.A.A...A.......A...A...A...A
    aAAA.AAA.AAA.AAA.AAA.A.A.AAA.AAA.AAA.A.AAA
    aA.....A.A.A...A...A.A.A.A...A.A...A.A...A
    aAAA.A.A.A.A.A.A.A.A.A.AAAAA.A.A.AAAAAAA.A
    aA.A.A.A.A...A.A.A.A.A...A.............A.A
    aA.A.AAA.A.AAA.AAAAAAA.AAA.A.AAA.AAAAAAA.A
    aA.....A.A.A.....A.A.A.A...A.A...A.A.....A
    aAAA.AAAAAAAAA.AAA.A.AAAAAAA.AAA.A.A.AAAAA
    aA...A...A.....A...A...A.......A.......A.A
    aA.AAAAA.A.AAA.AAA.A.AAA.A.A.AAA.A.A.AAA.A
    aA.......A.A.A.A.A.A.A...A.A.A.A.A.A.A.A.A
    aA.AAA.AAAAA.A.A.A.A.AAA.A.AAA.AAAAA.A.A.A
    aA...A.A.....A...........A.......A...A...A
    aAAAAA.A.AAA.A.A.AAAAAAAAA.AAAAAAA.AAA.AAA
    aA.A.....A.A...A.......A.......A.A.A.A.A.A
    aA.A.A.AAA.AAAAA.AAAAAAA.AAAAAAA.A.A.A.A.A
    aA.A.A.....A.A.A.....A.A.A.....A.....A.A.A
    aA.AAAAAAA.A.A.AAA.AAA.AAAAA.AAA.AAAAA.A.A
    aA.A...A.A.....A.A...A.A...A...A.A.A.....A
    aA.A.AAA.AAAAA.A.A.AAA.AAA.AAA.A.A.AAA.AAA
    aA...A.....A.....A.A.A.......A...........A
    aAAA.AAA.AAA.AAA.A.A.A.AAA.A.AAAAA.A.AAAAA
    aA.............A.A...A...A.A.A.....A.....A
    aA.AAAAA.AAAAA.AAA.AAAAAAA.AAAAAAAAAAA.AAA
    aA...A...A.A.....A.A.......A...A.A...A.A.A
    aAAAAA.AAA.A.AAA.AAA.AAA.AAAAA.A.AAA.A.A.A
    aA.A...A.......A.A...A.....A.A.A.....A...A
    aA.A.A.AAAAAAAAA.A.A.A.A.A.A.A.AAA.AAAAAAA
    aA...A.........A...A.A.A.A.........A.A...A
    aAAA.A.AAA.AAAAAAA.AAAAAAAAA.AAA.AAA.AAA.A
    aA...A.A.A.A.......A.A.A.A...A.A.A.......A
    aAAAAAAA.AAA.A.AAAAA.A.A.A.AAA.A.AAA.AAA.A
    aA.........A.A.....A.A.A.....A.A.....A...A
    aA.A.AAAAA.A.AAAAA.A.A.AAAAA.A.A.A.AAA.A.A
    aA.A...A.....A.............A...A.A...A.A.A
    aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.A
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ",
  );

  let trajectories = build_trajectories::<0, _, SmallVec<[_; 1]>>(&mut field, Player::Red, 80, &|| false);

  assert!(!trajectories.is_empty());
}

#[test]
fn build_trajectories_from_1() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ..a..a..
    .AAaaAA.
    .aa..aa.
    ........
    ",
  );

  let pos = field.to_pos(2, 1);
  let trajectories = build_trajectories_from::<2, _, SmallVec<[_; 1]>>(&mut field, pos, Player::Red, 2, &|| false);
  assert_eq!(trajectories.len(), 1);

  let pos = field.to_pos(3, 2);
  let trajectories = build_trajectories_from::<2, _, SmallVec<[_; 1]>>(&mut field, pos, Player::Red, 2, &|| false);
  assert_eq!(trajectories.len(), 1);

  let pos = field.to_pos(5, 1);
  let trajectories = build_trajectories_from::<2, _, SmallVec<[_; 1]>>(&mut field, pos, Player::Red, 2, &|| false);
  assert_eq!(trajectories.len(), 1);

  let pos = field.to_pos(4, 2);
  let trajectories = build_trajectories_from::<2, _, SmallVec<[_; 1]>>(&mut field, pos, Player::Red, 2, &|| false);
  assert_eq!(trajectories.len(), 1);
}
