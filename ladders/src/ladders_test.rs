use std::collections::HashSet;

use crate::ladders::{ladder_moves, ladders};
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(3, 3)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 4)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 4)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 4)]));
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
  // the longest line the defender can force for the same score
  assert_eq!(depth, 9);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 6)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(3, 8)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 4)]));
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
  assert_eq!(depth, 13);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 3)]));
}

// The ladder rotates around the black group, but near the end black 5:4
// captures the red point at 5:3 which is a part of the red ring.
#[test]
fn ladders_fake_rotate() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    ............
    .......a....
    ............
    .aAAa.......
    ..aa......a.
    ............
    ............
    ............
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(3, 2)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(2, 2)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
}

// After red 5:5 black 1:6 captures the red point at 2:6 which is a part of
// the red ring instead of blocking.
#[test]
fn ladders_not_viable_3() {
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
  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
}

// After red 4:4 black 12:4 captures the red point at 13:4 which is a part of
// the big red ring instead of blocking.
#[test]
fn ladders_not_viable_multi() {
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

  assert_eq!(pos, None);
  assert_eq!(score, 0);
  assert_eq!(depth, 0);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::new());
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
  assert_eq!(score, 7);

  // 4:4, 4:6 and 6:6 also start capturing trajectories, but black 4:5
  // captures the red point at 5:5 which is a part of the red ring
  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(5, 3)]));
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

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(6, 6)]));
}

#[test]
fn ladders_two() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..................
    ......a........a..
    ..................
    ..................
    ..aA.......aA.....
    .aAAa.....aAAa....
    ..aa.......aa.....
    ..................
    ",
  );

  let (pos, score, depth) = ladders(&mut field, Player::Red, &|| false);

  assert_eq!(pos, NonZeroPos::new(field.to_pos(3, 3)));
  assert_eq!(score, 3);
  assert_eq!(depth, 5);

  let moves = ladder_moves(&mut field, Player::Red, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(3, 3), field.to_pos(12, 3)]));
}

// Red threatens 6:9 or 7:9 capturing 5:9 and 5:10. Black 3:9 captures 4:9 and
// 4:10 right away and removes the threat since they are a part of the red
// ring. Black 2:9 starts a ladder that needs red to block at 2:8, but red
// captures with 6:9 instead, breaking the black ring.
#[test]
fn ladders_counter_capture() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    ............
    ....b.......
    ..bbbB......
    ..BbBB......
    ...Ba.B.....
    ...BaBaB....
    ...BBbBbb...
    ....Bbbb.b..
    ....bB..b...
    ...BbBbBB...
    ....Bb......
    ............
    ............
    ",
  );
  let current = field.score(Player::Black);

  let (pos, score, depth) = ladders(&mut field, Player::Black, &|| false);
  assert_eq!(pos, NonZeroPos::new(field.to_pos(3, 9)));
  assert_eq!(score, current + 2);
  assert_eq!(depth, 0);

  let moves = ladder_moves(&mut field, Player::Black, &|| false);
  assert_eq!(moves, HashSet::from([field.to_pos(3, 9)]));
}

// Black chases the red chain 6:13, 7:14, 8:13 with 5:13, 6:14 and 7:13, and
// 8:14 would threaten 7:15 capturing 7:14. But red 7:12 captures 7:13 which
// is a part of the black ring, so the ladder loses a point instead.
#[test]
fn ladders_counter_capture_deep() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .................
    .................
    ....BBBBBB.......
    ...bbbBbbB..b....
    ..bbAbBbbBbbB.b..
    ....bbBbbbbAbb...
    ..B..BbBBbBb.....
    ....BBb.BBBBB....
    ....Bbb..........
    ....BbAbB........
    ....Bbb..B.......
    ....BbAb.........
    ....Bbb..........
    .....BbBb........
    ......Bb.........
    .................
    .................
    ",
  );
  let current = field.score(Player::Black);

  let (pos, score, depth) = ladders(&mut field, Player::Black, &|| false);
  assert_eq!(pos, None);
  assert_eq!(score, current);
  assert_eq!(depth, 0);

  let moves = ladder_moves(&mut field, Player::Black, &|| false);
  assert_eq!(moves, HashSet::new());
}
