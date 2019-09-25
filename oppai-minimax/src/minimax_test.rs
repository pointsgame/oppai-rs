use crate::minimax::{Minimax, MinimaxConfig, MinimaxMovesSorting, MinimaxType};
use env_logger;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

const MINIMAX_CONFIG_NEGASCOUT: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::NegaScout,
  minimax_moves_sorting: MinimaxMovesSorting::TrajectoriesCount,
  hash_table_size: 10_000,
  rebuild_trajectories: false,
};

const MINIMAX_CONFIG_MTDF: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::MTDF,
  minimax_moves_sorting: MinimaxMovesSorting::TrajectoriesCount,
  hash_table_size: 10_000,
  rebuild_trajectories: false,
};

#[test]
fn find_best_move_1() {
  env_logger::try_init().ok();
  // 8 is the minimum depth value to detect correct move in this test.
  // With depth 7 after 3 moves we might have this position:
  // ........
  // .....a..
  // ...a....
  // ..AaAAa.
  // ...Aaa?.
  // ..A.A?..
  // ........
  // ........
  // Question marks here indicate trajectory that will be excluded because
  // it doesn't intersect any other trajectory with length 2.
  // Without this trajectory black player won't be able to find the escape.
  // So red player will think that he wins with move (5, 1).
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...a....
    ..AaA...
    ...Aaa..
    ..A.A...
    ........
    ........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}


#[test]
fn find_best_move_2() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...a.a..
    ...AAa..
    ...aAa..
    ....Aa..
    ...aaA..
    ........
    ........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(2, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(2, 3)));
}

#[test]
fn find_best_move_3() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...a....
    ..aA.a..
    ..aAA...
    ..aa....
    ........
    ........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 5)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 5)));
}

#[test]
fn find_best_move_4() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    ....a....
    .........
    ...Aa.A..
    ..A...A..
    ..AaaaA..
    ...AAAa..
    ......a..
    .........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
}

#[test]
fn find_best_move_5() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ...........
    ....aaa....
    ..AAa.A.A..
    .A.aAA...A.
    ...a.......
    ...a..a....
    ....aa.....
    ...........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 3)));
}

#[test]
fn find_best_move_6() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    ............
    ..A.a.......
    ...Aa..aa...
    ...aAaaaAA..
    ...aAAaA....
    ...a.A......
    ............
    ............
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(7, 6)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(7, 6)));
}

#[test]
#[ignore]
fn find_best_move_7() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    .......aa...
    .a...AaA.a..
    ..a.A.A.Aa..
    ..a..A.A.a..
    ...aaaaaa...
    ............
    ............
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 10);
  assert_eq!(pos, Some(field.to_pos(4, 1)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 10);
  assert_eq!(pos, Some(field.to_pos(4, 1)));
}

#[test]
fn find_best_move_8() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    ............
    .......AA...
    .....AAaaa..
    .....Aa.....
    ..A.Aa.a....
    ...Aa.A..a..
    ..Aa.a......
    ..Aa.a..A...
    ...AAAAA....
    ............
    ............
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 7)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 7)));
}

#[test]
#[ignore]
fn find_best_move_9() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ...........
    ...........
    ...aA...a..
    ..aA...a...
    ..aAA.a....
    ..aAAAAa...
    ..aaAaaA...
    ..AAaaAA...
    ....a......
    ...AaA.....
    ....A......
    ...........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 10);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 10);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
}

#[test]
fn find_best_move_10() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    ..........
    ....aaaA..
    .....AAa..
    ..A..A.a..
    ...A..a...
    ....A.a...
    .....Aa...
    ....Aa.a..
    ....Aa....
    ..........
    ..........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 6)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 6)));
}

#[test]
#[ignore]
fn find_best_move_11() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ...........
    ...........
    ..A........
    ..A........
    ..A...Aaa..
    ...AaaaA...
    ....AAA....
    ...........
    ...........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 12);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 12);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
}

#[test]
fn find_best_move_12() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ...........
    ...........
    ...a..a....
    ...AA.aAA..
    ...a.AAa...
    ...aaAaa...
    ..AAAa.....
    .....a.....
    ...........
    ...........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 3)));
}

#[test]
fn find_best_move_13() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .........
    .........
    ...AA.A..
    ...Aaa...
    ...Aa.A..
    ..aaAA...
    ....aa...
    .........
    .........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 5)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(6, 5)));
}

#[test]
fn find_best_move_14() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    ..........
    ...aa.....
    ..a..a....
    ..a...a...
    ..aAA.Aa..
    ..Aa..Aa..
    .....A.a..
    ...AA..a..
    ......a...
    ..........
    ..........
    ",
  );
  let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(4, 7)));
  let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
  let pos = minimax.minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(4, 7)));
}
