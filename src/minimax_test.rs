use crate::config::{set_minimax_type, MinimaxType};
use crate::hash_table::HashTable;
use crate::minimax::minimax;
use env_logger;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[test]
fn find_best_move() {
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
  // Question marks here are indicated trajectory that will be excluded
  // because it doesn't intersect any other trajectory with length 2.
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
  let hash_table = HashTable::new(1000);
  set_minimax_type(MinimaxType::NegaScout);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
  set_minimax_type(MinimaxType::MTDF);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
