use crate::config::{set_minimax_type, MinimaxType};
use crate::construct_field::{construct_field, DEFAULT_SEED};
use crate::hash_table::HashTable;
use crate::minimax::minimax;
use crate::player::Player;
use env_logger;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

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
  let mut field = construct_field(
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
  let mut rng = XorShiftRng::from_seed(DEFAULT_SEED);
  let hash_table = HashTable::new(1000);
  set_minimax_type(MinimaxType::NegaScout);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
  set_minimax_type(MinimaxType::MTDF);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
