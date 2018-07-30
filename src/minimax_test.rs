use env_logger;
use rand::XorShiftRng;
use player::Player;
use hash_table::HashTable;
use minimax::minimax;
use config::{MinimaxType, set_minimax_type};
use construct_field::construct_field;

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
    "
  );
  let mut rng = XorShiftRng::new_unseeded();
  let hash_table = HashTable::new(1000);
  set_minimax_type(MinimaxType::NegaScout);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
  set_minimax_type(MinimaxType::MTDF);
  let pos = minimax(&mut field, Player::Red, &hash_table, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
