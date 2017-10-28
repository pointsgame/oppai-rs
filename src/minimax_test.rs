use rand::XorShiftRng;
use player::Player;
use minimax::minimax;
use construct_field::construct_field;

#[test]
fn find_best_move() {
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
  let pos = minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
