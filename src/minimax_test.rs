use rand::XorShiftRng;
use player::Player;
use minimax::minimax;
use construct_field::construct_field;

#[test]
fn find_best_move() {
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
  let pos1 = minimax(&mut field, Player::Red, &mut rng, 5);
  assert_eq!(pos1, Some(field.to_pos(5, 2)));
  let pos2 = minimax(&mut field, Player::Red, &mut rng, 6);
  assert_eq!(pos2, Some(field.to_pos(5, 2)));
  let pos3 = minimax(&mut field, Player::Red, &mut rng, 7);
  assert_eq!(pos3, Some(field.to_pos(5, 2)));
  let pos4 = minimax(&mut field, Player::Red, &mut rng, 8);
  assert_eq!(pos4, Some(field.to_pos(5, 2)));
}
