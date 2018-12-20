use crate::construct_field::{construct_field, DEFAULT_SEED};
use crate::player::Player;
use crate::uct::UctRoot;
use env_logger;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

#[test]
fn find_best_move() {
  env_logger::try_init().ok();
  let field = construct_field(
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
  let mut uct = UctRoot::new(field.length());
  let pos = uct.best_move_with_iterations_count(&field, Player::Red, &mut rng, 500_000);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
