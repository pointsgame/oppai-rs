use crate::uct::{UcbType, UctConfig, UctKomiType, UctRoot};
use env_logger;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

const UCT_CONFIG: UctConfig = UctConfig {
  threads_count: 1,
  radius: 3,
  ucb_type: UcbType::Ucb1Tuned,
  draw_weight: 0.4,
  uctk: 1.0,
  when_create_children: 2,
  depth: 8,
  komi_type: UctKomiType::Dynamic,
  red: 0.45,
  green: 0.5,
  komi_min_iterations: 3_000,
};

#[test]
fn find_best_move() {
  env_logger::try_init().ok();
  let mut rng = XorShiftRng::from_seed(SEED);
  let field = construct_field(
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
  let mut uct = UctRoot::new(UCT_CONFIG, field.length());
  let pos = uct.best_move_with_iterations_count(&field, Player::Red, &mut rng, 100_000);
  assert_eq!(pos, Some(field.to_pos(5, 2)));
}
