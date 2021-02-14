use crate::uct::{UcbType, UctConfig, UctKomiType, UctRoot};
use oppai_field::construct_field::construct_field;
use oppai_field::field::NonZeroPos;
use oppai_field::player::Player;
use oppai_test_images::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::atomic::AtomicBool;

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

macro_rules! uct_test {
  ($(#[$($attr:meta),+])* $name:ident, $image:ident, $iterations:expr, $seed:expr) => {
    #[test]
    $(#[$($attr),+])*
    fn $name() {
      env_logger::try_init().ok();
      let mut rng = Xoshiro256PlusPlus::seed_from_u64($seed);
      let field = construct_field(&mut rng, $image.image);
      let mut uct = UctRoot::new(UCT_CONFIG, field.length());
      let should_stop = AtomicBool::new(false);
      let pos = uct.best_move(&field, Player::Red, &mut rng, &should_stop, $iterations);
      assert_eq!(pos, NonZeroPos::new(field.to_pos($image.solution.0, $image.solution.1)));
    }
  }
}

uct_test!(uct_1, IMAGE_1, 100_000, 7);
uct_test!(uct_2, IMAGE_2, 100_000, 7);
uct_test!(uct_3, IMAGE_3, 100_000, 5);
uct_test!(uct_4, IMAGE_4, 100_000, 5);
uct_test!(uct_5, IMAGE_5, 100_000, 7);
uct_test!(
  #[ignore]
  uct_6,
  IMAGE_6,
  1_000_000,
  7
);
uct_test!(
  #[ignore]
  uct_7,
  IMAGE_7,
  1_000_000,
  7
);
uct_test!(uct_8, IMAGE_8, 1_00_000, 7);
uct_test!(uct_9, IMAGE_9, 100_000, 7);
uct_test!(
  #[ignore]
  uct_10,
  IMAGE_10,
  1_000_000,
  7
);
// uct suggests (7, 6) after 1_000_000_000 iterations
// uct_test!(uct_11, IMAGE_11, 1_000_000_000, 7);
uct_test!(uct_12, IMAGE_12, 100_000, 7);
uct_test!(uct_13, IMAGE_13, 100_000, 7);
uct_test!(uct_14, IMAGE_14, 100_000, 39);
