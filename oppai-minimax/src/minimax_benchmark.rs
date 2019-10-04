#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_minimax::minimax::{Minimax, MinimaxConfig, MinimaxType};
use oppai_test_images::*;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

const MINIMAX_CONFIG_NEGASCOUT: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::NegaScout,
  hash_table_size: 1_000,
  rebuild_trajectories: false,
};

const MINIMAX_CONFIG_MTDF: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::MTDF,
  hash_table_size: 1_000,
  rebuild_trajectories: false,
};

macro_rules! minimax_bench {
  ($name:ident, $config:ident, $image:ident, $depth:expr) => {
    fn $name(bencher: &mut Bencher) {
      let mut rng = XorShiftRng::from_seed(SEED);
      let field = construct_field(&mut rng, $image.image);
      bencher.iter(|| {
        let minimax = Minimax::new($config);
        let mut local_field = field.clone();
        minimax.minimax(&mut local_field, Player::Red, $depth)
      });
    }
  };
}

macro_rules! minimax_benches {
  ($group:ident => { $($name:ident, $config:ident, $image:ident, $depth:expr;)* } ) => {
    $(minimax_bench!($name, $config, $image, $depth);)*

    fn $group() {
      let mut c = Criterion::default().sample_size(10).configure_from_args();
      $(c.bench_function(stringify!($name), $name);)*
    }
  }
}

minimax_benches!(
  negascout => {
    negascout_1, MINIMAX_CONFIG_NEGASCOUT, IMAGE_1, 8;
    negascout_2, MINIMAX_CONFIG_NEGASCOUT, IMAGE_2, 8;
    negascout_3, MINIMAX_CONFIG_NEGASCOUT, IMAGE_3, 8;
    negascout_4, MINIMAX_CONFIG_NEGASCOUT, IMAGE_4, 8;
    negascout_5, MINIMAX_CONFIG_NEGASCOUT, IMAGE_5, 8;
    negascout_6, MINIMAX_CONFIG_NEGASCOUT, IMAGE_6, 8;
    // negascout_7, MINIMAX_CONFIG_NEGASCOUT, IMAGE_7, 10;
    negascout_8, MINIMAX_CONFIG_NEGASCOUT, IMAGE_8, 8;
    // negascout_9, MINIMAX_CONFIG_NEGASCOUT, IMAGE_9, 10;
    negascout_10, MINIMAX_CONFIG_NEGASCOUT, IMAGE_10, 8;
    // negascout_11, MINIMAX_CONFIG_NEGASCOUT, IMAGE_11, 12;
    negascout_12, MINIMAX_CONFIG_NEGASCOUT, IMAGE_12, 8;
    negascout_13, MINIMAX_CONFIG_NEGASCOUT, IMAGE_13, 8;
    negascout_14, MINIMAX_CONFIG_NEGASCOUT, IMAGE_14, 8;
  }
);

minimax_benches!(
  mtdf => {
    mtdf_1, MINIMAX_CONFIG_MTDF, IMAGE_1, 8;
    mtdf_2, MINIMAX_CONFIG_MTDF, IMAGE_2, 8;
    mtdf_3, MINIMAX_CONFIG_MTDF, IMAGE_3, 8;
    mtdf_4, MINIMAX_CONFIG_MTDF, IMAGE_4, 8;
    mtdf_5, MINIMAX_CONFIG_MTDF, IMAGE_5, 8;
    mtdf_6, MINIMAX_CONFIG_MTDF, IMAGE_6, 8;
    // mtdf_7, MINIMAX_CONFIG_MTDF, IMAGE_7, 10;
    mtdf_8, MINIMAX_CONFIG_MTDF, IMAGE_8, 8;
    // mtdf_9, MINIMAX_CONFIG_MTDF, IMAGE_9, 10;
    mtdf_10, MINIMAX_CONFIG_MTDF, IMAGE_10, 8;
    // mtdf_11, MINIMAX_CONFIG_MTDF, IMAGE_11, 12;
    mtdf_12, MINIMAX_CONFIG_MTDF, IMAGE_12, 8;
    mtdf_13, MINIMAX_CONFIG_MTDF, IMAGE_13, 8;
    mtdf_14, MINIMAX_CONFIG_MTDF, IMAGE_14, 8;
  }
);

criterion_main!(negascout, mtdf);
