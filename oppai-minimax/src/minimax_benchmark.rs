#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_minimax::minimax::{Minimax, MinimaxConfig, MinimaxMovesSorting, MinimaxType};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

const MINIMAX_CONFIG_NEGASCOUT: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::NegaScout,
  minimax_moves_sorting: MinimaxMovesSorting::TrajectoriesCount,
  hash_table_size: 1_000,
  rebuild_trajectories: false,
};

const MINIMAX_CONFIG_MTDF: MinimaxConfig = MinimaxConfig {
  threads_count: 1,
  minimax_type: MinimaxType::MTDF,
  minimax_moves_sorting: MinimaxMovesSorting::TrajectoriesCount,
  hash_table_size: 1_000,
  rebuild_trajectories: false,
};

fn negascout_find_best_move(bencher: &mut Bencher) {
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
  bencher.iter(|| {
    let minimax = Minimax::new(MINIMAX_CONFIG_NEGASCOUT);
    let mut local_field = field.clone();
    minimax.minimax(&mut local_field, Player::Red, &mut rng.clone(), 8)
  });
}

fn mtdf_find_best_move(bencher: &mut Bencher) {
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
  bencher.iter(|| {
    let minimax = Minimax::new(MINIMAX_CONFIG_MTDF);
    let mut local_field = field.clone();
    minimax.minimax(&mut local_field, Player::Red, &mut rng.clone(), 8)
  });
}

fn negascout() {
  let mut c = Criterion::default().sample_size(10).configure_from_args();
  c.bench_function("negascout", negascout_find_best_move);
}

fn mtdf() {
  let mut c = Criterion::default().sample_size(10).configure_from_args();
  c.bench_function("mtdf", mtdf_find_best_move);
}

criterion_main!(negascout, mtdf);
