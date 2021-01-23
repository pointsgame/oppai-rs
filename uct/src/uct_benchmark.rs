#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion};
use oppai_field::construct_field::construct_field;
use oppai_field::field;
use oppai_field::player::Player;
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType, UctRoot};
use rand::rngs::SmallRng;
use rand::SeedableRng;

const SEED: u64 = 99991;

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

fn find_best_move(bencher: &mut Bencher) {
  let mut rng = SmallRng::seed_from_u64(SEED);
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
  let length = field::length(field.width(), field.height());
  bencher.iter(|| {
    let mut uct = UctRoot::new(UCT_CONFIG, length);
    uct.best_move_with_iterations_count(&field, Player::Red, &mut rng.clone(), 100_000)
  });
}

fn uct() {
  let mut c = Criterion::default().sample_size(10).configure_from_args();
  c.bench_function("uct", find_best_move);
}

criterion_main!(uct);
