#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_zero::mcgs::Search;
use oppai_zero::random_model::RandomModel;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const SEED: u64 = 7;

const SIMS: u32 = 1_000;

fn search(bencher: &mut Bencher) {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
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
    let mut search = Search::<f64>::new(false);
    let mut model = RandomModel(rng.clone());
    for _ in 0..SIMS {
      search
        .mcgs(&mut field, Player::Red, &mut model, 0, &mut rng.clone())
        .unwrap();
    }
    search.best_move()
  });
}

fn mcgs() {
  let mut c = Criterion::default().sample_size(10).configure_from_args();
  c.bench_function("mcgs", search);
}

criterion_main!(mcgs);
