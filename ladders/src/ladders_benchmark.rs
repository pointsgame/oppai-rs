#[macro_use]
extern crate criterion;

use criterion::Criterion;
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_ladders::ladders::ladders;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;

const SEED: u64 = 7;

fn ladders_rotate(c: &mut Criterion) {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ............
    .........a..
    .......a....
    ............
    .aAAa.......
    ..aa......a.
    ............
    ",
  );

  c.bench_function("ladders_rotate", |bencher| {
    bencher.iter(|| ladders(black_box(&mut field), Player::Red, &|| false))
  });
}

criterion_group!(bench, ladders_rotate);
criterion_main!(bench);
