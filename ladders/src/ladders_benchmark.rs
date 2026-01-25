#[macro_use]
extern crate criterion;

use criterion::Criterion;
use oppai_field::construct_field::{construct_field, construct_moves};
use oppai_field::field::{Field, to_xy};
use oppai_field::player::Player;
use oppai_ladders::ladders::ladders;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;

const SEED: u64 = 7;

fn ladders_long(c: &mut Criterion) {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let (width, _, moves) = construct_moves(
    "
    ..aa..
    .aAAa.
    ..aA..
    ......
    ",
  );
  let mut field = Field::new_from_rng(256, 256, &mut rng);
  for (player, pos) in moves {
    let (x, y) = to_xy(width + 1, pos);
    let pos = field.to_pos(x, y);
    field.put_point(pos, player);
  }

  c.bench_function("ladders_long", |bencher| {
    bencher.iter(|| ladders(black_box(&mut field), Player::Red, &|| false))
  });
}

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

criterion_group!(bench, ladders_rotate, ladders_long);
criterion_main!(bench);
