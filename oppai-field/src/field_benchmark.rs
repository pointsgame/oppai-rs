#[macro_use]
extern crate criterion;
extern crate oppai_field;

use criterion::black_box;
use criterion::Bencher;
use criterion::Criterion;
use oppai_field::field::{self, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;

const SEED_1: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const SEED_2: [u8; 16] = [23, 29, 31, 37, 41, 43, 47, 53, 2, 3, 5, 7, 11, 13, 17, 19];
const SEED_3: [u8; 16] = [11, 13, 17, 19, 23, 29, 31, 37, 2, 3, 5, 7, 41, 43, 47, 53];

fn random_game(bencher: &mut Bencher, width: u32, height: u32, seed_array: [u8; 16]) {
  let mut rng = XorShiftRng::from_seed(seed_array);
  let mut moves = (field::min_pos(width)..field::max_pos(width, height) + 1).collect::<Vec<Pos>>();
  moves.shuffle(&mut rng);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
  bencher.iter(|| {
    let mut field = Field::new(width, height, zobrist.clone());
    let mut player = Player::Red;
    for &pos in black_box(&moves) {
      if field.is_putting_allowed(pos) {
        field.put_point(pos, player);
        player = player.next();
      }
    }
    for _ in 0..field.moves_count() {
      field.undo();
    }
    field
  });
}

pub fn random_game_1(c: &mut Criterion) {
  c.bench_function("random_game_1", |bencher| {
    random_game(bencher, 30, 30, SEED_1)
  });
}

pub fn random_game_2(c: &mut Criterion) {
  c.bench_function("random_game_2", |bencher| {
    random_game(bencher, 30, 30, SEED_2)
  });
}

pub fn random_game_3(c: &mut Criterion) {
  c.bench_function("random_game_3", |bencher| {
    random_game(bencher, 30, 30, SEED_3)
  });
}

criterion_group!(random_games, random_game_1, random_game_2, random_game_3);
criterion_main!(random_games);
