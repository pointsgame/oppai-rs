#[macro_use]
extern crate criterion;

use criterion::{black_box, Bencher, Criterion};
use oppai_field::construct_field::construct_moves;
use oppai_field::field::{self, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::Arc;

const SEED_1: u64 = 3;
const SEED_2: u64 = 5;
const SEED_3: u64 = 7;

fn random_game(bencher: &mut Bencher, width: u32, height: u32, seed: u64) {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
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

fn random_game_1(c: &mut Criterion) {
  c.bench_function("random_game_1", |bencher| random_game(bencher, 30, 30, SEED_1));
}

fn random_game_2(c: &mut Criterion) {
  c.bench_function("random_game_2", |bencher| random_game(bencher, 30, 30, SEED_2));
}

fn random_game_3(c: &mut Criterion) {
  c.bench_function("random_game_3", |bencher| random_game(bencher, 30, 30, SEED_3));
}

fn game(bencher: &mut Bencher, width: u32, height: u32, moves: Vec<(Player, Pos)>) {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED_1);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
  bencher.iter(|| {
    let mut field = Field::new(width, height, zobrist.clone());
    for &(player, pos) in black_box(&moves) {
      if field.is_putting_allowed(pos) {
        field.put_point(pos, player);
      }
    }
    for _ in 0..field.moves_count() {
      field.undo();
    }
    field
  });
}

fn game_without_surroundings(c: &mut Criterion) {
  let (width, height, moves) = construct_moves(
    "
    ..............
    .aDFgjMOprUWx.
    .BceHKlnQStvY.
    ..............
    ",
  );
  c.bench_function("game_without_surroundings", |bencher| {
    game(bencher, width, height, moves.clone())
  });
}

fn game_with_surroundings(c: &mut Criterion) {
  let (width, height, moves) = construct_moves(
    "
    .........
    ....R....
    ...QhS...
    ..PgCjT..
    .OfBaDkU.
    ..ZnElV..
    ...YmW...
    ....X....
    .........
    ",
  );
  c.bench_function("game_with_surroundings", |bencher| {
    game(bencher, width, height, moves.clone())
  });
}

criterion_group!(random_games, random_game_1, random_game_2, random_game_3);
criterion_group!(games, game_without_surroundings, game_with_surroundings);
criterion_main!(random_games, games);
