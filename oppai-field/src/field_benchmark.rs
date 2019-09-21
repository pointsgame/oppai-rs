#[macro_use]
extern crate criterion;
extern crate oppai;

use criterion::black_box;
use criterion::Criterion;

use oppai::fibonacci;

use field::{self, Field, Pos};
use player::Player;
use rand::{Rng, SeedableRng, XorShiftRng};
use std::sync::Arc;
use test::Bencher;
use zobrist::Zobrist;

fn random_game(bencher: &mut Bencher, width: u32, height: u32, seed_array: [u32; 4]) {
  let mut rng = XorShiftRng::from_seed(seed_array);
  let mut moves = (field::min_pos(width)..field::max_pos(width, height) + 1).collect::<Vec<Pos>>();
  rng.shuffle(&mut moves);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
  bencher.iter(|| {
    let mut field = Field::new(width, height, zobrist.clone());
    let mut player = Player::Red;
    for &pos in &moves {
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

#[bench]
fn random_game_1(bencher: &mut Bencher) {
  random_game(bencher, 100, 100, [3, 1, 7, 5]);
}

#[bench]
fn random_game_2(bencher: &mut Bencher) {
  random_game(bencher, 100, 100, [1, 3, 5, 7]);
}

#[bench]
fn random_game_3(bencher: &mut Bencher) {
  random_game(bencher, 100, 100, [7, 1, 3, 5]);
}
