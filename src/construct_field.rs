use crate::field::{self, Field};
use crate::player::Player;
use crate::zobrist::Zobrist;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;

pub const DEFAULT_SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

pub fn construct_field(image: &str) -> Field {
  let lines = image
    .split('\n')
    .map(|line| line.trim_matches(' '))
    .filter(|line| !line.is_empty())
    .collect::<Vec<&str>>();
  let height = lines.len() as u32;
  assert!(height > 0);
  let width = lines.first().unwrap().len() as u32;
  assert!(lines.iter().all(|line| line.len() as u32 == width));
  let mut moves = lines
    .into_iter()
    .enumerate()
    .flat_map(|(y, line)| {
      line
        .chars()
        .enumerate()
        .filter(|&(_, c)| c.to_ascii_lowercase() != c.to_ascii_uppercase())
        .map(move |(x, c)| (c, x as u32, y as u32))
    })
    .collect::<Vec<(char, u32, u32)>>();
  moves.sort_by(|&(c1, ..), &(c2, ..)| {
    (c1.to_ascii_lowercase(), c1.is_lowercase()).cmp(&(c2.to_ascii_lowercase(), c2.is_lowercase()))
  });
  let mut rng = XorShiftRng::from_seed(DEFAULT_SEED);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
  let mut field = Field::new(width, height, zobrist);
  for (c, x, y) in moves {
    let player = Player::from_bool(c.is_uppercase());
    let pos = field.to_pos(x, y);
    field.put_point(pos, player);
  }
  field
}
