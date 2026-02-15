use crate::field::{Field, Pos, to_pos};
use crate::player::Player;
use crate::zobrist::Zobrist;
use rand::Rng;
use std::sync::Arc;

pub fn construct_moves(image: &str) -> (u32, u32, Vec<(Player, Pos)>) {
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
        .map(move |(x, c)| (c, to_pos(width + 1, x as u32, y as u32)))
    })
    .collect::<Vec<_>>();
  moves.sort_by(|&(c1, ..), &(c2, ..)| {
    (c1.to_ascii_lowercase(), c1.is_lowercase()).cmp(&(c2.to_ascii_lowercase(), c2.is_lowercase()))
  });
  (
    width,
    height,
    moves
      .into_iter()
      .map(|(c, pos)| (Player::from_bool(c.is_uppercase()), pos))
      .collect(),
  )
}

pub fn construct_field_with_zobrist(zobrist: Arc<Zobrist<u64>>, image: &str) -> Field {
  let (width, height, moves) = construct_moves(image);
  let mut field = Field::new(width, height, zobrist);
  for (player, pos) in moves {
    assert!(field.put_point(pos, player));
  }
  field
}

pub fn construct_field<T: Rng>(rng: &mut T, image: &str) -> Field {
  let (width, height, moves) = construct_moves(image);
  let mut field = Field::new_from_rng(width, height, rng);
  for (player, pos) in moves {
    assert!(field.put_point(pos, player));
  }
  field
}
