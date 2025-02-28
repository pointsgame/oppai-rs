use crate::field::Pos;
use rand::Rng;
use std::iter;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Zobrist {
  hashes: Vec<u64>,
}

impl Zobrist {
  pub fn new<R: Rng>(length: Pos, rng: &mut R) -> Zobrist {
    Zobrist {
      hashes: iter::repeat_with(|| rng.random()).take(length).collect(),
    }
  }

  pub fn get_hash(&self, pos: Pos) -> u64 {
    self.hashes[pos]
  }
}
