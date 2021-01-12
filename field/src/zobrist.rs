use crate::field::Pos;
use rand::Rng;
use std::iter;

#[derive(Clone, PartialEq, Debug)]
pub struct Zobrist {
  hashes: Vec<u64>,
}

impl Zobrist {
  pub fn new<T: Rng>(length: Pos, rng: &mut T) -> Zobrist {
    Zobrist {
      hashes: iter::repeat_with(|| rng.gen()).take(length).collect(),
    }
  }

  pub fn get_hash(&self, pos: Pos) -> u64 {
    self.hashes[pos]
  }
}
