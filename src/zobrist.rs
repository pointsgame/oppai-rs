use crate::field::Pos;
use itertools;
use rand::Rng;

#[derive(Clone, PartialEq, Debug)]
pub struct Zobrist {
  hashes: Vec<u64>,
}

impl Zobrist {
  pub fn new<T: Rng>(length: Pos, rng: &mut T) -> Zobrist {
    Zobrist {
      hashes: itertools::repeat_call(|| rng.gen()).take(length).collect(),
    }
  }

  pub fn get_hash(&self, pos: Pos) -> u64 {
    self.hashes[pos]
  }
}
