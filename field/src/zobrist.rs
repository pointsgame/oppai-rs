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

  #[inline]
  #[cfg(not(feature = "unsafe"))]
  pub fn get_hash(&self, pos: Pos) -> u64 {
    self.hashes[pos]
  }

  #[inline]
  #[cfg(feature = "unsafe")]
  pub fn get_hash(&self, pos: Pos) -> u64 {
    unsafe { *self.hashes.get_unchecked(pos) }
  }

  pub fn len(&self) -> usize {
    self.hashes.len()
  }

  pub fn is_empty(&self) -> bool {
    self.hashes.is_empty()
  }
}
