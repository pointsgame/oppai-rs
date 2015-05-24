use types::*;
use rand::*;

#[derive(Clone)]
pub struct Zobrist {
  hashes: Vec<u64>
}

impl Zobrist {
  pub fn new<T: Rng>(length: Pos, rng: &mut T) -> Zobrist {
    let mut zobrist = Zobrist {
      hashes: Vec::with_capacity(length)
    };
    for _ in 0 .. length {
      zobrist.hashes.push(rng.gen());
    }
    zobrist
  }

  pub fn get_hash(&self, pos: Pos) -> u64 {
    self.hashes[pos]
  }
}
