use rand::*;

#[derive(Clone)]
pub struct Zobrist {
  hashes: Vec<u64>
}

impl Zobrist {
  pub fn new<T: Rng>(length: usize, rng: &mut T) -> Zobrist {
    let mut zobrist = Zobrist {
      hashes: Vec::with_capacity(length)
    };
    for _ in 0 .. length {
      zobrist.hashes.push(rng.gen());
    }
    zobrist
  }

  pub fn get_hash(&self, pos: usize) -> u64 {
    self.hashes[pos]
  }
}
