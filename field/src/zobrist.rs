use crate::{field::Pos, points_vec::PointsVec};
use rand::Rng;
use std::iter;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Zobrist {
  pub hashes: PointsVec<u64>,
}

impl Zobrist {
  pub fn new<R: Rng>(length: Pos, rng: &mut R) -> Zobrist {
    Zobrist {
      hashes: PointsVec(iter::repeat_with(|| rng.random()).take(length).collect()),
    }
  }
}
