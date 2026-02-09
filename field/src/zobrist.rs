use crate::{field::Pos, points_vec::PointsVec};
use rand::{
  Rng,
  distr::{Distribution, StandardUniform},
};
use std::{iter, ops::BitXor};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Zobrist<N: BitXor + Copy> {
  pub hashes: PointsVec<N>,
}

impl<N> Zobrist<N>
where
  N: BitXor<Output = N> + Copy,
  StandardUniform: Distribution<N>,
{
  pub fn new<R: Rng>(length: Pos, rng: &mut R) -> Zobrist<N> {
    Zobrist {
      hashes: PointsVec(iter::repeat_with(|| rng.random()).take(length).collect()),
    }
  }
}
