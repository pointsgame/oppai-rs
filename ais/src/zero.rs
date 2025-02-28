use num_traits::Float;
use oppai_ai::{ai::AI, analysis::SimpleAnalysis};
use oppai_field::{field::Field, player::Player};
use oppai_zero::{model::Model, zero::Zero as InnerZero};
use rand::{distr::StandardUniform, prelude::Distribution, Rng, SeedableRng};
use std::{
  any::TypeId,
  fmt::{Debug, Display},
  iter::Sum,
};

pub struct Zero<N: Float + Sum + Display + Debug, M: Model<N>>(pub InnerZero<N, M>);

impl<N: Float + Sum + Display + Debug + PartialOrd + 'static, M: Model<N> + 'static> AI for Zero<N, M> {
  type Analysis = SimpleAnalysis<u64, N, usize>;
  type Confidence = usize;

  fn analyze<S, R, SS>(
    &mut self,
    rng: &mut R,
    field: &mut Field,
    player: Player,
    confidence: Option<Self::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis
  where
    R: Rng + SeedableRng<Seed = S>,
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    if let Ok((moves, confidence, estimation)) =
      self
        .0
        .best_moves(field, player, rng, should_stop, confidence.unwrap_or(usize::MAX))
    {
      SimpleAnalysis {
        moves,
        estimation,
        confidence,
        origin: TypeId::of::<Self>(),
      }
    } else {
      SimpleAnalysis {
        moves: Vec::new(),
        estimation: N::zero(),
        confidence: 0,
        origin: TypeId::of::<Self>(),
      }
    }
  }
}
