use std::any::TypeId;

use oppai_ai::{ai::AI, analysis::SimpleAnalysis};
use oppai_field::{field::Field, player::Player};
use oppai_uct::uct::UctRoot;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

pub struct Uct(UctRoot);

impl AI for Uct {
  type Analysis = SimpleAnalysis<f64, f64, usize>;
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
    R: Rng + SeedableRng<Seed = S> + Send,
    Standard: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    let (moves, confidence, estimation) =
      self
        .0
        .best_moves(field, player, rng, should_stop, confidence.unwrap_or(usize::MAX));
    SimpleAnalysis {
      moves,
      estimation,
      confidence,
      origin: TypeId::of::<Self>(),
    }
  }
}
