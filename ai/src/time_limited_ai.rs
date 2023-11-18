use crate::ai::AI;
use oppai_field::{field::Field, player::Player};
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use web_time::{Duration, Instant};

pub struct TimeLimitedAI<I: AI>(pub Duration, pub I);

impl<I: AI> AI for TimeLimitedAI<I> {
  type Analysis = I::Analysis;
  type Confidence = I::Confidence;

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
    let duration = self.0;
    let now = Instant::now();
    self.1.analyze(rng, field, player, confidence, &|| {
      should_stop() || Instant::now() - now >= duration
    })
  }
}
