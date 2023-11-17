use crate::ai::AI;
use oppai_field::{field::Field, player::Player};
use web_time::{Duration, Instant};

pub struct TimeLimitedAI<I: AI>(Duration, I);

impl<I: AI> AI for TimeLimitedAI<I> {
  type Analysis = I::Analysis;
  type Confidence = I::Confidence;

  fn analyze<SS: Fn() -> bool + Sync>(
    &mut self,
    field: &mut Field,
    player: Player,
    confidence: Option<I::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis {
    let duration = self.0;
    let now = Instant::now();
    self.1.analyze(field, player, confidence, &|| {
      should_stop() || Instant::now() - now >= duration
    })
  }
}
