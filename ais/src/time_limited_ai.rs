use oppai_ai::ai::AI;
use oppai_field::{field::Field, player::Player};
use rand::{distr::StandardUniform, prelude::Distribution, Rng, SeedableRng};
use web_time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

pub struct TimeLimitedAI<I: AI>(pub Duration, pub I);

#[cfg(not(target_arch = "wasm32"))]
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
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    let atomic_should_stop = std::sync::atomic::AtomicBool::new(false);
    let (s, r) = crossbeam::channel::bounded(1);
    crossbeam::scope(|scope| {
      scope.spawn(|_| {
        if r.recv_timeout(self.0).is_err() {
          atomic_should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
      });
      let result = self.1.analyze(rng, field, player, confidence, &|| {
        should_stop() || atomic_should_stop.load(std::sync::atomic::Ordering::Relaxed)
      });
      s.send(()).unwrap();
      result
    })
    .unwrap()
  }
}

#[cfg(target_arch = "wasm32")]
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
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    let duration = self.0;
    let now = Instant::now();
    self.1.analyze(rng, field, player, confidence, &|| {
      should_stop() || Instant::now() - now >= duration
    })
  }
}
