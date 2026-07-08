use oppai_ai::ai::AI;
use oppai_field::{field::Field, player::Player};
use rand::{Rng, SeedableRng, distr::StandardUniform, prelude::Distribution};
use web_time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

pub struct TimeLimitedAI<I: AI>(pub Duration, pub I);

#[cfg(not(target_arch = "wasm32"))]
impl<I: AI> AI for TimeLimitedAI<I> {
  type Analysis = I::Analysis;
  type Confidence = I::Confidence;

  async fn analyze<S, R, SS>(
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
    let atomic_should_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (s, r) = crossbeam::channel::bounded(1);
    let timer = std::thread::spawn({
      let atomic_should_stop = atomic_should_stop.clone();
      let duration = self.0;
      move || {
        if r.recv_timeout(duration).is_err() {
          atomic_should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
      }
    });
    let result = self
      .1
      .analyze(rng, field, player, confidence, &|| {
        should_stop() || atomic_should_stop.load(std::sync::atomic::Ordering::Relaxed)
      })
      .await;
    // The send fails if the timer already fired and dropped the receiver -
    // that just means there is nothing left to wake up.
    let _ = s.send(());
    timer.join().unwrap();
    result
  }
}

#[cfg(target_arch = "wasm32")]
impl<I: AI> AI for TimeLimitedAI<I> {
  type Analysis = I::Analysis;
  type Confidence = I::Confidence;

  async fn analyze<S, R, SS>(
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
    self
      .1
      .analyze(rng, field, player, confidence, &|| {
        should_stop() || Instant::now() - now >= duration
      })
      .await
  }
}
