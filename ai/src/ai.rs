use crate::analysis::Analysis;
use either::Either;
use oppai_field::{field::Field, player::Player};
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};

pub trait AI {
  /// Analysis result of this AI.
  type Analysis: Analysis;
  // Desired confidence of the AI analysis.
  type Confidence: Clone + 'static;

  /// Analyze the game position.
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
    SS: Fn() -> bool + Sync;
}

impl AI for () {
  type Analysis = ();
  type Confidence = ();

  fn analyze<S, R, SS>(
    &mut self,
    _: &mut R,
    _: &mut Field,
    _: Player,
    _: Option<Self::Confidence>,
    _: &SS,
  ) -> Self::Analysis {
  }
}

impl<T: AI> AI for &mut T {
  type Analysis = T::Analysis;
  type Confidence = T::Confidence;

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
    (*self).analyze(rng, field, player, confidence, should_stop)
  }
}

impl<A: AI, B: AI> AI for (A, B) {
  type Analysis = Either<A::Analysis, B::Analysis>;
  type Confidence = (A::Confidence, B::Confidence);

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
    let analysis = self.0.analyze(
      rng,
      field,
      player,
      confidence.as_ref().map(|c| &c.0).cloned(),
      should_stop,
    );
    if analysis.is_empty() {
      Either::Right(self.1.analyze(rng, field, player, confidence.map(|c| c.1), should_stop))
    } else {
      Either::Left(analysis)
    }
  }
}

impl<A: AI, B: AI> AI for Either<A, B> {
  type Analysis = Either<A::Analysis, B::Analysis>;
  type Confidence = (A::Confidence, B::Confidence);

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
    match self {
      Either::Left(ai) => Either::Left(ai.analyze(rng, field, player, confidence.map(|c| c.0), should_stop)),
      Either::Right(ai) => Either::Right(ai.analyze(rng, field, player, confidence.map(|c| c.1), should_stop)),
    }
  }
}
