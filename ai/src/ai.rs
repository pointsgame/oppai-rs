use crate::analysis::Analysis;
use either::Either;
use oppai_field::{field::Field, player::Player};

pub trait AI {
  /// Analysis result of this AI.
  type Analysis: Analysis;
  // Desired confidence of the AI analysis.
  type Confidence: PartialOrd + Clone + 'static;

  /// Analyze the game position.
  fn analyze<SS: Fn() -> bool + Sync>(
    &mut self,
    field: &mut Field,
    player: Player,
    confidence: Option<Self::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis;
}

impl AI for () {
  type Analysis = ();
  type Confidence = ();

  fn analyze<SS: Fn() -> bool + Sync>(&mut self, _: &mut Field, _: Player, _: Option<()>, _: &SS) -> Self::Analysis {}
}

impl<A: AI, B: AI> AI for (A, B) {
  type Analysis = Either<A::Analysis, B::Analysis>;
  type Confidence = (A::Confidence, B::Confidence);

  fn analyze<SS: Fn() -> bool + Sync>(
    &mut self,
    field: &mut Field,
    player: Player,
    confidence: Option<(A::Confidence, B::Confidence)>,
    should_stop: &SS,
  ) -> Self::Analysis {
    let analysis = self
      .0
      .analyze(field, player, confidence.as_ref().map(|c| &c.0).cloned(), should_stop);
    if analysis.is_empty() {
      Either::Right(self.1.analyze(field, player, confidence.map(|c| c.1), should_stop))
    } else {
      Either::Left(analysis)
    }
  }
}

impl<A: AI, B: AI> AI for Either<A, B> {
  type Analysis = Either<A::Analysis, B::Analysis>;
  type Confidence = (A::Confidence, B::Confidence);

  fn analyze<SS: Fn() -> bool + Sync>(
    &mut self,
    field: &mut Field,
    player: Player,
    confidence: Option<(A::Confidence, B::Confidence)>,
    should_stop: &SS,
  ) -> Self::Analysis {
    match self {
      Either::Left(ai) => Either::Left(ai.analyze(field, player, confidence.map(|c| c.0), should_stop)),
      Either::Right(ai) => Either::Right(ai.analyze(field, player, confidence.map(|c| c.1), should_stop)),
    }
  }
}
