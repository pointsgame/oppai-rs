use oppai_ai::{ai::AI, analysis::SingleAnalysis};
use oppai_field::{field::Field, player::Player};
use oppai_ladders::ladders::ladders;
use std::any::TypeId;

pub struct Ladders;

impl AI for Ladders {
  type Analysis = SingleAnalysis<i32, ()>;
  type Confidence = ();

  fn analyze<S, R, SS>(
    &mut self,
    _: &mut R,
    field: &mut Field,
    player: Player,
    _: Option<Self::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis
  where
    SS: Fn() -> bool + Sync,
  {
    let (pos, score, _) = ladders(field, player, should_stop);
    SingleAnalysis {
      best_move: pos,
      estimation: score,
      confidence: (),
      origin: TypeId::of::<Self>(),
    }
  }
}
