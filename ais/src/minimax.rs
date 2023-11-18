use oppai_ai::{ai::AI, analysis::SingleAnalysis};
use oppai_field::{field::Field, player::Player};
use oppai_minimax::minimax::Minimax as InnerMinimax;
use std::any::TypeId;

pub struct Minimax(InnerMinimax);

impl AI for Minimax {
  type Analysis = SingleAnalysis<i32, u32>;
  type Confidence = u32;

  fn analyze<S, R, SS>(
    &mut self,
    _: &mut R,
    field: &mut Field,
    player: Player,
    confidence: Option<Self::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis
  where
    SS: Fn() -> bool + Sync,
  {
    let (pos, estimation, confidence) = match confidence {
      Some(confidence) => {
        let (pos, estimation) = self.0.minimax(field, player, confidence, should_stop);
        (pos, estimation, confidence)
      }
      None => self.0.minimax_with_time(field, player, should_stop),
    };
    SingleAnalysis {
      best_move: pos,
      estimation,
      confidence,
      origin: TypeId::of::<Self>(),
    }
  }
}
