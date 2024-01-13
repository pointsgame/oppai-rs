use oppai_ai::{ai::AI, analysis::FlatAnalysis};
use oppai_field::{field::Field, player::Player};
use oppai_patterns::patterns::Patterns as InnerPatterns;
use std::{any::TypeId, sync::Arc};

pub struct Patterns(pub Arc<InnerPatterns>);

impl AI for Patterns {
  type Analysis = FlatAnalysis<(), ()>;
  type Confidence = ();

  fn analyze<S, R, SS>(
    &mut self,
    _: &mut R,
    field: &mut Field,
    player: Player,
    _: Option<Self::Confidence>,
    _: &SS,
  ) -> Self::Analysis {
    let moves = self.0.find(field, player, false);
    FlatAnalysis {
      moves,
      estimation: (),
      confidence: (),
      origin: TypeId::of::<Self>(),
    }
  }
}
