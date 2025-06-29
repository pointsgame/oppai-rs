use oppai_ai::{ai::AI, analysis::SingleAnalysis};
use oppai_field::{
  field::{Field, NonZeroPos},
  player::Player,
};
use rand::{
  Rng, SeedableRng,
  distr::{Distribution, StandardUniform},
};
use std::{any::TypeId, cmp};

pub fn initial_move(field: &Field) -> Option<NonZeroPos> {
  let result = match field.moves_count() {
    0 => Some((field.width() / 2, field.height() / 2)),
    1 => {
      let width = field.width();
      let height = field.height();
      let pos = field.moves[0];
      let x = field.to_x(pos);
      let y = field.to_y(pos);
      if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
        Some((width / 2, height / 2))
      } else if cmp::min(x, width - x - 1) < cmp::min(y, height - y - 1) {
        if x < width - x - 1 {
          Some((x + 1, y))
        } else {
          Some((x - 1, y))
        }
      } else if cmp::min(x, width - x - 1) > cmp::min(y, height - y - 1) {
        if y < height - y - 1 {
          Some((x, y + 1))
        } else {
          Some((x, y - 1))
        }
      } else {
        let dx = x as i32 - (width / 2) as i32;
        let dy = y as i32 - (height / 2) as i32;
        if dx.abs() > dy.abs() {
          if dx < 0 { Some((x + 1, y)) } else { Some((x - 1, y)) }
        } else if dy < 0 {
          Some((x, y + 1))
        } else {
          Some((x, y - 1))
        }
      }
    }
    _ => None,
  };
  result.and_then(|(x, y)| NonZeroPos::new(field.to_pos(x, y)))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Initial;

impl AI for Initial {
  type Analysis = SingleAnalysis<(), ()>;
  type Confidence = ();

  fn analyze<S, R, SS>(
    &mut self,
    _rng: &mut R,
    field: &mut Field,
    _player: Player,
    _confidence: Option<Self::Confidence>,
    _should_stop: &SS,
  ) -> Self::Analysis
  where
    R: Rng + SeedableRng<Seed = S> + Send,
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    SingleAnalysis {
      best_move: initial_move(field),
      estimation: (),
      confidence: (),
      origin: TypeId::of::<Self>(),
    }
  }
}
