use oppai_ai::ai::AI;
use oppai_ai::analysis::SimpleAnalysis;
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use std::any::TypeId;

static CG_SUM: [i32; 9] = [-5, -1, 0, 0, 1, 2, 5, 20, 30];

fn heuristic_estimation(field: &Field, pos: Pos, player: Player) -> i32 {
  let enemy = player.next();
  let g1 = field.number_near_groups(pos, player) as i32;
  let g2 = field.number_near_groups(pos, enemy) as i32;
  let c1 = CG_SUM[field.number_near_points_diag(pos, player) as usize];
  let c2 = CG_SUM[field.number_near_points_diag(pos, enemy) as usize];
  let mut result = (g1 * 3 + g2 * 2) * (5 - (g1 - g2).abs()) - c1 - c2;
  if let Some(&last_pos) = field.moves().last() {
    if field.is_near(last_pos, pos) {
      result += 5;
    }
  }
  result
}

fn heuristic(field: &Field, player: Player) -> Vec<(Pos, i32)> {
  // TODO: check for stupid move.
  (field.min_pos()..=field.max_pos())
    .filter(|&pos| field.cell(pos).is_putting_allowed())
    .map(|pos| (pos, heuristic_estimation(field, pos, player)))
    .collect()
}

pub struct Heuristic;

impl AI for Heuristic {
  type Analysis = SimpleAnalysis<i32, (), ()>;
  type Confidence = ();

  fn analyze<S, R, SS>(
    &mut self,
    _: &mut R,
    field: &mut Field,
    player: Player,
    _: Option<Self::Confidence>,
    _: &SS,
  ) -> Self::Analysis
  where
    SS: Fn() -> bool + Sync,
  {
    SimpleAnalysis {
      moves: heuristic(field, player),
      estimation: (),
      confidence: (),
      origin: TypeId::of::<Self>(),
    }
  }
}
