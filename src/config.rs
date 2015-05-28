use types::*;

pub enum UcbType {
  Ucb1,
  Ucb1Tuned
}

pub static UCT_RADIUS: CoordSum = 3;

pub static UCB_TYPE: UcbType = UcbType::Ucb1Tuned;

pub static UCT_DRAW_WEIGHT: f32 = 0.4;

pub static UCTK: f32 = 1.0;

pub static UCT_WHEN_CREATE_CHILDREN: usize = 2;

pub static UCT_DEPTH: Depth = 8;
