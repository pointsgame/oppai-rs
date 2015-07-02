use std::fmt::{Display, Formatter, Result};
use num_cpus;
use types::{CoordSum, Depth, Time};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UcbType {
  Ucb1,
  Ucb1Tuned
}

impl Display for UcbType {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match self {
      &UcbType::Ucb1 => write!(f, "Ucb1"),
      &UcbType::Ucb1Tuned => write!(f, "Ucb1Tuned")
    }
  }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UctKomiType {
  None,
  Static,
  Dynamic
}

impl Display for UctKomiType {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match self {
      &UctKomiType::None => write!(f, "None"),
      &UctKomiType::Static => write!(f, "Static"),
      &UctKomiType::Dynamic => write!(f, "Dynamic")
    }
  }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Solver {
  Uct,
  Heuristic
}

impl Display for Solver {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match self {
      &Solver::Uct => write!(f, "Uct"),
      &Solver::Heuristic => write!(f, "Heuristic")
    }
  }
}

static UCT_RADIUS: CoordSum = 3;

static UCB_TYPE: UcbType = UcbType::Ucb1Tuned;

static UCT_DRAW_WEIGHT: f32 = 0.4;

static UCTK: f32 = 1.0;

static UCT_WHEN_CREATE_CHILDREN: usize = 2;

static UCT_DEPTH: Depth = 8;

static mut THREADS_COUNT: usize = 4;

static UCT_KOMI_TYPE: UctKomiType = UctKomiType::Static;

static UCT_RED: f32 = 0.45;

static UCT_GREEN: f32 = 0.5;

static UCT_KOMI_MIN_ITERATIONS: usize = 1000;

static TIME_GAP: Time = 100;

pub fn init() {
  unsafe {
    THREADS_COUNT = num_cpus::get();
  }
}

#[inline]
pub fn uct_radius() -> CoordSum {
  UCT_RADIUS
}

#[inline]
pub fn ucb_type() -> UcbType {
  UCB_TYPE
}

#[inline]
pub fn uct_draw_weight() -> f32 {
  UCT_DRAW_WEIGHT
}

#[inline]
pub fn uctk() -> f32 {
  UCTK
}

#[inline]
pub fn uct_when_create_children() -> usize {
  UCT_WHEN_CREATE_CHILDREN
}

#[inline]
pub fn uct_depth() -> Depth {
  UCT_DEPTH
}

#[inline]
pub fn threads_count() -> usize {
  unsafe { THREADS_COUNT }
}

#[inline]
pub fn uct_komi_type() -> UctKomiType {
  UCT_KOMI_TYPE
}

#[inline]
pub fn uct_red() -> f32 {
  UCT_RED
}

#[inline]
pub fn uct_green() -> f32 {
  UCT_GREEN
}

#[inline]
pub fn uct_komi_min_iterations() -> usize {
  UCT_KOMI_MIN_ITERATIONS
}

#[inline]
pub fn time_gap() -> Time {
  TIME_GAP
}
