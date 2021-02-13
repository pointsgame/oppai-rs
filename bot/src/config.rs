use oppai_minimax::minimax::MinimaxConfig;
use oppai_uct::uct::UctConfig;
use std::time::Duration;
use strum::{EnumString, EnumVariantNames};

#[derive(Clone, Copy, PartialEq, Debug, EnumString, EnumVariantNames)]
pub enum Solver {
  Uct,
  Minimax,
  Heuristic,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub uct: UctConfig,
  pub minimax: MinimaxConfig,
  pub time_gap: Duration,
  pub solver: Solver,
  pub ladders: bool,
  pub ladders_score_limit: u32,
  pub ladders_depth_limit: u32,
  pub ladders_time_limit: Duration,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      uct: UctConfig::default(),
      minimax: MinimaxConfig::default(),
      time_gap: Duration::from_millis(100),
      solver: Solver::Uct,
      ladders: true,
      ladders_score_limit: 0,
      ladders_depth_limit: 0,
      ladders_time_limit: Duration::from_secs(1),
    }
  }
}
