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
}

impl Default for Config {
  fn default() -> Self {
    Self {
      uct: UctConfig::default(),
      minimax: MinimaxConfig::default(),
      time_gap: Duration::from_millis(100),
      solver: Solver::Uct,
    }
  }
}
