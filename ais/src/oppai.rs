use crate::{ladders::Ladders, minimax::Minimax, patterns::Patterns, uct::Uct, zero::Zero};
use burn::backend::WgpuBackend;
use either::Either;
use oppai_ai::{
  ai::AI,
  analysis::{FlatAnalysis, SimpleAnalysis, SingleAnalysis},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::{field::Field, player::Player};
use oppai_minimax::minimax::MinimaxConfig;
use oppai_uct::uct::UctConfig;
use oppai_zero_burn::model::Model;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use std::time::Duration;
use strum::{EnumString, EnumVariantNames};

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, EnumVariantNames)]
pub enum Solver {
  Uct,
  Minimax,
  Zero,
  // Heuristic,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub uct: UctConfig,
  pub minimax: MinimaxConfig,
  pub solver: Solver,
  pub ladders: bool,
  pub ladders_score_limit: u32,
  pub ladders_depth_limit: u32,
  pub ladders_time_limit: Duration,
}

#[derive(Clone, Debug)]
pub struct InConfidence {
  pub minimax_depth: u32,
  pub uct_iterations: usize,
  pub zero_iterations: usize,
}

pub struct Oppai {
  config: Config,
  patterns: Patterns,
  ladders: Ladders,
  minimax: Minimax,
  uct: Uct,
  zero: Zero<f32, Model<WgpuBackend>>,
}

impl AI for Oppai {
  type Analysis = Either<
    FlatAnalysis<(), ()>,
    Either<
      Either<
        SingleAnalysis<i32, ()>,
        Either<SingleAnalysis<i32, u32>, Either<SimpleAnalysis<f64, f64, usize>, SimpleAnalysis<u64, f32, usize>>>,
      >,
      Either<SingleAnalysis<i32, u32>, Either<SimpleAnalysis<f64, f64, usize>, SimpleAnalysis<u64, f32, usize>>>,
    >,
  >;
  type Confidence = InConfidence;

  fn analyze<S, R, SS>(
    &mut self,
    rng: &mut R,
    field: &mut Field,
    player: Player,
    confidence: Option<Self::Confidence>,
    should_stop: &SS,
  ) -> Self::Analysis
  where
    R: Rng + SeedableRng<Seed = S> + Send,
    Standard: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    let ai = match self.config.solver {
      Solver::Minimax => Either::Left(&mut self.minimax),
      Solver::Uct => Either::Right(Either::Left(&mut self.uct)),
      Solver::Zero => Either::Right(Either::Right(&mut self.zero)),
    };
    let ai = if self.config.ladders {
      Either::Left((TimeLimitedAI(self.config.ladders_time_limit, &mut self.ladders), ai))
    } else {
      Either::Right(ai)
    };
    let mut ai = (&mut self.patterns, ai);

    let confidence = confidence.map(|confidence| {
      (
        (),
        (
          (
            (),
            (
              confidence.minimax_depth,
              (confidence.uct_iterations, confidence.zero_iterations),
            ),
          ),
          (
            confidence.minimax_depth,
            (confidence.uct_iterations, confidence.zero_iterations),
          ),
        ),
      )
    });

    ai.analyze(rng, field, player, confidence, should_stop)
  }
}
