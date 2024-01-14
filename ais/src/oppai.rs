use crate::{heuristic::Heuristic, ladders::Ladders, minimax::Minimax, patterns::Patterns, uct::Uct, zero::Zero};
use burn::backend::Wgpu;
use either::Either;
use oppai_ai::{
  ai::AI,
  analysis::{Analysis, FlatAnalysis, SimpleAnalysis, SingleAnalysis},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::{
  field::{Field, Pos},
  player::Player,
};
use oppai_minimax::minimax::{Minimax as InnerMinimax, MinimaxConfig};
use oppai_patterns::patterns::Patterns as InnerPatterns;
use oppai_uct::uct::{UctConfig, UctRoot};
use oppai_zero::zero::Zero as InnerZero;
use oppai_zero_burn::model::Model;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use std::{any::TypeId, convert::identity, sync::Arc, time::Duration};
use strum::{EnumString, EnumVariantNames};

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, EnumVariantNames)]
pub enum Solver {
  Heuristic,
  Minimax,
  Uct,
  Zero,
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

impl Default for Config {
  fn default() -> Self {
    Self {
      uct: Default::default(),
      minimax: Default::default(),
      solver: Solver::Uct,
      ladders: true,
      ladders_score_limit: 0,
      ladders_depth_limit: 0,
      ladders_time_limit: Duration::from_secs(1),
    }
  }
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
  heuristic: Heuristic,
  minimax: Minimax,
  uct: Uct,
  zero: Zero<f32, Model<Wgpu>>,
}

impl AI for Oppai {
  type Analysis = Either<
    FlatAnalysis<(), ()>,
    Either<
      SingleAnalysis<i32, ()>,
      Either<
        Either<SimpleAnalysis<i32, (), ()>, Either<SingleAnalysis<i32, u32>, SimpleAnalysis<i32, (), ()>>>,
        Either<SimpleAnalysis<f64, f64, usize>, SimpleAnalysis<u64, f32, usize>>,
      >,
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
      Solver::Heuristic => Either::Left(Either::Left(&mut self.heuristic)),
      Solver::Minimax => Either::Left(Either::Right((&mut self.minimax, &mut self.heuristic))),
      Solver::Uct => Either::Right(Either::Left(&mut self.uct)),
      Solver::Zero => Either::Right(Either::Right(&mut self.zero)),
    };
    let ai = if self.config.ladders {
      Either::Left((TimeLimitedAI(self.config.ladders_time_limit, &mut self.ladders), ai))
    } else {
      Either::Right(ai)
    }
    .map(|a| a.either(identity, Either::Right), |c| (((), c), c));
    let mut ai = (&mut self.patterns, ai);

    let confidence = confidence.map(|confidence| {
      (
        (),
        (
          ((), (confidence.minimax_depth, ())),
          (confidence.uct_iterations, confidence.zero_iterations),
        ),
      )
    });

    ai.analyze(rng, field, player, confidence, should_stop)
  }
}

impl Oppai {
  pub fn new(config: Config, length: Pos, patterns: Arc<InnerPatterns>, model: Model<Wgpu>) -> Self {
    let minimax_config = config.minimax.clone();
    let uct_config = config.uct.clone();
    Oppai {
      config,
      patterns: Patterns(patterns),
      ladders: Ladders,
      heuristic: Heuristic,
      minimax: Minimax(InnerMinimax::new(minimax_config)),
      uct: Uct(UctRoot::new(uct_config, length)),
      zero: Zero(InnerZero::new(model)),
    }
  }
}