use crate::{
  heuristic::Heuristic, initial::Initial, ladders::Ladders, minimax::Minimax, patterns::Patterns,
  time_limited_ai::TimeLimitedAI, uct::Uct, zero::Zero,
};
use either::Either;
use num_traits::Float;
use oppai_ai::{
  ai::AI,
  analysis::{Analysis, FlatAnalysis, SimpleAnalysis, SingleAnalysis},
};
use oppai_field::{
  field::{length, Field},
  player::Player,
};
use oppai_minimax::minimax::{Minimax as InnerMinimax, MinimaxConfig};
use oppai_patterns::patterns::Patterns as InnerPatterns;
use oppai_uct::uct::{UctConfig, UctRoot};
use oppai_zero::{model::Model, zero::Zero as InnerZero};
use rand::{distr::StandardUniform, prelude::Distribution, Rng, SeedableRng};
use std::{
  convert::identity,
  fmt::{Debug, Display},
  iter::Sum,
  sync::Arc,
  time::Duration,
};
use strum::{EnumString, VariantNames};

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
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

pub struct Oppai<N: Float + Sum + Display + Debug, M: Model<N>> {
  config: Config,
  initial: Initial,
  patterns: Patterns,
  ladders: Ladders,
  heuristic: Heuristic,
  minimax: Minimax,
  uct: Uct,
  zero: Zero<N, M>,
}

type InnerAnalysis<N> = Either<
  SingleAnalysis<(), ()>,
  Either<
    FlatAnalysis<(), ()>,
    Either<
      SingleAnalysis<i32, ()>,
      Either<
        Either<SimpleAnalysis<i32, (), ()>, Either<SingleAnalysis<i32, u32>, SimpleAnalysis<i32, (), ()>>>,
        Either<SimpleAnalysis<f64, f64, usize>, SimpleAnalysis<u64, N, usize>>,
      >,
    >,
  >,
>;

#[derive(Clone, PartialEq, PartialOrd)]
pub struct OppaiWeight<N: Float + Sum + Display + Debug + 'static>(<InnerAnalysis<N> as Analysis>::Weight);

impl<N: Float + Sum + Display + Debug + 'static> OppaiWeight<N> {
  pub fn to_f64(&self) -> Option<f64> {
    match self.0 {
      Either::Left(()) => None,
      Either::Right(Either::Left(())) => None,
      Either::Right(Either::Right(Either::Left(()))) => None,
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Left(w))))) => Some(w as f64),
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Left(())))))) => None,
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Right(w)))))) => Some(w as f64),
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Left(w))))) => Some(w),
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Right(w))))) => Some(w as f64),
    }
  }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct OppaiEstimation<N: Float + Sum + Display + Debug + 'static>(<InnerAnalysis<N> as Analysis>::Estimation);

impl<N: Float + Sum + Display + Debug + 'static> OppaiEstimation<N> {
  pub fn to_f64(&self) -> Option<f64> {
    match self.0 {
      Either::Left(()) => None,
      Either::Right(Either::Left(())) => None,
      Either::Right(Either::Right(Either::Left(e))) => Some(e as f64),
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Left(()))))) => None,
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Left(e)))))) => Some(e as f64),
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Right(())))))) => None,
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Left(e))))) => Some(e),
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Right(e))))) => e.to_f64(),
    }
  }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct OppaiConfidence<N: Float + Sum + Display + Debug + 'static>(<InnerAnalysis<N> as Analysis>::Confidence);

impl<N: Float + Sum + Display + Debug + 'static> OppaiConfidence<N> {
  pub fn to_f64(&self) -> Option<f64> {
    match self.0 {
      Either::Left(()) => None,
      Either::Right(Either::Left(())) => None,
      Either::Right(Either::Right(Either::Left(()))) => None,
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Left(()))))) => None,
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Left(c)))))) => Some(c as f64),
      Either::Right(Either::Right(Either::Right(Either::Left(Either::Right(Either::Right(())))))) => None,
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Left(c))))) => Some(c as f64),
      Either::Right(Either::Right(Either::Right(Either::Right(Either::Right(c))))) => Some(c as f64),
    }
  }
}

pub struct OppaiAnalysis<N: Float + Sum + Display + Debug + 'static>(InnerAnalysis<N>);

impl<N: Float + Sum + Display + Debug + 'static> Analysis for OppaiAnalysis<N> {
  type Weight = OppaiWeight<N>;
  type Estimation = OppaiEstimation<N>;
  type Confidence = OppaiConfidence<N>;

  fn moves(&self) -> impl Iterator<Item = (oppai_field::field::Pos, Self::Weight)> {
    self.0.moves().map(|(pos, weight)| (pos, OppaiWeight(weight)))
  }

  fn estimation(&self) -> Self::Estimation {
    OppaiEstimation(self.0.estimation())
  }

  fn confidence(&self) -> Self::Confidence {
    OppaiConfidence(self.0.confidence())
  }

  fn origin(&self) -> std::any::TypeId {
    self.0.origin()
  }
}

impl<N: Float + Sum + Display + Debug + 'static, M: Model<N> + 'static> AI for Oppai<N, M> {
  type Analysis = OppaiAnalysis<N>;
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
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    let ai = match self.config.solver {
      Solver::Heuristic => Either::Left(Either::Left(&mut self.heuristic)),
      Solver::Minimax => Either::Left(Either::Right((&mut self.minimax, &mut self.heuristic))),
      Solver::Uct => Either::Right(Either::Left(&mut self.uct)),
      Solver::Zero => Either::Right(Either::Right(&mut self.zero)),
    };
    let ai = if self.config.ladders {
      Either::Left((TimeLimitedAI(self.config.ladders_time_limit, self.ladders), ai))
    } else {
      Either::Right(ai)
    }
    .map(|a| a.either(identity, Either::Right), |c| (((), c), c));
    let ai = (&mut self.patterns, ai);
    let mut ai = (self.initial, ai);

    let confidence = confidence.map(|confidence| {
      (
        (),
        (
          (),
          (
            ((), (confidence.minimax_depth, ())),
            (confidence.uct_iterations, confidence.zero_iterations),
          ),
        ),
      )
    });

    OppaiAnalysis(ai.analyze(rng, field, player, confidence, should_stop))
  }
}

impl<N: Float + Sum + Display + Debug + 'static, M: Model<N> + 'static> Oppai<N, M> {
  pub fn new(width: u32, height: u32, config: Config, patterns: Arc<InnerPatterns>, model: M) -> Self {
    let minimax_config = config.minimax.clone();
    let uct_config = config.uct.clone();
    Oppai {
      config,
      initial: Initial,
      patterns: Patterns(patterns),
      ladders: Ladders,
      heuristic: Heuristic,
      minimax: Minimax(InnerMinimax::new(minimax_config)),
      uct: Uct(UctRoot::new(uct_config, length(width, height))),
      zero: Zero(InnerZero::new(model)),
    }
  }

  // pub fn weight_descr(weight: <<Self as AI>::Analysis as Analysis>::Weight) -> (String, f32) {
  //   todo!()
  // }

  // pub fn estimation_descr(weight: <<Self as AI>::Analysis as Analysis>::Estimation) -> String {
  //   todo!()
  // }

  // pub fn confidence_descr(weight: <<Self as AI>::Analysis as Analysis>::Confidence) -> String {
  //   todo!()
  // }

  // pub fn origin_descr(origin: TypeId) -> String {
  //   todo!()
  // }
}
