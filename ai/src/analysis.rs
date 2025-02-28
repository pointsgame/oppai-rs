use either::Either;
use oppai_field::field::{NonZeroPos, Pos};
use rand::Rng;
use std::{any::TypeId, cmp::Ordering, iter};

pub trait Analysis {
  /// Weight for the move. It could be the value of the minimax estimation
  /// function or the probability of winning.
  type Weight: PartialOrd + Clone + 'static;
  /// Quantifies how advantageous the current game position is for a player.
  type Estimation: PartialOrd + Clone + 'static;
  /// Quantifies the confidence level of the AI in the produced analysis.
  /// This metric could be the analysis depth for minimax-based algorithms
  /// or the number of iterations performed for MCTS-based algorithms.
  type Confidence: PartialOrd + Clone + 'static;

  /// Collection of moves with their priorities.
  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)>;
  /// Estimation of the current game state.
  fn estimation(&self) -> Self::Estimation;
  /// Confidence level of the current analysis.
  fn confidence(&self) -> Self::Confidence;
  /// The origin of this analysis.
  fn origin(&self) -> TypeId;
  /// The optimal move.
  fn best_move<R: Rng>(&self, rng: &mut R) -> Option<NonZeroPos> {
    self
      .moves()
      .reduce(
        |(pos1, value1), (pos2, value2)| match value1.partial_cmp(&value2).unwrap_or(Ordering::Equal) {
          Ordering::Greater => (pos1, value1),
          Ordering::Less => (pos2, value2),
          Ordering::Equal => {
            if rng.random() {
              (pos1, value1)
            } else {
              (pos2, value2)
            }
          }
        },
      )
      .and_then(|(pos, _)| NonZeroPos::new(pos))
  }
  /// Whether this analysis doesn't have suggested moves.
  fn is_empty(&self) -> bool {
    self.moves().next().is_none()
  }
  // Map the analysis result.
  fn map<
    W: PartialOrd + Clone + 'static,
    E: PartialOrd + Clone + 'static,
    C: PartialOrd + Clone + 'static,
    WF: Fn(Self::Weight) -> W,
    EF: Fn(Self::Estimation) -> E,
    CF: Fn(Self::Confidence) -> C,
  >(
    self,
    wf: WF,
    ef: EF,
    cf: CF,
  ) -> impl Analysis<Weight = W, Estimation = E, Confidence = C>
  where
    Self: Sized,
  {
    MapAnalysis { a: self, wf, ef, cf }
  }
  // Convert to a simple struct.
  fn to_simple(&self) -> SimpleAnalysis<Self::Weight, Self::Estimation, Self::Confidence> {
    SimpleAnalysis {
      moves: self.moves().collect(),
      estimation: self.estimation(),
      confidence: self.confidence(),
      origin: self.origin(),
    }
  }
}

impl Analysis for () {
  type Weight = ();
  type Estimation = ();
  type Confidence = ();

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    iter::empty()
  }

  fn estimation(&self) -> Self::Estimation {}

  fn confidence(&self) -> Self::Confidence {}

  fn origin(&self) -> TypeId {
    TypeId::of::<Self>()
  }

  fn best_move<R: Rng>(&self, _: &mut R) -> Option<NonZeroPos> {
    None
  }

  fn is_empty(&self) -> bool {
    true
  }
}

impl<A: Analysis, B: Analysis> Analysis for Either<A, B> {
  type Weight = Either<A::Weight, B::Weight>;
  type Estimation = Either<A::Estimation, B::Estimation>;
  type Confidence = Either<A::Confidence, B::Confidence>;

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    Box::new(self.as_ref().map_either(
      |a| a.moves().map(|(pos, weight)| (pos, Either::Left(weight))),
      |a| a.moves().map(|(pos, weight)| (pos, Either::Right(weight))),
    ))
  }

  fn estimation(&self) -> Self::Estimation {
    self.as_ref().map_either(Analysis::estimation, Analysis::estimation)
  }

  fn confidence(&self) -> Self::Confidence {
    self.as_ref().map_either(Analysis::confidence, Analysis::confidence)
  }

  fn origin(&self) -> TypeId {
    self.as_ref().either(Analysis::origin, Analysis::origin)
  }

  fn best_move<R: Rng>(&self, rng: &mut R) -> Option<NonZeroPos> {
    match self {
      Either::Left(analysis) => analysis.best_move(rng),
      Either::Right(analysis) => analysis.best_move(rng),
    }
  }

  fn is_empty(&self) -> bool {
    self.as_ref().either(Analysis::is_empty, Analysis::is_empty)
  }
}

pub struct SimpleAnalysis<W, E, C> {
  /// Collection of moves with their priorities.
  pub moves: Vec<(Pos, W)>,
  /// Estimation of the current game state.
  pub estimation: E,
  /// Confidence level of the current analysis.
  pub confidence: C,
  /// The origin of this analysis.
  pub origin: TypeId,
}

impl<W, E, C> Analysis for SimpleAnalysis<W, E, C>
where
  W: PartialOrd + Clone + 'static,
  E: PartialOrd + Clone + 'static,
  C: PartialOrd + Clone + 'static,
{
  type Weight = W;
  type Estimation = E;
  type Confidence = C;

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    self.moves.iter().cloned()
  }

  fn estimation(&self) -> Self::Estimation {
    self.estimation.clone()
  }

  fn confidence(&self) -> Self::Confidence {
    self.confidence.clone()
  }

  fn origin(&self) -> TypeId {
    self.origin
  }

  fn is_empty(&self) -> bool {
    self.moves.is_empty()
  }
}

pub struct FlatAnalysis<E, C> {
  /// Collection of moves with their priorities.
  pub moves: Vec<Pos>,
  /// Estimation of the current game state.
  pub estimation: E,
  /// Confidence level of the current analysis.
  pub confidence: C,
  /// The origin of this analysis.
  pub origin: TypeId,
}

impl<E, C> Analysis for FlatAnalysis<E, C>
where
  E: PartialOrd + Clone + 'static,
  C: PartialOrd + Clone + 'static,
{
  type Weight = ();
  type Estimation = E;
  type Confidence = C;

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    Box::new(self.moves.iter().map(|&pos| (pos, ())))
  }

  fn estimation(&self) -> Self::Estimation {
    self.estimation.clone()
  }

  fn confidence(&self) -> Self::Confidence {
    self.confidence.clone()
  }

  fn origin(&self) -> TypeId {
    self.origin
  }

  fn best_move<R: Rng>(&self, _: &mut R) -> Option<NonZeroPos> {
    self.moves.first().and_then(|&pos| NonZeroPos::new(pos))
  }

  fn is_empty(&self) -> bool {
    self.moves.is_empty()
  }
}

pub struct SingleAnalysis<E, C> {
  /// Best move.
  pub best_move: Option<NonZeroPos>,
  /// Estimation of the current game state.
  pub estimation: E,
  /// Confidence level of the current analysis.
  pub confidence: C,
  /// The origin of this analysis.
  pub origin: TypeId,
}

impl<E, C> Analysis for SingleAnalysis<E, C>
where
  E: PartialOrd + Clone + 'static,
  C: PartialOrd + Clone + 'static,
{
  type Weight = ();
  type Estimation = E;
  type Confidence = C;

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    self.best_move.map(|pos| (pos.get(), ())).into_iter()
  }

  fn estimation(&self) -> Self::Estimation {
    self.estimation.clone()
  }

  fn confidence(&self) -> Self::Confidence {
    self.confidence.clone()
  }

  fn origin(&self) -> TypeId {
    self.origin
  }

  fn best_move<R: Rng>(&self, _: &mut R) -> Option<NonZeroPos> {
    self.best_move
  }

  fn is_empty(&self) -> bool {
    self.best_move.is_none()
  }
}

pub struct MapAnalysis<
  W1: PartialOrd + Clone + 'static,
  W2: PartialOrd + Clone + 'static,
  E1: PartialOrd + Clone + 'static,
  E2: PartialOrd + Clone + 'static,
  C1: PartialOrd + Clone + 'static,
  C2: PartialOrd + Clone + 'static,
  WF: Fn(W1) -> W2,
  EF: Fn(E1) -> E2,
  CF: Fn(C1) -> C2,
  A: Analysis<Weight = W1, Estimation = E1, Confidence = C1>,
> {
  a: A,
  wf: WF,
  ef: EF,
  cf: CF,
}

impl<
    W1: PartialOrd + Clone + 'static,
    W2: PartialOrd + Clone + 'static,
    E1: PartialOrd + Clone + 'static,
    E2: PartialOrd + Clone + 'static,
    C1: PartialOrd + Clone + 'static,
    C2: PartialOrd + Clone + 'static,
    WF: Fn(W1) -> W2,
    EF: Fn(E1) -> E2,
    CF: Fn(C1) -> C2,
    A: Analysis<Weight = W1, Estimation = E1, Confidence = C1>,
  > Analysis for MapAnalysis<W1, W2, E1, E2, C1, C2, WF, EF, CF, A>
{
  type Weight = W2;
  type Estimation = E2;
  type Confidence = C2;

  fn moves(&self) -> impl Iterator<Item = (Pos, Self::Weight)> {
    self.a.moves().map(|(pos, w)| (pos, (self.wf)(w)))
  }

  fn estimation(&self) -> Self::Estimation {
    (self.ef)(self.a.estimation())
  }

  fn confidence(&self) -> Self::Confidence {
    (self.cf)(self.a.confidence())
  }

  fn origin(&self) -> TypeId {
    self.a.origin()
  }
}
