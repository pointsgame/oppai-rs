use crate::{
  field_features::{field_features, global},
  mcgs::Search,
  model::Model,
};
use ndarray::Axis;
use num_traits::Float;
use oppai_field::{
  field::{Field, Hash, Pos, to_x, to_y},
  player::Player,
};
use rand::Rng;
use std::{
  fmt::{Debug, Display},
  iter::Sum,
};

type Analysis<N> = (Vec<(Pos, u64)>, usize, N);

type PolicyAnalysis<N> = (Vec<(Pos, N)>, N);

#[derive(Clone)]
pub struct Zero<N: Float, M: Model<N>> {
  model: M,
  search: Search<N>,
  /// Player to move at the search root.
  player: Player,
  /// Number of moves played on the field when the root was set.
  moves_count: usize,
  /// Zobrist hash of the root position, used to detect history divergence.
  hash: Hash,
}

impl<N, M> Zero<N, M>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
{
  pub fn new(model: M) -> Self {
    Zero {
      model,
      search: Search::new(),
      // A fresh search holds an empty root, which corresponds to the empty
      // board: zero moves played, zero hash, Red to move.
      player: Player::Red,
      moves_count: 0,
      hash: 0,
    }
  }

  pub fn clear(&mut self) {
    self.search = Search::new();
    self.player = Player::Red;
    self.moves_count = 0;
    self.hash = 0;
  }

  fn init(&mut self, field: &Field, player: Player) {
    self.search = Search::new();
    self.player = player;
    self.moves_count = field.moves_count();
    self.hash = field.hash();
  }

  /// Advances the persistent graph to the current `field`/`player`, reusing the
  /// subtree already explored when the field is a continuation of the previously
  /// searched position.
  fn update(&mut self, field: &Field, player: Player) {
    // The stored root must still sit on the field's actual history; otherwise the
    // graph is stale and the search restarts from the current position.
    if field.hash_at(self.moves_count) != Some(self.hash) {
      self.init(field, player);
      return;
    }

    // Replay the moves played since the last search, descending into the
    // matching child for each one.
    let moves_count = field.moves_count();
    while self.moves_count < moves_count {
      let pos = field.moves[self.moves_count];
      if !self.search.next_root(pos) {
        // The continuation was never expanded - start fresh from here.
        self.init(field, player);
        return;
      }
      self.moves_count += 1;
      self.player = self.player.next();
    }

    if self.player != player {
      self.init(field, player);
      return;
    }

    self.hash = field.hash();
    self.search.compact();
  }

  pub fn best_moves<SS: Fn() -> bool, R: Rng>(
    &mut self,
    field: &Field,
    player: Player,
    rng: &mut R,
    should_stop: &SS,
    max_iterations_count: usize,
  ) -> Result<Analysis<N>, <M as Model<N>>::E> {
    self.update(field, player);

    // TODO: check if game is over
    let mut iterations = 0;
    let mut field = field.clone();
    while !should_stop() && iterations < max_iterations_count {
      self.search.mcgs(&mut field, player, &mut self.model, 0, rng)?;
      iterations += 1;
    }

    Ok((self.search.visits().collect(), iterations, self.search.value()))
  }
}

/// Returns the raw neural network policy for the current position, without
/// running any Monte Carlo search. A single forward pass produces the policy
/// and value; the legal moves are returned weighted by their policy priors
/// (renormalized over the legal moves) and the value is the estimation.
pub fn policy_moves<N, M>(model: &mut M, field: &Field, player: Player) -> Result<PolicyAnalysis<N>, <M as Model<N>>::E>
where
  N: Float + Sum,
  M: Model<N>,
{
  // The raw policy carries no notion of komi, matching the search which is
  // driven with a zero komi here.
  let komi_x_2 = 0;
  let features = field_features::<N>(field, player, field.width(), field.height(), 0).insert_axis(Axis(0));
  let global = global::<N>(field, player, komi_x_2).insert_axis(Axis(0));

  let (policies, values) = model.predict(features, global)?;

  let policy = policies.index_axis(Axis(0), 0);
  let value = values[(0, 0)] - values[(0, 1)];

  let stride = field.stride;
  let mut moves = Vec::new();
  for pos in field.min_pos()..=field.max_pos() {
    if !field.is_putting_allowed(pos) {
      continue;
    }
    let x = to_x(stride, pos);
    let y = to_y(stride, pos);
    moves.push((pos, policy[(y as usize, x as usize)]));
  }

  Ok((moves, value))
}
