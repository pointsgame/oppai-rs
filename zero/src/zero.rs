use crate::{mcgs::Search, model::Model};
use num_traits::Float;
use oppai_field::{
  field::{Field, Pos},
  player::Player,
};
use std::{
  fmt::{Debug, Display},
  iter::Sum,
};

type Analysis<N> = (Vec<(Pos, u64)>, usize, N);

#[derive(Clone)]
pub struct Zero<N: Float, M: Model<N>> {
  model: M,
  search: Search<N>,
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
    }
  }

  pub fn clear(&mut self) {
    self.search = Search::new();
  }

  pub fn best_moves<SS: Fn() -> bool>(
    &mut self,
    field: &Field,
    player: Player,
    should_stop: &SS,
    max_iterations_count: usize,
  ) -> Result<Analysis<N>, <M as Model<N>>::E> {
    // TODO: persistent tree
    self.clear();

    // TODO: check if game is over
    let mut iterations = 0;
    while !should_stop() && iterations < max_iterations_count {
      self.search.mcgs(&mut field.clone(), player, &mut self.model)?;
      iterations += 1;
    }

    Ok((self.search.visits().collect(), iterations, self.search.value()))
  }
}
