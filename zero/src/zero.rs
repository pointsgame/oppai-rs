use std::{
  fmt::{Debug, Display},
  iter::Sum,
};

use num_traits::Float;
use oppai_field::{
  field::{Field, NonZeroPos},
  player::Player,
};
use rand::Rng;

use crate::{episode::mcts, mcts::MctsNode, model::Model};

#[derive(Clone)]
pub struct Zero<N: Float, M: Model<N>> {
  model: M,
  node: MctsNode<N>,
}

impl<N, M> Zero<N, M>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
{
  pub fn new(model: M) -> Self {
    Zero {
      model,
      node: MctsNode::default(),
    }
  }

  pub fn clear(&mut self) {
    self.node = MctsNode::default();
  }

  pub fn best_move<R: Rng, SS: Fn() -> bool>(
    &mut self,
    field: &Field,
    player: Player,
    rng: &mut R,
    should_stop: &SS,
    max_iterations_count: usize,
  ) -> Result<Option<NonZeroPos>, <M as Model<N>>::E> {
    // TODO: persistent tree
    self.clear();

    // TODO: check if game is over
    let mut iterations = 0;
    while !should_stop() && iterations < max_iterations_count {
      mcts(&mut field.clone(), player, &mut self.node, &self.model, rng)?;
      iterations += 1;
    }

    Ok(self.node.best_move())
  }
}
