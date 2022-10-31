use std::{
  fmt::{Debug, Display},
  iter::Sum,
  sync::atomic::{AtomicBool, Ordering},
};

use num_traits::Float;
use oppai_field::{
  field::{Field, NonZeroPos},
  player::Player,
};
use rand::Rng;

use crate::{episode::mcts, mcts::MctsNode, model::Model};

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

  pub fn best_move<R: Rng>(
    &mut self,
    field: &Field,
    player: Player,
    rng: &mut R,
    should_stop: &AtomicBool,
    max_iterations_count: usize,
  ) -> Result<Option<NonZeroPos>, <M as Model<N>>::E> {
    // TODO: persistent tree
    self.clear();

    // TODO: check if game is over
    let mut iterations = 0;
    while !should_stop.load(Ordering::Relaxed) && iterations < max_iterations_count {
      mcts(&mut field.clone(), player, &mut self.node, &self.model, rng)?;
      iterations += 1;
    }

    Ok(self.node.best_move())
  }
}
