use std::sync::atomic::{AtomicBool, Ordering};

use oppai_field::{
  field::{Field, NonZeroPos},
  player::Player,
};
use rand::Rng;

use crate::{episode::mcts, mcts::MctsNode, model::Model};

pub struct Zero<M> {
  model: M,
  node: MctsNode,
}

impl<M: Model> Zero<M> {
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
  ) -> Result<Option<NonZeroPos>, <M as Model>::E> {
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
