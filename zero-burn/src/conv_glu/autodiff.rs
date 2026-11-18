//! The backward pass of [`ConvGluBackend::conv_glu`]: it keeps the hidden
//! values and the filters, and recomputes the convolution from them.

use super::ConvGluBackend;
use burn::backend::{
  autodiff::{
    Autodiff, NodeId,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
  },
  tensor::FloatTensor,
};

#[derive(Debug)]
struct ConvGluBackward;

impl<B: ConvGluBackend> Backward<B, 2> for ConvGluBackward {
  /// The hidden values and the filters, which are differentiated; then
  /// `on_board` and the height and width of the grid.
  type State = ([NodeId; 2], FloatTensor<B>, usize, usize);

  fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
    let grad = grads.consume::<B>(&ops.node);
    let ([hidden, weight], on_board, height, width) = ops.state;
    let (grad_hidden, grad_weight) = B::conv_glu_backward(
      checkpointer.retrieve_node_output(hidden),
      on_board,
      checkpointer.retrieve_node_output(weight),
      grad,
      height,
      width,
    );
    for (parent, grad) in ops.parents.into_iter().zip([grad_hidden, grad_weight]) {
      if let Some(parent) = parent {
        grads.register::<B>(parent.id, grad);
      }
    }
  }
}

impl<B: ConvGluBackend, C: CheckpointStrategy> ConvGluBackend for Autodiff<B, C> {
  fn conv_glu(
    hidden: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    weight: FloatTensor<Self>,
    height: usize,
    width: usize,
  ) -> FloatTensor<Self> {
    match ConvGluBackward
      .prepare::<C>([hidden.node(), weight.node()])
      .compute_bound()
      .stateful()
    {
      OpsKind::Tracked(mut prep) => {
        let nodes = [prep.checkpoint(&hidden), prep.checkpoint(&weight)];
        let on_board = on_board.into_primitive();
        let output = B::conv_glu(
          hidden.into_primitive(),
          on_board.clone(),
          weight.into_primitive(),
          height,
          width,
        );
        prep.finish((nodes, on_board, height, width), output)
      }
      OpsKind::UnTracked(prep) => prep.finish(B::conv_glu(
        hidden.into_primitive(),
        on_board.into_primitive(),
        weight.into_primitive(),
        height,
        width,
      )),
    }
  }

  fn conv_glu_backward(
    _hidden: FloatTensor<Self>,
    _on_board: FloatTensor<Self>,
    _weight: FloatTensor<Self>,
    _grad: FloatTensor<Self>,
    _height: usize,
    _width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>) {
    unimplemented!("the backward pass is not differentiated")
  }
}
