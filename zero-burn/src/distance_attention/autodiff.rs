//! The backward pass of the distance attention: what it keeps of the forward
//! pass is the output and the log-sum-exp of every query's scores, from which
//! [`DistanceAttentionBackend::distance_attention_backward`] recomputes the
//! attention.

use super::DistanceAttentionBackend;
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
struct DistanceAttentionBackward;

impl<B: DistanceAttentionBackend> Backward<B, 4> for DistanceAttentionBackward {
  /// The queries, the keys, the values and the slopes, which are
  /// differentiated; then `on_board`, the output, the log-sum-exp, and the
  /// width of the board.
  type State = (
    [NodeId; 4],
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    usize,
  );

  fn backward(self, ops: Ops<Self::State, 4>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
    let grad = grads.consume::<B>(&ops.node);
    let ([query, key, value, slopes], on_board, output, log_sum_exp, width) = ops.state;
    let (grad_query, grad_key, grad_value, grad_slopes) = B::distance_attention_backward(
      checkpointer.retrieve_node_output(query),
      checkpointer.retrieve_node_output(key),
      checkpointer.retrieve_node_output(value),
      on_board,
      checkpointer.retrieve_node_output(slopes),
      output,
      log_sum_exp,
      grad,
      width,
    );
    for (parent, grad) in ops
      .parents
      .into_iter()
      .zip([grad_query, grad_key, grad_value, grad_slopes])
    {
      if let Some(parent) = parent {
        grads.register::<B>(parent.id, grad);
      }
    }
  }
}

impl<B: DistanceAttentionBackend, C: CheckpointStrategy> DistanceAttentionBackend for Autodiff<B, C> {
  fn distance_attention(
    query: FloatTensor<Self>,
    key: FloatTensor<Self>,
    value: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    slopes: FloatTensor<Self>,
    width: usize,
  ) -> FloatTensor<Self> {
    match DistanceAttentionBackward
      .prepare::<C>([query.node(), key.node(), value.node(), slopes.node()])
      .compute_bound()
      .stateful()
    {
      OpsKind::Tracked(mut prep) => {
        let nodes = [
          prep.checkpoint(&query),
          prep.checkpoint(&key),
          prep.checkpoint(&value),
          prep.checkpoint(&slopes),
        ];
        let on_board = on_board.into_primitive();
        let (output, log_sum_exp) = B::distance_attention_forward(
          query.into_primitive(),
          key.into_primitive(),
          value.into_primitive(),
          on_board.clone(),
          slopes.into_primitive(),
          width,
        );
        prep.finish((nodes, on_board, output.clone(), log_sum_exp, width), output)
      }
      OpsKind::UnTracked(prep) => prep.finish(B::distance_attention(
        query.into_primitive(),
        key.into_primitive(),
        value.into_primitive(),
        on_board.into_primitive(),
        slopes.into_primitive(),
        width,
      )),
    }
  }
}
