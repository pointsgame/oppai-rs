//! Cross-game batched inference.
//!
//! Runs many self-play games concurrently on one executor while a single
//! evaluator collects the positions they submit and evaluates them in one
//! large forward pass. A forward pass costs roughly a fixed overhead plus a
//! small marginal cost per position, so the tiny per-game batches (a handful
//! of leaves from one search) leave a GPU mostly idle; merging the requests
//! of dozens of games into one pass multiplies throughput. Positions from
//! different games are independent, so unlike widening the in-game batch
//! this costs no search quality.
//!
//! [`BatchModel`] is a clonable [`Model`] handle: each game owns a clone and
//! uses it as its model; `predict` forwards the positions to the evaluator
//! and waits for its slice of the merged result. The handles also keep the
//! evaluator's bookkeeping: cloning one announces a new game and dropping it
//! announces that the game is done, so [`run_evaluator`] always knows how
//! many games are in flight. It dispatches a forward pass exactly when every
//! one of them has submitted its positions: between two predictions a game
//! only does a bounded amount of synchronous work, so each live game always
//! eventually either submits a request or finishes, and the wait cannot
//! deadlock. The evaluator terminates once every handle is dropped.

use crate::model::Model;
use futures::{
  StreamExt,
  channel::{mpsc, oneshot},
};
use ndarray::{Array2, Array3, Array4, s};
use num_traits::Float;
use std::fmt::{self, Display, Formatter};
use std::mem;

/// One game's positions awaiting evaluation, along with the channel its slice
/// of the merged result is sent back through.
pub struct BatchRequest<N: Float> {
  features: Array4<N>,
  global: Array2<N>,
  reply: oneshot::Sender<(Array3<N>, Array2<N>)>,
}

/// What [`BatchModel`] handles tell the evaluator.
pub enum Message<N: Float> {
  /// A handle was cloned: one more game will be sending requests.
  Started,
  /// A cloned handle was dropped: its game will send no more requests.
  Finished,
  /// A game's positions to evaluate.
  Request(BatchRequest<N>),
}

/// The evaluator was dropped or failed, so the prediction cannot complete.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Closed;

impl Display for Closed {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "the batch evaluator is closed")
  }
}

impl std::error::Error for Closed {}

/// A [`Model`] that forwards predictions to a shared evaluator.
pub struct BatchModel<N: Float> {
  messages: mpsc::UnboundedSender<Message<N>>,
  /// Whether this handle counts as an active game for the evaluator. The
  /// original handle returned by [`batch_model`] is only a clone source, so
  /// only clones are counted.
  counted: bool,
}

impl<N: Float> Clone for BatchModel<N> {
  fn clone(&self) -> Self {
    let _ = self.messages.unbounded_send(Message::Started);
    BatchModel {
      messages: self.messages.clone(),
      counted: true,
    }
  }
}

impl<N: Float> Drop for BatchModel<N> {
  fn drop(&mut self) {
    if self.counted {
      let _ = self.messages.unbounded_send(Message::Finished);
    }
  }
}

/// Creates a [`BatchModel`] handle and the message stream to pass to
/// [`run_evaluator`]. The returned handle is only a source of clones - give
/// each game its own clone and drop the original once all games are created,
/// so that the evaluator terminates with the last game.
pub fn batch_model<N: Float>() -> (BatchModel<N>, mpsc::UnboundedReceiver<Message<N>>) {
  let (messages, receiver) = mpsc::unbounded();
  (
    BatchModel {
      messages,
      counted: false,
    },
    receiver,
  )
}

impl<N: Float> Model<N> for BatchModel<N> {
  type E = Closed;

  async fn predict(&mut self, inputs: Array4<N>, global: Array2<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    let (reply, result) = oneshot::channel();
    self
      .messages
      .unbounded_send(Message::Request(BatchRequest {
        features: inputs,
        global,
        reply,
      }))
      .map_err(|_| Closed)?;
    result.await.map_err(|_| Closed)
  }
}

/// Serves prediction requests from [`BatchModel`] handles with the underlying
/// model until all handles are dropped, merging the requests of all
/// concurrently running games into large forward passes.
pub async fn run_evaluator<N, M>(model: &mut M, mut messages: mpsc::UnboundedReceiver<Message<N>>) -> Result<(), M::E>
where
  N: Float,
  M: Model<N>,
{
  let mut active = 0usize;
  let mut pending: Vec<BatchRequest<N>> = Vec::new();

  loop {
    // Wait until every game in flight has submitted its positions, so that
    // each forward pass batches the requests of all of them. Games that
    // finish meanwhile announce it and are no longer waited for.
    while pending.is_empty() || pending.len() < active {
      match messages.next().await {
        Some(Message::Started) => active += 1,
        Some(Message::Finished) => active = active.saturating_sub(1),
        Some(Message::Request(request)) => pending.push(request),
        // All handles are gone: any leftover requests belong to cancelled
        // games, so there is nobody left to reply to.
        None => return Ok(()),
      }
    }
    let batch = mem::take(&mut pending);

    // Merge into one forward pass, zero-padding the spatial dimensions: games
    // may play on different board sizes, and the network is masked, so padded
    // evaluation matches training (which always pads to the config size) and
    // the padded area gets no policy mass.
    let channels = batch[0].features.dim().1;
    let global_features = batch[0].global.dim().1;
    let mut positions = 0;
    let mut height = 0;
    let mut width = 0;
    for request in &batch {
      let (n, _, h, w) = request.features.dim();
      positions += n;
      height = height.max(h);
      width = width.max(w);
    }

    let mut features = Array4::zeros((positions, channels, height, width));
    let mut global = Array2::zeros((positions, global_features));
    let mut offset = 0;
    for request in &batch {
      let (n, _, h, w) = request.features.dim();
      features
        .slice_mut(s![offset..offset + n, .., ..h, ..w])
        .assign(&request.features);
      global.slice_mut(s![offset..offset + n, ..]).assign(&request.global);
      offset += n;
    }

    let (policies, values) = model.predict(features, global).await?;

    let mut offset = 0;
    for request in batch {
      let (n, _, h, w) = request.features.dim();
      let policy = policies.slice(s![offset..offset + n, ..h, ..w]).to_owned();
      let value = values.slice(s![offset..offset + n, ..]).to_owned();
      offset += n;
      // The requesting game may have been dropped meanwhile; nothing to do
      // about it here.
      let _ = request.reply.send((policy, value));
    }
  }
}
