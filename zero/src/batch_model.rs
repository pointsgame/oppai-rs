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
//! many games are in flight. The evaluator terminates once every handle is
//! dropped.
//!
//! It dispatches a forward pass as soon as `batch_games` of them have submitted
//! their positions rather than waiting for all of them. Waiting for all is what
//! makes the device and the CPU take turns: no forward pass starts until the last
//! game has submitted, and by then every game is blocked on its reply, so nothing
//! selects while the device works and nothing computes while the games select.
//! Dispatching a part of them instead leaves the rest still selecting, so the two
//! overlap. The batch is smaller for it, which is the trade `batch_games` sets.
//!
//! The wait still cannot deadlock: the target is capped at the number of games
//! actually running, and between two predictions a game only does a bounded
//! amount of synchronous work, so every live game eventually either submits a
//! request or finishes. A dispatch always takes *all* the requests queued, not
//! just the target's worth, so no game waits through a batch it was ready for.

use crate::model::Model;
use futures::{
  StreamExt,
  channel::{mpsc, oneshot},
};
use ndarray::{Array1, Array2, Array3, Array4, s};
use num_traits::Float;
use std::fmt::{self, Display, Formatter};
use std::mem;

/// One game's positions awaiting evaluation, along with the channel its slice
/// of the merged result is sent back through.
pub struct BatchRequest<N: Float> {
  features: Array4<N>,
  global: Array2<N>,
  /// The policy optimism of each position. It travels with the positions rather
  /// than being a property of the evaluator because the games sharing one
  /// evaluator need not search with the same parameters.
  optimism: Array1<N>,
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
  /// What the evaluator's model answers to [`Model::predicts_uncertainty`].
  /// A handle cannot ask it, since the model behind the channel is not part of
  /// this type, so it is told at construction.
  predicts_uncertainty: bool,
}

impl<N: Float> BatchModel<N> {
  /// Another clone source for the same evaluator, which - unlike a clone - does
  /// not announce a game of its own.
  ///
  /// Give one to each thread that plays games, so that every game still descends
  /// from a handle and gets counted, while the per-thread sources themselves do
  /// not. Counting them would leave the evaluator waiting for submissions from
  /// threads rather than games, and those never come.
  pub fn source(&self) -> Self {
    BatchModel {
      messages: self.messages.clone(),
      counted: false,
      predicts_uncertainty: self.predicts_uncertainty,
    }
  }
}

impl<N: Float> Clone for BatchModel<N> {
  fn clone(&self) -> Self {
    let _ = self.messages.unbounded_send(Message::Started);
    BatchModel {
      messages: self.messages.clone(),
      counted: true,
      predicts_uncertainty: self.predicts_uncertainty,
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
///
/// `predicts_uncertainty` must be what the model given to [`run_evaluator`]
/// reports for [`Model::predicts_uncertainty`]; the handles cannot see it.
pub fn batch_model<N: Float>(predicts_uncertainty: bool) -> (BatchModel<N>, mpsc::UnboundedReceiver<Message<N>>) {
  let (messages, receiver) = mpsc::unbounded();
  (
    BatchModel {
      messages,
      counted: false,
      predicts_uncertainty,
    },
    receiver,
  )
}

impl<N: Float> Model<N> for BatchModel<N> {
  type E = Closed;

  fn predicts_uncertainty(&self) -> bool {
    self.predicts_uncertainty
  }

  async fn predict(
    &mut self,
    inputs: Array4<N>,
    global: Array2<N>,
    optimism: Array1<N>,
  ) -> Result<(Array3<N>, Array2<N>), Self::E> {
    let (reply, result) = oneshot::channel();
    self
      .messages
      .unbounded_send(Message::Request(BatchRequest {
        features: inputs,
        global,
        optimism,
        reply,
      }))
      .map_err(|_| Closed)?;
    result.await.map_err(|_| Closed)
  }
}

/// Serves prediction requests from [`BatchModel`] handles with the underlying
/// model until all handles are dropped, merging the requests of concurrently
/// running games into large forward passes.
///
/// `batch_games` is how many games' requests are enough to dispatch a forward
/// pass. The games beyond that keep selecting their next positions while the
/// device works through the batch, so a lower value overlaps the two more and a
/// higher one makes the batches bigger; the number of games in flight is the
/// point where it stops mattering, since the evaluator never waits for more games
/// than are actually running.
pub async fn run_evaluator<N, M>(
  model: &mut M,
  mut messages: mpsc::UnboundedReceiver<Message<N>>,
  batch_games: usize,
) -> Result<(), M::E>
where
  N: Float,
  M: Model<N>,
{
  let batch_games = batch_games.max(1);
  let mut active = 0usize;
  let mut pending: Vec<BatchRequest<N>> = Vec::new();

  loop {
    // Wait for a batch's worth of games, or for every game in flight if fewer
    // than that are left - which is what keeps the last games from waiting for
    // submissions that will never come. Games that finish meanwhile announce it
    // and are no longer waited for.
    while pending.len() < batch_games.min(active).max(1) {
      match messages.next().await {
        Some(Message::Started) => active += 1,
        Some(Message::Finished) => active = active.saturating_sub(1),
        Some(Message::Request(request)) => pending.push(request),
        // All handles are gone: any leftover requests belong to cancelled
        // games, so there is nobody left to reply to.
        None => return Ok(()),
      }
    }
    // Everything queued goes in, not just the target's worth: a game that was
    // ready in time should never be held back to the next forward pass.
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
    let mut optimism = Array1::zeros(positions);
    let mut offset = 0;
    for request in &batch {
      let (n, _, h, w) = request.features.dim();
      features
        .slice_mut(s![offset..offset + n, .., ..h, ..w])
        .assign(&request.features);
      global.slice_mut(s![offset..offset + n, ..]).assign(&request.global);
      optimism.slice_mut(s![offset..offset + n]).assign(&request.optimism);
      offset += n;
    }

    let (policies, values) = model.predict(features, global, optimism).await?;

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
