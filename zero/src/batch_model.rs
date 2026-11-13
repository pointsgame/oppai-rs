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
//!
//! Dispatching does not wait for the pass it dispatched either: the evaluator
//! keeps `in_flight_passes` of them going at once, so the batch that follows is
//! assembled and enqueued while the previous one is still being computed and
//! read back.
//!
//! A dispatched batch need not be a single pass. A pass is one tensor, so every
//! position in it is padded to the largest board in it, and a small board
//! evaluated next to a large one costs as much as the large one. Each dispatch
//! is therefore cut into the passes that cost the least in total, see
//! [`partition`], weighing the padding a pass saves against the fixed overhead
//! of one more pass.

use crate::model::Model;
use futures::{
  StreamExt,
  channel::{mpsc, oneshot},
  select_biased,
  stream::FuturesUnordered,
};
use ndarray::{Array1, Array2, Array3, Array4, s};
use num_traits::Float;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::time::{Duration, Instant};

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
    &self,
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

/// Evaluates one batch in a single forward pass and hands each request its own
/// slice of the result.
async fn evaluate<N, M>(model: &M, batch: Vec<BatchRequest<N>>) -> Result<(), M::E>
where
  N: Float,
  M: Model<N>,
{
  // Merge into one forward pass, zero-padding the spatial dimensions: games
  // may play on different board sizes, and the network is masked, so padded
  // evaluation matches training (which always pads to the config size) and
  // the padded area gets no policy mass.
  let channels = batch[0].features.dim().1;
  let global_features = batch[0].global.dim().1;
  let Shape {
    positions,
    height,
    width,
    ..
  } = Shape::of(&batch);

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

  Ok(())
}

/// The size of a pass made of some requests.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Shape {
  positions: usize,
  /// The tallest board, which every position is padded to.
  height: usize,
  /// The widest board, which every position is padded to.
  width: usize,
  /// Board cells of the positions as submitted, without the padding.
  cells: usize,
}

impl Shape {
  fn of<N: Float>(batch: &[BatchRequest<N>]) -> Self {
    batch.iter().fold(Shape::default(), |shape, request| {
      let (n, _, h, w) = request.features.dim();
      Shape {
        positions: shape.positions + n,
        height: shape.height.max(h),
        width: shape.width.max(w),
        cells: shape.cells + n * h * w,
      }
    })
  }

  /// Board cells the pass evaluates, padding included.
  fn padded_cells(&self) -> usize {
    self.positions * self.height * self.width
  }
}

/// Where to cut requests of these `(positions, height, width)` sizes, already
/// sorted by board area, into forward passes: the returned indices are the ends
/// of the passes, the last one being `sizes.len()`.
///
/// A pass costs `overhead + positions * height * width`, its height and width
/// being the largest among its requests, and the cuts minimize the total cost.
/// Only contiguous runs of the sorted requests are considered, which is what
/// keeps this a dynamic program over the cut points rather than a search over
/// all the ways to group the requests. Sorting by area keeps the boards that pad
/// each other cheaply next to each other; with boards of other proportions than
/// square it is not exact, but a request of a different shape and similar area
/// pads the others by little either way.
///
/// An infinite overhead never cuts, a zero one cuts between any two shapes.
pub(crate) fn partition_sizes(sizes: &[(usize, usize, usize)], overhead: f64) -> Vec<usize> {
  if !overhead.is_finite() {
    return vec![sizes.len()];
  }
  // `best[j]` is the least cost of the first `j` requests, with the fewest
  // passes among the splits that cost as much, and `cut[j]` where the last pass
  // of that split starts.
  let mut best = vec![(0.0, 0); sizes.len() + 1];
  let mut cut = vec![0; sizes.len() + 1];
  for end in 1..=sizes.len() {
    best[end] = (f64::INFINITY, 0);
    let (mut positions, mut height, mut width) = (0, 0, 0);
    for start in (0..end).rev() {
      let (n, h, w) = sizes[start];
      positions += n;
      height = height.max(h);
      width = width.max(w);
      let (cost, passes) = best[start];
      let candidate = (cost + overhead + (positions * height * width) as f64, passes + 1);
      if candidate.0 < best[end].0 || candidate.0 == best[end].0 && candidate.1 < best[end].1 {
        best[end] = candidate;
        cut[end] = start;
      }
    }
  }
  let mut ends = Vec::new();
  let mut end = sizes.len();
  while end > 0 {
    ends.push(end);
    end = cut[end];
  }
  ends.reverse();
  ends
}

/// Cuts a dispatch into forward passes as [`partition_sizes`] says, given the
/// fixed overhead of a pass in board cells.
fn partition<N: Float>(mut requests: Vec<BatchRequest<N>>, overhead: f64) -> Vec<Vec<BatchRequest<N>>> {
  if !overhead.is_finite() || requests.len() < 2 {
    return vec![requests];
  }
  requests.sort_by_key(|request| {
    let (_, _, h, w) = request.features.dim();
    (h * w, h, w)
  });
  let sizes = requests
    .iter()
    .map(|request| {
      let (n, _, h, w) = request.features.dim();
      (n, h, w)
    })
    .collect::<Vec<_>>();
  let mut passes = Vec::new();
  let mut start = 0;
  for end in partition_sizes(&sizes, overhead) {
    let rest = requests.split_off(end - start);
    passes.push(mem::replace(&mut requests, rest));
    start = end;
  }
  passes
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
///
/// `pass_overhead` is the fixed cost of a forward pass, in the board cells
/// evaluating which would take as long. It decides when a dispatch of boards of
/// different sizes is cut into several passes rather than padded to the largest
/// of them; infinity never cuts. Measured on the device alone it is a few
/// thousand cells, but an extra pass costs more than its device time: the
/// evaluator spends more on merging and reading back, and when the games rather
/// than the device are what holds self-play back, that is all it costs. So it
/// had better be overestimated, which only gives up the cuts that would save
/// little, while those that matter - small boards padded to far larger ones -
/// save many times any overhead.
///
/// `in_flight_passes` is how many forward passes may be under way at once. A
/// pass is not done with the CPU when it reaches the device: its result still
/// has to be read back, and the batch after it has to be merged and enqueued,
/// and the device idles through all of that if it has nothing else queued.
/// A second pass fills that gap - the device starts it as soon as it frees up,
/// while the previous one is being read back and the next is being assembled.
/// Beyond that there is little left to hide, one device being all there is to
/// keep busy, and the batches only get smaller: a pass dispatched earlier is a
/// pass fewer of the still-selecting games make it into. A dispatch cut into
/// several passes may go over the limit, in which case the next one waits until
/// enough of them are done.
pub async fn run_evaluator<N, M>(
  model: &M,
  mut messages: mpsc::UnboundedReceiver<Message<N>>,
  batch_games: usize,
  pass_overhead: f64,
  in_flight_passes: usize,
) -> Result<(), M::E>
where
  N: Float,
  M: Model<N>,
{
  let batch_games = batch_games.max(1);
  let in_flight_passes = in_flight_passes.max(1);
  let mut active = 0usize;
  let mut pending: Vec<BatchRequest<N>> = Vec::new();
  let mut in_flight = FuturesUnordered::new();
  let mut throughput = Throughput::new();

  loop {
    throughput.report_every(REPORT_PERIOD);
    // Wait for a batch's worth of games, or for every game in flight if fewer
    // than that are left - which is what keeps the last games from waiting for
    // submissions that will never come. Games that finish meanwhile announce it
    // and are no longer waited for. Requests that arrive while every pass slot
    // is taken wait here too, and join the batch dispatched once one frees up.
    while in_flight.len() >= in_flight_passes || pending.len() < batch_games.min(active).max(1) {
      let message = if in_flight.is_empty() {
        messages.next().await
      } else {
        select_biased! {
          // The passes already dispatched come first: the games waiting on them
          // have nothing to submit until they are answered, so this is the only
          // branch that can be making progress when the other one is idle.
          result = in_flight.next() => {
            if let Some(result) = result {
              result?;
            }
            if in_flight.is_empty() {
              throughput.drained();
            }
            continue;
          },
          message = messages.next() => message,
        }
      };
      match message {
        Some(Message::Started) => active += 1,
        Some(Message::Finished) => active = active.saturating_sub(1),
        Some(Message::Request(request)) => pending.push(request),
        // All handles are gone: any leftover requests belong to cancelled
        // games, and so do the passes still in flight, so there is nobody left
        // to reply to.
        None => {
          throughput.report();
          return Ok(());
        }
      }
    }
    // Everything queued goes in, not just the target's worth: a game that was
    // ready in time should never be held back to the next forward pass.
    let passes = partition(mem::take(&mut pending), pass_overhead);
    throughput.dispatched(&passes, in_flight.is_empty());
    // The passes are not awaited here - they are driven by the loop above, which
    // meanwhile keeps collecting the requests of the games it does not block.
    for pass in passes {
      in_flight.push(evaluate(model, pass));
    }
  }
}

/// How often [`run_evaluator`] logs its throughput.
const REPORT_PERIOD: Duration = Duration::from_secs(60);

/// What [`run_evaluator`] has done since its last report, which is what tells
/// whether self-play is held back by the device or by the games feeding it.
struct Throughput {
  since: Instant,
  dispatches: u64,
  passes: u64,
  requests: u64,
  positions: u64,
  /// Board cells of the positions as submitted.
  cells: u64,
  /// Board cells actually evaluated, including the padding up to the largest
  /// board of each pass.
  padded_cells: u64,
  /// Time with no pass in flight: the device had nothing of ours to work on
  /// because every game was still selecting.
  idle: Duration,
  /// Since when no pass is in flight.
  idle_since: Option<Instant>,
}

impl Throughput {
  fn new() -> Self {
    let now = Instant::now();
    Throughput {
      since: now,
      dispatches: 0,
      passes: 0,
      requests: 0,
      positions: 0,
      cells: 0,
      padded_cells: 0,
      idle: Duration::ZERO,
      idle_since: Some(now),
    }
  }

  /// The last pass in flight has completed.
  fn drained(&mut self) {
    self.idle_since = Some(Instant::now());
  }

  /// These passes are about to be dispatched together.
  fn dispatched<N: Float>(&mut self, passes: &[Vec<BatchRequest<N>>], was_idle: bool) {
    self.dispatches += 1;
    for pass in passes {
      let shape = Shape::of(pass);
      self.passes += 1;
      self.requests += pass.len() as u64;
      self.positions += shape.positions as u64;
      self.cells += shape.cells as u64;
      self.padded_cells += shape.padded_cells() as u64;
    }
    if was_idle && let Some(since) = self.idle_since.take() {
      self.idle += since.elapsed();
    }
  }

  fn report_every(&mut self, period: Duration) {
    if self.since.elapsed() >= period {
      self.report();
    }
  }

  fn report(&mut self) {
    let now = Instant::now();
    let elapsed = now - self.since;
    let mut idle = self.idle;
    if let Some(since) = self.idle_since {
      idle += now - since;
      self.idle_since = Some(now);
    }
    if self.passes > 0 {
      let seconds = elapsed.as_secs_f64();
      let passes = self.passes as f64;
      log::info!(
        "Evaluator: {:.0} positions/s, {:.1} passes/s, {:.1} positions and {:.1} games per pass, {:.2} passes per \
         dispatch, {:.0}% padding, {:.0}% of the time with no pass in flight",
        self.positions as f64 / seconds,
        passes / seconds,
        self.positions as f64 / passes,
        self.requests as f64 / passes,
        passes / self.dispatches as f64,
        100.0 * (1.0 - self.cells as f64 / self.padded_cells as f64),
        100.0 * idle.as_secs_f64() / seconds,
      );
    }
    let idle_since = self.idle_since;
    *self = Throughput::new();
    self.idle_since = idle_since;
  }
}
