mod config;

use anyhow::{Error, Result};
use burn::{
  module::Module,
  optim::{SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig},
  tensor::{Device, DeviceConfig, FloatDType},
};
use config::{
  Action, BenchParams, Config, CountParams, InitParams, PitParams, PlayParams, RecalcParams, TrainParams, cli_parse,
};
use flate2::{Compression, read::MultiGzDecoder, write::GzEncoder};
use futures::StreamExt;
use ndarray::{Array1, Array2, Array4};
use num_traits::Float;
use oppai_field::{
  any_field::AnyField,
  extended_field::ExtendedField,
  field::{Field, Hash, length},
  player::Player,
  zobrist::Zobrist,
};
use oppai_sgf::{from_sgf, to_sgf};
use oppai_zero::{
  batch_model::{batch_model, run_evaluator},
  episode::{Visits, episode},
  examples::{Examples, is_validation_game},
  field_features::{CHANNELS, GLOBAL_FEATURES},
  mcgs::{Params, Search},
  model::{Model, TrainableModel},
  opening::opening,
  pit,
  random_model::RandomModel,
};
use oppai_zero_burn::device::list_devices;
use oppai_zero_burn::model::{Learner, Model as BurnModel, Predictor, ema_update};
use oppai_zero_sgf::{sgf_to_visits, visits_to_sgf};
use rand::{Rng, RngExt, SeedableRng, distr::uniform::SampleUniform, make_rng, rngs::SmallRng};
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use sgf_parse::{GameTree, SgfNode, SimpleText, serialize, unknown_game::Prop};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  fs::File,
  io::{BufRead, BufReader, Write},
  iter::{self, Sum},
  path::Path,
  process::ExitCode,
  sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize},
  },
};

fn init(params: InitParams, device: Device) -> Result<ExitCode> {
  let mut model = BurnModel::new(&device, &params.model_config);
  model.initialize(&device);
  model.save_file(params.model)?;

  let optimizer = SgdConfig::new()
    .with_weight_decay((params.weight_decay > 0.0).then(|| WeightDecayConfig::new(params.weight_decay)))
    .with_momentum(Some(MomentumConfig::new().with_dampening(0.0)))
    .init();
  optimizer.save(params.optimizer)?;

  Ok(ExitCode::SUCCESS)
}

/// Streams games from a file of concatenated gzip members, decompressing and
/// parsing one game (line) at a time.
fn read_games<P: AsRef<Path>>(path: P) -> Result<impl Iterator<Item = Result<SgfNode<Prop>>>> {
  let file = File::open(path)?;
  let lines = BufReader::new(MultiGzDecoder::new(BufReader::new(file))).lines();
  Ok(lines.flat_map(|line| {
    let nodes = line.map_err(Error::from).and_then(|line| {
      Ok(
        sgf_parse::parse(&line)?
          .into_iter()
          .filter_map(|tree| match tree {
            GameTree::Unknown(node) => Some(node),
            GameTree::GoGame(_) => None,
          })
          .collect::<Vec<_>>(),
      )
    });
    match nodes {
      Ok(nodes) => nodes.into_iter().map(Ok).collect::<Vec<_>>(),
      Err(e) => vec![Err(e)],
    }
  }))
}

/// Appends a game as a separate gzip member, so an interrupted process never
/// corrupts previously written games and appending remains valid gzip.
fn write_sgf(file: &mut File, sgf: &str) -> Result<()> {
  let mut encoder = GzEncoder::new(&mut *file, Compression::default());
  writeln!(encoder, "{sgf}")?;
  encoder.finish()?;
  file.flush()?;
  Ok(())
}

fn write_game(file: &mut File, field: &ExtendedField, visits: &[Visits], komi_x_2: i32) -> Result<()> {
  if let Some(mut node) = to_sgf(field) {
    visits_to_sgf(&mut node, visits, field.field().stride, field.field().moves_count());
    let score_x_2 = field.field().score(Player::Red) * 2 + komi_x_2;
    node.properties.push(Prop::RE(match score_x_2.cmp(&0) {
      Ordering::Equal => "0".into(),
      Ordering::Greater => SimpleText {
        text: format!("W+{}", score_x_2 as f32 / 2.0),
      },
      Ordering::Less => SimpleText {
        text: format!("B+{}", score_x_2.abs() as f32 / 2.0),
      },
    }));
    node
      .properties
      .push(Prop::Unknown("KM".into(), vec![(komi_x_2 as f32 / 2.0).to_string()]));
    let sgf = serialize(iter::once(&GameTree::Unknown(node)));
    write_sgf(file, &sgf)?;
  }
  Ok(())
}

/// Plays games claimed from the shared `next_game` ticket, up to `parallel` of
/// them concurrently, creating a fresh model per game with `new_model`.
///
/// One of these runs per thread. Games are claimed one at a time rather than
/// handed out in equal shares up front, because their lengths vary widely: a
/// thread that drew the short ones would otherwise sit idle while the others were
/// still playing. Every thread keeps taking games until none are left.
///
/// The games file is shared with the other threads. Games are written as
/// independent gzip members, so the lock only has to keep two of them from
/// interleaving.
#[allow(clippy::too_many_arguments)]
async fn play_games<N, M, MF, R>(
  params: &PlayParams,
  next_game: &AtomicUsize,
  parallel: usize,
  mut new_model: MF,
  rng: &mut R,
  should_stop: &AtomicBool,
  file: &Mutex<File>,
) -> Result<()>
where
  N: Float + Sum + SampleUniform + Display + Debug,
  M: Model<N>,
  M::E: Debug,
  MF: FnMut() -> M,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  let games = iter::from_fn(|| {
    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
      return None;
    }
    // Claim the next game, or stop once they have all been taken. The stream below
    // pulls from here only as a slot frees up, so a game is claimed exactly when a
    // thread is about to start playing it.
    //
    // The ticket counts up rather than a stock counting down, because a counter of
    // games left would have to be decremented with `fetch_sub`, and that wraps: the
    // thread that took the last game would leave `usize::MAX` behind and every
    // later claim would succeed. Counting up cannot wrap into a valid ticket.
    if next_game.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= params.count {
      return None;
    }
    Some(())
  })
  .map(|()| {
    let mut rng = SmallRng::from_seed(rng.random());
    let model = new_model();
    let width = params.width[rng.random_range(0..params.width.len())];
    let height = params.height[rng.random_range(0..params.height.len())];
    let op = opening(width, height, &mut rng);
    let komi_x_2_count = params
      .komi_x_2
      .iter()
      .copied()
      .filter(|&komi_x_2| (komi_x_2.unsigned_abs() as usize) < op.len())
      .count();
    let komi_x_2 = params
      .komi_x_2
      .iter()
      .copied()
      .filter(|&komi_x_2| (komi_x_2.unsigned_abs() as usize) < op.len())
      .nth(rng.random_range(0..komi_x_2_count))
      .unwrap();
    async move {
      let mut player = Player::Red;
      let mut field = Field::new_from_rng(width, height, &mut rng);
      for (x, y) in op {
        let pos = field.to_pos(x, y);
        assert!(field.put_point(pos, player));
        field.update_grounded();
        player = player.next();
      }

      let visits = episode(&mut field, player, &model, komi_x_2, params.readouts, &mut rng)
        .await
        .map_err(|e| anyhow::anyhow!("model failure: {:?}", e))?;

      Ok::<_, Error>((field, visits, komi_x_2))
    }
  });

  let mut games = futures::stream::iter(games).buffer_unordered(parallel);

  while let Some(game) = games.next().await {
    let (field, visits, komi_x_2) = game?;
    let mut file = file
      .lock()
      .map_err(|_| anyhow::anyhow!("the games file lock is poisoned"))?;
    write_game(&mut file, &field.into(), &visits, komi_x_2)?;
  }

  Ok(())
}

/// Splits `total` as evenly as possible into `parts`, giving the remainder to the
/// first ones.
fn split(total: usize, parts: usize, index: usize) -> usize {
  total / parts + usize::from(index < total % parts)
}

/// Plays `params.count` games across `params.threads` threads.
///
/// The games are CPU bound between forward passes - selecting paths, expanding
/// nodes, replaying the field - so running them concurrently on one thread, as a
/// single executor does, leaves that work on a single core no matter how many
/// games are in flight. Here each thread drives its own share of the games while
/// one evaluator, on this thread, serves all of them from one device.
fn play<R: Rng>(params: PlayParams, device: Device, rng: &mut R, should_stop: Arc<AtomicBool>) -> Result<ExitCode> {
  // `count` and `parallel_games` stay the totals they always were, so existing
  // configurations keep their meaning. The games are claimed from one ticket as
  // threads become free, while the concurrency budget is split up front since it
  // only bounds how many games a thread juggles at once. More threads than either
  // total would leave the extra ones with nothing to do.
  let threads = params.threads.clamp(1, params.count.min(params.parallel_games).max(1));
  let next_game = AtomicUsize::new(0);
  let file = Mutex::new(File::options().append(true).create(true).open(&params.games)?);
  log::info!(
    "Playing {} games on {} threads, {} concurrent",
    params.count,
    threads,
    params.parallel_games
  );

  match params.model.clone() {
    Some(model_path) => {
      let model = BurnModel::load(&device, &params.model_config, model_path)?;
      let predictor = Predictor { model, device };

      // All games share one evaluator: their positions are merged into large
      // forward passes instead of each game evaluating its own tiny batch. Only
      // this thread ever touches the model, so the backend needs nothing of its
      // own to be safe under threads.
      let (handle, requests) = batch_model::<f32>(predictor.predicts_uncertainty());

      std::thread::scope(|scope| -> Result<()> {
        let workers = (0..threads)
          .map(|i| {
            // Seeded here rather than inside the thread: the seeds come from one
            // generator, which only this thread can reach.
            let mut shard_rng = SmallRng::from_seed(rng.random());
            let parallel = split(params.parallel_games, threads, i);
            let source = handle.source();
            let params = &params;
            let should_stop = should_stop.as_ref();
            let file = &file;
            let next_game = &next_game;
            scope.spawn(move || {
              futures::executor::block_on(play_games(
                params,
                next_game,
                parallel,
                || source.clone(),
                &mut shard_rng,
                should_stop,
                file,
              ))
            })
          })
          .collect::<Vec<_>>();
        // Every game's handle descends from a worker's source, so once the workers
        // are gone the channel closes and the evaluator returns.
        drop(handle);

        let evaluator_result = futures::executor::block_on(run_evaluator(
          &predictor,
          requests,
          params.batch_games,
          params.pass_overhead,
          params.in_flight_passes,
        ));
        // Join before reporting either failure, so a panicking worker is never
        // left running past the end of the scope.
        let games_results = workers
          .into_iter()
          .map(|worker| {
            worker
              .join()
              .map_err(|_| anyhow::anyhow!("a self-play thread panicked"))
          })
          .collect::<Vec<_>>();
        evaluator_result?;
        for result in games_results {
          result??;
        }
        Ok(())
      })?;
    }
    None => {
      std::thread::scope(|scope| -> Result<()> {
        let workers = (0..threads)
          .map(|i| {
            let mut shard_rng = SmallRng::from_seed(rng.random());
            let mut seeder = SmallRng::from_seed(shard_rng.random());
            let parallel = split(params.parallel_games, threads, i);
            let params = &params;
            let should_stop = should_stop.as_ref();
            let file = &file;
            let next_game = &next_game;
            scope.spawn(move || {
              futures::executor::block_on(play_games::<f32, _, _, _>(
                params,
                next_game,
                parallel,
                || RandomModel::new(SmallRng::from_seed(seeder.random())),
                &mut shard_rng,
                should_stop,
                file,
              ))
            })
          })
          .collect::<Vec<_>>();
        for worker in workers {
          worker
            .join()
            .map_err(|_| anyhow::anyhow!("a self-play thread panicked"))??;
        }
        Ok(())
      })?;
    }
  }

  Ok(ExitCode::SUCCESS)
}

fn train<R: Rng>(params: TrainParams, device: Device, rng: &mut R, should_stop: Arc<AtomicBool>) -> Result<ExitCode> {
  let model = BurnModel::new(&device, &params.model_config).try_load_file(params.model)?;
  // The gradients are clipped by their combined norm before the step instead of
  // by the optimizer, which would bound every parameter's norm on its own.
  log::info!(
    "Optimizer {} with weight decay {} and {}, learning rate from {} to {}",
    params.optimizer_config.kind,
    params.optimizer_config.weight_decay,
    params.optimizer_config.lookahead.map_or_else(
      || "no lookahead".to_string(),
      |lookahead| format!("lookahead of {} steps by {}", lookahead.steps, lookahead.alpha)
    ),
    params.learning_rate_start,
    params.learning_rate_end
  );
  let optimizer = params.optimizer_config.init().load(&params.optimizer)?;
  // The SWA model is an exponential moving average of the trained weights,
  // updated every `swa_period` batches and saved separately; it's the model to
  // export for self-play while training continues from the raw weights.
  let mut swa_model = params
    .model_swa_new
    .is_some()
    .then(|| {
      params.model_swa.as_ref().map_or_else(
        || Ok(model.clone()),
        |path| BurnModel::new(&device, &params.model_config).try_load_file(path),
      )
    })
    .transpose()?;
  let predictor = Predictor {
    model,
    device: device.clone(),
  };
  let mut learner = Learner {
    predictor,
    optimizer,
    gradient_clipping: params.gradient_clipping,
    compute_dtype: params.mixed_precision.then_some(FloatDType::BF16),
  };

  let mut examples = Examples::default();
  let mut validation = Examples::default();
  for path in &params.games {
    for node in read_games(path)? {
      let node = node?;
      let field = from_sgf::<Field, _>(&node, rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
      let visits = sgf_to_visits(&node, field.stride);
      let komi_x_2 = node
        .properties
        .iter()
        .find_map(|prop| match prop {
          Prop::Unknown(name, values) if name == "KM" => values.first().map(|value| {
            let value = value.parse::<f32>().unwrap();
            (value * 2.0).round() as i32
          }),
          _ => None,
        })
        .unwrap_or(0);

      if field.width() > params.width || field.height() > params.height {
        return Err(anyhow::anyhow!(
          "Game is bigger than config: {}:{}",
          field.width(),
          field.height()
        ));
      }

      let examples = if is_validation_game(&field, komi_x_2, params.validation_fraction) {
        &mut validation
      } else {
        &mut examples
      };
      examples.add(
        komi_x_2,
        visits,
        &field,
        field.width() <= params.height && field.height() <= params.width,
        !params.ignore_surprise,
        rng,
      );
    }
  }

  log::info!(
    "Training on {} examples of {} games, validating on {} examples of {} games",
    examples.len(),
    examples.games.len(),
    validation.len(),
    validation.games.len()
  );
  let zobrist = Arc::new(Zobrist::new(length(params.width, params.height) * 3, rng));
  validation.group_by_size(params.batch_size, rng);
  validate(&learner, &validation, params.batch_size, zobrist.clone(), "before");

  // Shuffled, so that the batches trained on are a random sample of the
  // examples when only some of them are, then grouped by board size so that
  // a batch is padded to its own boards rather than the largest of all.
  examples.shuffle(rng);
  examples.group_by_size(params.batch_size, rng);
  let batches_count = params.train_examples.map_or_else(
    || examples.batches_count(params.batch_size),
    |train_examples| {
      examples
        .batches_count(params.batch_size)
        .min((train_examples / params.batch_size).max(1))
    },
  );
  log::info!("Training on {} batches of {}", batches_count, params.batch_size);
  // By default average a snapshot every half-epoch, so a single training run
  // contributes two snapshots to the moving average.
  let swa_period = params.swa_period.unwrap_or((batches_count / 2).max(1));
  for (i, batch) in examples
    .batches(zobrist.clone(), params.batch_size)
    .take(batches_count)
    .enumerate()
    .skip(params.skip)
  {
    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
      log::info!("Stopping training after {} batches", i);
      break;
    }
    if i.is_multiple_of(64) {
      log::info!("Batch {} out of {}", i, batches_count);
    }
    let progress = if batches_count > 1 {
      i as f64 / (batches_count - 1) as f64
    } else {
      0.0
    };
    let learning_rate = params.learning_rate_start + (params.learning_rate_end - params.learning_rate_start) * progress;
    learner = learner.train(
      batch.inputs,
      batch.global,
      batch.policies,
      batch.opponent_policies,
      batch.values,
      batch.td_values,
      batch.td_scores,
      batch.scores,
      batch.captured,
      batch.q_values,
      learning_rate,
    )?;

    if let Some(swa) = swa_model.take() {
      swa_model = Some(if (i + 1).is_multiple_of(swa_period) {
        ema_update(swa, &learner.predictor.model, 1.0 / params.swa_scale, &device)
      } else {
        swa
      });
    }
  }

  validate(&learner, &validation, params.batch_size, zobrist, "after");

  learner.predictor.model.save_file(params.model_new)?;

  if let (Some(swa), Some(path)) = (swa_model, params.model_swa_new) {
    swa.save_file(path)?;
  }

  learner.optimizer.save(params.optimizer_new)?;

  Ok(ExitCode::SUCCESS)
}

/// Logs every loss term of the model averaged over the validation examples,
/// `when` saying at which point of the training run.
fn validate(learner: &Learner, validation: &Examples, batch_size: usize, zobrist: Arc<Zobrist<u64>>, when: &str) {
  if validation.is_empty() {
    return;
  }
  // Nothing is trained on these losses, so they are computed without
  // recording a backward pass, which would keep every activation around.
  let predictor = learner.predictor.valid();
  let mut sums: Vec<(&'static str, f64)> = Vec::new();
  let mut rows = 0;
  for batch in validation.batches(zobrist, batch_size) {
    // The terms are means over the batch, so they are weighed by its rows.
    let batch_rows = batch.inputs.dim().0;
    let losses = predictor
      .losses(
        learner.compute_dtype,
        batch.inputs,
        batch.global,
        batch.policies,
        batch.opponent_policies,
        batch.values,
        batch.td_values,
        batch.td_scores,
        batch.scores,
        batch.captured,
        batch.q_values,
      )
      .values();
    if sums.is_empty() {
      sums = losses.iter().map(|&(name, _)| (name, 0.0)).collect();
    }
    for ((_, sum), (_, loss)) in sums.iter_mut().zip(losses) {
      *sum += f64::from(loss) * batch_rows as f64;
    }
    rows += batch_rows;
  }
  let mut message = format!("Validation loss {when}:");
  for (name, sum) in sums {
    message += &format!(" {} {}", name, sum / rows as f64);
  }
  log::info!("{}", message);
}

/// The board and the opening a pair of pit games is played on.
struct PitPair {
  width: u32,
  height: u32,
  opening: Vec<(u32, u32)>,
}

/// A finished pit game: its place in the order games are counted in, its score
/// from the new model's side, and the game itself to save.
struct PitGame {
  index: usize,
  score: i32,
  sgf: Option<String>,
}

/// Plays pit games claimed from the shared `next_game` ticket, up to `parallel`
/// of them concurrently, and sends each one that finishes to `results`. Games
/// `2k` and `2k + 1` are the `k`-th pair, played from the same opening, the new
/// model moving first in the even one. A game still being played when `stop`
/// turns true is abandoned, and no more are claimed.
#[allow(clippy::too_many_arguments)]
async fn pit_games<M, R, S>(
  pairs: &[PitPair],
  zobrist: &Arc<Zobrist<Hash>>,
  next_game: &AtomicUsize,
  parallel: usize,
  model_old: &M,
  model_new: &M,
  rng: &mut R,
  stop: S,
  results: &futures::channel::mpsc::UnboundedSender<PitGame>,
) -> Result<()>
where
  M: Model<f32> + Clone,
  M::E: Debug,
  R: Rng,
  S: Fn() -> bool + Copy,
{
  let total = pairs.len() * 2;
  let games = iter::from_fn(|| {
    if stop() {
      return None;
    }
    let index = next_game.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (index < total).then_some(index)
  })
  .map(|index| {
    let mut rng = SmallRng::from_seed(rng.random());
    let pair = &pairs[index / 2];
    async move {
      let mut field = Field::new(pair.width, pair.height, zobrist.clone());
      let mut player = Player::Red;
      for &(x, y) in &pair.opening {
        let pos = field.to_pos(x, y);
        assert!(field.put_point(pos, player));
        field.update_grounded();
        player = player.next();
      }
      let score = if index.is_multiple_of(2) {
        pit::play(&mut field, player, model_new, model_old, 0, &mut rng, stop).await
      } else {
        pit::play(&mut field, player, model_old, model_new, 0, &mut rng, stop)
          .await
          .map(|score| score.map(|score| -score))
      }
      .map_err(|e| anyhow::anyhow!("model failure: {:?}", e))?;
      Ok::<_, Error>(score.map(|score| PitGame {
        index,
        score,
        sgf: to_sgf(&field.into()).map(|node| serialize(iter::once(&GameTree::Unknown(node)))),
      }))
    }
  });

  let mut games = futures::stream::iter(games).buffer_unordered(parallel);
  while let Some(game) = games.next().await {
    if let Some(game) = game? {
      // Once the outcome is decided nobody listens any more, and the games that
      // finish after that are of no use.
      let _ = results.unbounded_send(game);
    }
  }

  Ok(())
}

/// Whether the games played so far settle if the new model is accepted:
/// `Some` with the answer once they do, `None` while more games are needed.
///
/// Called after every pair, so the colours stay balanced in whatever is counted.
/// The answer is settled once no result of the games left can change it, or
/// earlier by a sequential probability ratio test between the new model winning
/// `win_rate_threshold` plus and minus `early_stop_margin` of its games, at the
/// `early_stop_error` rate of picking the wrong one of the two. A model far
/// stronger or weaker than the threshold is then decided after a handful of
/// pairs, while one near it still plays them all.
fn pit_decision(params: &PitParams, wins: u64, losses: u64, played: u64) -> Option<bool> {
  let total = params.count * 2;
  let draws = played - wins - losses;
  let win_rate = |more_wins: u64| ((wins + more_wins) as f64 + draws as f64 / 2.0) / total as f64;
  if win_rate(total - played) <= params.win_rate_threshold {
    return Some(false);
  }
  if win_rate(0) > params.win_rate_threshold {
    return Some(true);
  }

  if params.early_stop_margin > 0.0 {
    let better = params.win_rate_threshold + params.early_stop_margin;
    let worse = params.win_rate_threshold - params.early_stop_margin;
    let wins = wins as f64 + draws as f64 / 2.0;
    let losses = losses as f64 + draws as f64 / 2.0;
    let llr = wins * (better / worse).ln() + losses * ((1.0 - better) / (1.0 - worse)).ln();
    let error = params.early_stop_error;
    let bound = ((1.0 - error) / error).ln();
    if llr.abs() >= bound {
      log::info!(
        "Decided by the sequential test after {} games: log likelihood ratio {:.2}, bound {:.2}",
        played,
        llr,
        bound
      );
      return Some(llr > 0.0);
    }
  }

  None
}

/// Counts the pit games sent to `results` and saves them, both in the order of
/// their indices rather than as they finish: short games finish first, and
/// deciding on whichever games happen to be done would favour the model that
/// wins quickly. Returns the outcome once it's decided, or `false` if the games
/// stop coming before that.
async fn pit_results(
  params: &PitParams,
  mut results: futures::channel::mpsc::UnboundedReceiver<PitGame>,
) -> Result<bool> {
  let total = params.count as usize * 2;
  let mut file = params
    .games
    .as_ref()
    .map(|games| File::options().append(true).create(true).open(games))
    .transpose()?;
  let mut finished = iter::repeat_with(|| None).take(total).collect::<Vec<Option<PitGame>>>();
  let mut wins = 0u64;
  let mut losses = 0u64;
  let mut played = 0usize;

  while let Some(game) = results.next().await {
    let index = game.index;
    finished[index] = Some(game);
    while let Some(game) = finished.get_mut(played).and_then(Option::take) {
      match game.score.cmp(&0) {
        Ordering::Less => losses += 1,
        Ordering::Greater => wins += 1,
        Ordering::Equal => {}
      };
      if let (Some(file), Some(sgf)) = (file.as_mut(), game.sgf) {
        write_sgf(file, &sgf)?;
      }
      played += 1;
      log::info!(
        "Game {}, result {}/{}/{}",
        played,
        wins,
        played as u64 - wins - losses,
        losses
      );
      if played.is_multiple_of(2)
        && let Some(outcome) = pit_decision(params, wins, losses, played as u64)
      {
        return Ok(outcome);
      }
    }
  }

  log::info!("Stopping after {} games", played);
  Ok(false)
}

/// Pits the new model against the old one, playing up to `params.count` pairs
/// of games across `params.threads` threads.
///
/// As in self-play, the games run concurrently and each model's positions are
/// merged into large forward passes by an evaluator of its own, both on this
/// thread. A game counts for an evaluator only while that evaluator's model is
/// on the move (see [`pit::play`]), so each evaluator waits for the games
/// searching with its model and never for ones blocked on the other.
fn pit<R: Rng>(params: PitParams, device: Device, rng: &mut R, should_stop: Arc<AtomicBool>) -> Result<ExitCode> {
  anyhow::ensure!(
    params.early_stop_margin >= 0.0
      && (params.early_stop_margin == 0.0
        || (params.win_rate_threshold - params.early_stop_margin > 0.0
          && params.win_rate_threshold + params.early_stop_margin < 1.0)),
    "the win rate threshold plus and minus the early stop margin must be win rates"
  );
  anyhow::ensure!(
    params.early_stop_error > 0.0 && params.early_stop_error < 0.5,
    "the early stop error must be between 0 and 0.5"
  );

  let model_old = BurnModel::load(&device, &params.model_config, params.model.clone())?;
  let model_old = Predictor {
    model: model_old,
    device: device.clone(),
  };

  let model_new = BurnModel::load(&device, &params.model_config_new, params.model_new.clone())?;
  let model_new = Predictor {
    model: model_new,
    device,
  };

  let zobrist = Arc::new(Zobrist::new(
    length(
      *Iterator::max(params.width.iter()).unwrap(),
      *Iterator::max(params.height.iter()).unwrap(),
    ) * 3,
    rng,
  ));

  let pairs = (0..params.count)
    .map(|_| {
      let width = params.width[rng.random_range(0..params.width.len())];
      let height = params.height[rng.random_range(0..params.height.len())];
      let opening = opening(width, height, rng);
      PitPair { width, height, opening }
    })
    .collect::<Vec<_>>();
  let total = pairs.len() * 2;
  let threads = params.threads.clamp(1, params.parallel_games.min(total).max(1));
  let next_game = AtomicUsize::new(0);
  // Set once the outcome is decided, or a thread failed.
  let decided = AtomicBool::new(false);
  let stop =
    || decided.load(std::sync::atomic::Ordering::Relaxed) || should_stop.load(std::sync::atomic::Ordering::Relaxed);
  log::info!(
    "Playing up to {} games on {} threads, {} concurrent",
    total,
    threads,
    params.parallel_games
  );

  let (handle_old, requests_old) = batch_model::<f32>(model_old.predicts_uncertainty());
  let (handle_new, requests_new) = batch_model::<f32>(model_new.predicts_uncertainty());
  let (results, results_receiver) = futures::channel::mpsc::unbounded();

  std::thread::scope(|scope| -> Result<ExitCode> {
    let workers = (0..threads)
      .map(|i| {
        let mut shard_rng = SmallRng::from_seed(rng.random());
        let parallel = split(params.parallel_games, threads, i);
        let source_old = handle_old.source();
        let source_new = handle_new.source();
        let results = results.clone();
        let pairs = &pairs;
        let zobrist = &zobrist;
        let next_game = &next_game;
        let decided = &decided;
        scope.spawn(move || {
          let result = futures::executor::block_on(pit_games(
            pairs,
            zobrist,
            next_game,
            parallel,
            &source_old,
            &source_new,
            &mut shard_rng,
            stop,
            &results,
          ));
          if result.is_err() {
            decided.store(true, std::sync::atomic::Ordering::Relaxed);
          }
          result
        })
      })
      .collect::<Vec<_>>();
    // Every game's handles and sender descend from a worker's, so once the
    // workers are gone the channels close and the evaluators return. Each
    // evaluator's target is every game it can be waiting for, as the two models
    // taking turns already keeps the device busy while one side selects.
    drop(handle_old);
    drop(handle_new);
    drop(results);

    let (outcome, evaluator_old, evaluator_new) = futures::executor::block_on(async {
      futures::join!(
        async {
          let outcome = pit_results(&params, results_receiver).await;
          decided.store(true, std::sync::atomic::Ordering::Relaxed);
          outcome
        },
        run_evaluator(
          &model_old,
          requests_old,
          params.parallel_games,
          params.pass_overhead,
          params.in_flight_passes
        ),
        run_evaluator(
          &model_new,
          requests_new,
          params.parallel_games,
          params.pass_overhead,
          params.in_flight_passes
        ),
      )
    });
    let games_results = workers
      .into_iter()
      .map(|worker| worker.join().map_err(|_| anyhow::anyhow!("a pit thread panicked")))
      .collect::<Vec<_>>();
    evaluator_old?;
    evaluator_new?;
    for result in games_results {
      result??;
    }

    Ok(if outcome? { ExitCode::SUCCESS } else { 2.into() })
  })
}

fn count<R: Rng>(params: CountParams, rng: &mut R) -> Result<ExitCode> {
  let mut games = 0u32;
  let mut examples = 0u32;

  for path in params.games {
    for node in read_games(path)? {
      let node = node?;
      let field = from_sgf::<Field, _>(&node, rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
      let visits = sgf_to_visits(&node, field.stride);
      games += 1;
      examples += visits.iter().filter(|v| v.1 > 0.0).count() as u32;
    }
  }

  println!("Games: {games}; examples: {examples}");

  Ok(ExitCode::SUCCESS)
}

/// Recomputes the policy surprise (KL divergence from the model's raw policy
/// prior to the visit-count target) and the raw network value of every searched
/// position of a single game.
async fn recalc_game<N, M, R>(node: SgfNode<Prop>, model: &M, rng: &mut R) -> Result<(ExtendedField, Vec<Visits>, i32)>
where
  N: Float + Sum,
  M: Model<N>,
  M::E: Debug,
  R: Rng,
{
  let field = from_sgf::<ExtendedField, _>(&node, rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
  let stride = field.field().stride;
  let mut visits = sgf_to_visits(&node, stride);
  let komi_x_2 = node
    .properties
    .iter()
    .find_map(|prop| match prop {
      Prop::Unknown(name, values) if name == "KM" => values.first().map(|value| {
        let value = value.parse::<f32>().unwrap();
        (value * 2.0).round() as i32
      }),
      _ => None,
    })
    .unwrap_or(0);

  // Games recorded before value surprise weighting store no search or raw
  // network values (parsed as 0). Without the search values the value
  // target can't be reconstructed, so recalculating the raw value would
  // only manufacture a bogus value surprise - leave such games value-free.
  let has_value_surprise = visits.iter().any(|visits| visits.3 != 0.0 || visits.4 != 0.0);

  let width = field.field().width();
  let height = field.field().height();
  let moves: Vec<_> = field.field().colored_moves().collect();
  // Moves played before the first searched position (e.g. the opening).
  let initial_moves = moves.len() - visits.len();
  let zobrist = Arc::new(Zobrist::new(length(width, height) * 3, rng));

  let mut position_field = Field::new(width, height, zobrist);
  let mut placed = 0;

  // Cheap searches need the surprise too, since it decides whether they earn
  // training weight. The search value is left untouched: a single root
  // expansion cannot reproduce a search's estimate.
  for (i, current) in visits.iter_mut().enumerate() {
    let position = initial_moves + i;
    let player = moves[position].1;
    let komi_x_2 = if player == Player::Red { komi_x_2 } else { -komi_x_2 };

    for &(pos, player) in &moves[placed..position] {
      assert!(position_field.put_point(pos, player));
      position_field.update_grounded();
    }
    placed = position;

    // A single search step expands the root with the network, filling in the
    // raw child priors used to measure the surprise.
    let mut search = Search::<N>::new(Params::SELF_PLAY);
    search
      .mcgs(&mut position_field, player, model, komi_x_2, rng)
      .await
      .map_err(|e| anyhow::anyhow!("model failure: {:?}", e))?;
    let mut priors = vec![N::zero(); position_field.length()];
    search.root_priors(&mut priors);
    let target = current
      .0
      .iter()
      .map(|&(pos, weight)| (pos, N::from(weight).unwrap()))
      .collect::<Vec<_>>();
    current.2 = Search::policy_surprise(&target, &priors).to_f64().unwrap();
    if has_value_surprise {
      current.4 = search.raw_winloss().to_f64().unwrap();
    }
  }

  Ok((field, visits, komi_x_2))
}

/// Recalculates every game of `params.games`, up to `params.parallel_games` of
/// them concurrently, creating a fresh model per game with `new_model`.
async fn recalc_games<N, M, MF, R>(
  params: &RecalcParams,
  mut new_model: MF,
  rng: &mut R,
  should_stop: &AtomicBool,
) -> Result<()>
where
  N: Float + Sum,
  M: Model<N>,
  M::E: Debug,
  MF: FnMut() -> M,
  R: Rng,
{
  // All the game files are opened up front, so an unreadable one fails before
  // any recalculation is done rather than hours into it.
  let nodes = params
    .games
    .iter()
    .map(read_games)
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten();

  let games = nodes
    .enumerate()
    .take_while(|&(i, _)| {
      if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
        log::info!("Stopping surprise recalculation after {} games", i);
        false
      } else {
        true
      }
    })
    .map(|(_, node)| {
      let mut rng = SmallRng::from_seed(rng.random());
      let model = new_model();
      async move { recalc_game(node?, &model, &mut rng).await }
    });

  let mut games = futures::stream::iter(games).buffer_unordered(params.parallel_games);

  let mut file = File::options().append(true).create(true).open(&params.games_new)?;
  while let Some(game) = games.next().await {
    let (field, visits, komi_x_2) = game?;
    write_game(&mut file, &field, &visits, komi_x_2)?;
  }

  Ok(())
}

async fn recalc<R: Rng>(
  params: RecalcParams,
  device: Device,
  rng: &mut R,
  should_stop: Arc<AtomicBool>,
) -> Result<ExitCode> {
  let model = BurnModel::load(&device, &params.model_config, params.model.clone())?;
  let predictor = Predictor { model, device };

  // All games share one evaluator: the single position each of them expands at
  // a time is merged into one large forward pass instead of being evaluated on
  // its own, which would leave the device almost entirely idle.
  let (handle, requests) = batch_model::<f32>(predictor.predicts_uncertainty());
  let games = async {
    let result = recalc_games(&params, || handle.clone(), rng, &should_stop).await;
    // Close the channel so the evaluator terminates with the last game.
    drop(handle);
    result
  };
  let (games_result, evaluator_result) = futures::join!(
    games,
    run_evaluator(
      &predictor,
      requests,
      params.parallel_games.div_ceil(2),
      params.pass_overhead,
      params.in_flight_passes
    )
  );
  evaluator_result?;
  games_result?;

  Ok(ExitCode::SUCCESS)
}

fn run(config: Config, action: Action, should_stop: Arc<AtomicBool>) -> Result<ExitCode> {
  let mut device = config.device;
  let evaluates_only = matches!(
    action,
    Action::Play(_) | Action::Pit(_) | Action::Recalc(_) | Action::Bench(_)
  );
  if config.half && evaluates_only {
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F16))?;
  }
  let mut rng = config.seed.map_or_else(make_rng, SmallRng::seed_from_u64);

  match action {
    Action::Init(params) => init(params, device.autodiff()),
    Action::Play(params) => play(params, device, &mut rng, should_stop),
    Action::Train(params) => {
      let device = if params.gradient_checkpointing {
        device.autodiff().gradient_checkpointing()
      } else {
        device.autodiff()
      };
      train(params, device, &mut rng, should_stop)
    }
    Action::Pit(params) => pit(params, device, &mut rng, should_stop),
    Action::Count(params) => count(params, &mut rng),
    Action::Recalc(params) => futures::executor::block_on(recalc(params, device, &mut rng, should_stop)),
    Action::Bench(params) => bench(params, device, &mut rng),
    Action::Devices => {
      list_devices();
      Ok(ExitCode::SUCCESS)
    }
  }
}

/// Measures how many positions per second the device evaluates when it is kept
/// busy, which is the ceiling for self-play: comparing it with the throughput
/// the evaluator logs during self-play tells how much of the device the search
/// actually uses.
fn bench<R: Rng>(params: BenchParams, device: Device, rng: &mut R) -> Result<ExitCode> {
  let model = match params.model {
    Some(path) => BurnModel::load(&device, &params.model_config, path)?,
    None => {
      let mut model = BurnModel::new(&device, &params.model_config);
      model.initialize(&device);
      model
    }
  };
  let predictor = Predictor { model, device };
  let (width, height) = (params.width as usize, params.height as usize);
  let in_flight_passes = params.in_flight_passes.max(1);

  for &batch_size in &params.batch_sizes {
    let inputs = Array4::from_shape_simple_fn((batch_size, CHANNELS, height, width), || rng.random::<f32>());
    let global = Array2::from_shape_simple_fn((batch_size, GLOBAL_FEATURES), || rng.random::<f32>());
    let optimism = Array1::zeros(batch_size);
    // The first passes of a shape are slow - kernels get compiled and tuned for
    // it - so they are run and dropped before measuring.
    let run = |seconds: f64| -> Result<(u64, f64)> {
      futures::executor::block_on(async {
        let start = std::time::Instant::now();
        let mut passes = 0;
        let mut in_flight = futures::stream::FuturesUnordered::new();
        loop {
          while in_flight.len() < in_flight_passes {
            in_flight.push(predictor.predict(inputs.clone(), global.clone(), optimism.clone()));
          }
          if let Some(result) = in_flight.next().await {
            result?;
            passes += 1;
          }
          if start.elapsed().as_secs_f64() >= seconds {
            while let Some(result) = in_flight.next().await {
              result?;
              passes += 1;
            }
            return Ok((passes, start.elapsed().as_secs_f64()));
          }
        }
      })
    };
    run(params.seconds)?;
    let (passes, elapsed) = run(params.seconds)?;
    log::info!(
      "Batch {}: {:.0} positions/s, {:.1} ms per pass",
      batch_size,
      (passes * batch_size as u64) as f64 / elapsed,
      1000.0 * elapsed / passes as f64,
    );
  }

  Ok(ExitCode::SUCCESS)
}

fn main() -> Result<ExitCode> {
  // The kernel compiler of some backends logs the IR of every kernel it compiles
  // at the info level - millions of lines, and time spent writing them.
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info,pliron=warn");
  env_logger::Builder::from_env(env).init();

  let should_stop = Arc::new(AtomicBool::new(false));
  let should_stop_c = should_stop.clone();
  ctrlc::set_handler(move || {
    if should_stop_c.load(std::sync::atomic::Ordering::Relaxed) {
      log::info!("Stopping immediately");
      std::process::exit(1);
    }
    should_stop_c.store(true, std::sync::atomic::Ordering::Relaxed);
  })?;

  let (config, action) = cli_parse();

  run(config, action, should_stop)
}
