mod config;

use anyhow::{Error, Result};
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "flex")]
use burn::backend::Flex;
#[cfg(feature = "ndarray")]
use burn::backend::NdArray;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(any(feature = "vulkan", feature = "webgpu"))]
use burn::backend::Wgpu;
use burn::{
  backend::Autodiff,
  module::Module,
  optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig},
  record::{DefaultFileRecorder, FullPrecisionSettings, Record, Recorder},
  tensor::{
    backend::{AutodiffBackend, Backend, Device, DeviceId},
    ops::FloatElem,
  },
};
use config::{
  Action, Backend as ConfigBackend, Config, CountParams, InitParams, PitParams, PlayParams, RecalcParams, TrainParams,
  cli_parse,
};
use flate2::{Compression, read::MultiGzDecoder, write::GzEncoder};
use futures::StreamExt;
use num_traits::Float;
use oppai_field::{
  any_field::AnyField,
  extended_field::ExtendedField,
  field::{Field, length},
  player::Player,
  zobrist::Zobrist,
};
use oppai_sgf::{from_sgf, to_sgf};
use oppai_zero::{
  batch_model::{batch_model, run_evaluator},
  episode::{Visits, episode},
  examples::Examples,
  mcgs::{Params, Search},
  model::{Model, TrainableModel},
  opening::opening,
  pit,
  random_model::RandomModel,
};
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

fn init<B>(params: InitParams, device: B::Device) -> Result<ExitCode>
where
  B: AutodiffBackend,
{
  let mut model = BurnModel::<B>::new(&device, &params.model_config);
  model.initialize(&device);
  model.save_file(params.model, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let optimizer = SgdConfig::new()
    .with_weight_decay((params.weight_decay > 0.0).then(|| WeightDecayConfig::new(params.weight_decay)))
    .with_momentum(Some(MomentumConfig::new()))
    .init::<B, BurnModel<_>>();
  let record = optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    params.optimizer,
  )?;

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

      let visits = episode(&mut field, player, &model, komi_x_2, &mut rng)
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
fn play<B, R: Rng>(params: PlayParams, device: B::Device, rng: &mut R, should_stop: Arc<AtomicBool>) -> Result<ExitCode>
where
  B: Backend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug + Send,
  StandardNormal: Distribution<FloatElem<B>>,
  Exp1: Distribution<FloatElem<B>>,
  Open01: Distribution<FloatElem<B>>,
{
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
      let model = BurnModel::<B>::new(&device, &params.model_config);
      let model = model.load_file(
        model_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
      )?;
      let predictor = Predictor { model, device };

      // All games share one evaluator: their positions are merged into large
      // forward passes instead of each game evaluating its own tiny batch. Only
      // this thread ever touches the model, so the backend needs nothing of its
      // own to be safe under threads.
      let (handle, requests) = batch_model::<FloatElem<B>>(predictor.predicts_uncertainty());

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
              futures::executor::block_on(play_games(
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

fn train<B, R: Rng>(
  params: TrainParams,
  device: B::Device,
  rng: &mut R,
  should_stop: Arc<AtomicBool>,
) -> Result<ExitCode>
where
  B: AutodiffBackend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device, &params.model_config);
  let model = model.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  // The gradients are clipped by their combined norm before the step instead of
  // by the optimizer, which would bound every parameter's norm on its own.
  let optimizer = SgdConfig::new()
    .with_weight_decay((params.weight_decay > 0.0).then(|| WeightDecayConfig::new(params.weight_decay)))
    .with_momentum(Some(MomentumConfig::new()))
    .init::<B, BurnModel<_>>();
  let item = Recorder::<B>::load_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &mut params.optimizer.clone(),
  )?;
  let record = Record::from_item::<FullPrecisionSettings>(item, &device);
  let optimizer = optimizer.load_record(record);
  // The SWA model is an exponential moving average of the trained weights,
  // updated every `swa_period` batches and saved separately; it's the model to
  // export for self-play while training continues from the raw weights.
  let mut swa_model = params
    .model_swa_new
    .is_some()
    .then(|| {
      params.model_swa.as_ref().map_or_else(
        || Ok(model.clone()),
        |path| {
          BurnModel::<B>::new(&device, &params.model_config).load_file(
            path,
            &DefaultFileRecorder::<FullPrecisionSettings>::new(),
            &device,
          )
        },
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
  };

  let mut examples = Examples::default();
  for path in params.games {
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

  examples.shuffle(rng);
  let batches_count = examples.batches_count(params.batch_size);
  // By default average a snapshot every half-epoch, so a single training run
  // contributes two snapshots to the moving average.
  let swa_period = params.swa_period.unwrap_or((batches_count / 2).max(1));
  let zobrist = Arc::new(Zobrist::new(length(params.width, params.height) * 3, rng));
  for (i, batch) in examples
    .batches(params.width, params.height, zobrist, params.batch_size)
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

  learner
    .predictor
    .model
    .save_file(params.model_new, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  if let (Some(swa), Some(path)) = (swa_model, params.model_swa_new) {
    swa.save_file(path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;
  }

  let record = learner.optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    params.optimizer_new,
  )?;

  Ok(ExitCode::SUCCESS)
}

async fn pit<B, R: Rng>(
  params: PitParams,
  device: B::Device,
  rng: &mut R,
  should_stop: Arc<AtomicBool>,
) -> Result<ExitCode>
where
  B: Backend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug,
{
  let model_old = BurnModel::<B>::new(&device, &params.model_config);
  let model_old = model_old.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let model_old = Predictor {
    model: model_old,
    device: device.clone(),
  };

  let model_new = BurnModel::<B>::new(&device, &params.model_config_new);
  let model_new = model_new.load_file(
    params.model_new,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let model_new = Predictor {
    model: model_new,
    device,
  };

  let mut player = Player::Red;
  let total_games = params.count * 2;

  // Returns the win rate assuming all remaining games go best/worst case.
  // best=true: remaining games are all wins; best=false: remaining games are all losses.
  #[inline]
  fn win_rate_bound(wins: u64, losses: u64, played: u64, total: u64, best: bool) -> f64 {
    let draws = played - wins - losses;
    let remaining = total - played;
    let best_wins = if best { wins + remaining } else { wins };
    (best_wins as f64 + draws as f64 / 2.0) / total as f64
  }

  let zobrist = Arc::new(Zobrist::new(
    length(
      *Iterator::max(params.width.iter()).unwrap(),
      *Iterator::max(params.height.iter()).unwrap(),
    ) * 3,
    rng,
  ));

  let mut width = params.width[rng.random_range(0..params.width.len())];
  let mut height = params.height[rng.random_range(0..params.height.len())];
  let mut field = Field::new(width, height, zobrist.clone());

  let mut op = opening(width, height, rng);
  for &(x, y) in op.iter() {
    let pos = field.to_pos(x, y);
    assert!(field.put_point(pos, player));
    field.update_grounded();
    player = player.next();
  }

  let mut wins = 0u64;
  let mut losses = 0u64;

  let mut i = 0u64;
  let outcome = loop {
    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
      log::info!("Stopping after {} games", i);
      break false;
    }

    let result = if i.is_multiple_of(2) {
      pit::play(&mut field, player, &model_new, &model_old, 0, rng).await?
    } else {
      -pit::play(&mut field, player, &model_old, &model_new, 0, rng).await?
    };

    match result.cmp(&0) {
      Ordering::Less => losses += 1,
      Ordering::Greater => wins += 1,
      Ordering::Equal => {}
    };

    if let Some(ref games) = params.games
      && let Some(node) = to_sgf(&field.into())
    {
      let sgf = serialize(iter::once(&GameTree::Unknown(node)));
      let mut file = File::options().append(true).create(true).open(games).unwrap();
      write_sgf(&mut file, &sgf).unwrap();
    }

    i += 1;

    log::info!("Game {}, result {}/{}/{}", i, wins, i - wins - losses, losses);

    // Check early exit: outcome is already determined regardless of remaining games.
    if win_rate_bound(wins, losses, i, total_games, true) <= params.win_rate_threshold {
      break false;
    }
    if win_rate_bound(wins, losses, i, total_games, false) > params.win_rate_threshold {
      break true;
    }

    if i == total_games {
      // All games played, no early exit triggered; do final evaluation.
      let draws = i - wins - losses;
      let win_rate = (wins as f64 + draws as f64 / 2.0) / total_games as f64;
      break win_rate > params.win_rate_threshold;
    }

    if i.is_multiple_of(2) {
      width = params.width[rng.random_range(0..params.width.len())];
      height = params.height[rng.random_range(0..params.height.len())];
      op = opening(width, height, rng);
    }

    player = Player::Red;
    field = Field::new(width, height, zobrist.clone());
    for &(x, y) in op.iter() {
      let pos = field.to_pos(x, y);
      assert!(field.put_point(pos, player));
      field.update_grounded();
      player = player.next();
    }
  };

  Ok(if outcome { ExitCode::SUCCESS } else { 2.into() })
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

async fn recalc<B, R: Rng>(
  params: RecalcParams,
  device: B::Device,
  rng: &mut R,
  should_stop: Arc<AtomicBool>,
) -> Result<ExitCode>
where
  B: Backend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device, &params.model_config);
  let model = model.load_file(
    params.model.clone(),
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let predictor = Predictor { model, device };

  // All games share one evaluator: the single position each of them expands at
  // a time is merged into one large forward pass instead of being evaluated on
  // its own, which would leave the device almost entirely idle.
  let (handle, requests) = batch_model::<FloatElem<B>>(predictor.predicts_uncertainty());
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
      params.in_flight_passes
    )
  );
  evaluator_result?;
  games_result?;

  Ok(ExitCode::SUCCESS)
}

fn run<B>(config: Config, action: Action, should_stop: Arc<AtomicBool>) -> Result<ExitCode>
where
  B: Backend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<FloatElem<B>>,
  Exp1: Distribution<FloatElem<B>>,
  Open01: Distribution<FloatElem<B>>,
{
  let device = B::Device::from_id(DeviceId::new(config.device_type, config.device_id));
  let mut rng = config.seed.map_or_else(make_rng, SmallRng::seed_from_u64);

  match action {
    Action::Init(params) => init::<Autodiff<B>>(params, device),
    Action::Play(params) => play::<B, _>(params, device, &mut rng, should_stop),
    Action::Train(params) => train::<Autodiff<B>, _>(params, device, &mut rng, should_stop),
    Action::Pit(params) => futures::executor::block_on(pit::<B, _>(params, device, &mut rng, should_stop)),
    Action::Count(params) => count(params, &mut rng),
    Action::Recalc(params) => futures::executor::block_on(recalc::<B, _>(params, device, &mut rng, should_stop)),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
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

  match config.backend {
    #[cfg(feature = "flex")]
    ConfigBackend::Flex => run::<Flex>(config, action, should_stop),
    #[cfg(feature = "ndarray")]
    ConfigBackend::Ndarray => run::<NdArray>(config, action, should_stop),
    #[cfg(any(feature = "vulkan", feature = "webgpu"))]
    ConfigBackend::Wgpu => run::<Wgpu>(config, action, should_stop),
    #[cfg(feature = "cuda")]
    ConfigBackend::Cuda => run::<Cuda>(config, action, should_stop),
    #[cfg(feature = "rocm")]
    ConfigBackend::Rocm => run::<Rocm>(config, action, should_stop),
  }
}
