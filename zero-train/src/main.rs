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
  grad_clipping::GradientClippingConfig,
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
use num_traits::{Float, ToPrimitive, Zero};
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
  mcgs::Search,
  model::{Model, TrainableModel},
  opening::opening,
  pit,
  random_model::RandomModel,
};
use oppai_zero_burn::model::{Learner, Model as BurnModel, Predictor};
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
  sync::{Arc, atomic::AtomicBool},
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

/// Plays `params.count` games, up to `params.parallel_games` of them
/// concurrently, creating a fresh model per game with `new_model`.
async fn play_games<N, M, MF, R>(
  params: &PlayParams,
  mut new_model: MF,
  rng: &mut R,
  should_stop: &AtomicBool,
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
  let games = (0..params.count)
    .take_while(|&i| {
      if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
        log::info!("Stopping after {} games", i);
        false
      } else {
        true
      }
    })
    .map(|_| {
      let mut rng = SmallRng::from_seed(rng.random());
      let mut model = new_model();
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

        let visits = episode(&mut field, player, &mut model, komi_x_2, &mut rng)
          .await
          .map_err(|e| anyhow::anyhow!("model failure: {:?}", e))?;

        Ok::<_, Error>((field, visits, komi_x_2))
      }
    });

  let mut games = futures::stream::iter(games).buffer_unordered(params.parallel_games);

  let mut file = File::options().append(true).create(true).open(&params.games)?;
  while let Some(game) = games.next().await {
    let (field, visits, komi_x_2) = game?;
    write_game(&mut file, &field.into(), &visits, komi_x_2)?;
  }

  Ok(())
}

async fn play<B, R: Rng>(
  params: PlayParams,
  device: B::Device,
  rng: &mut R,
  should_stop: Arc<AtomicBool>,
) -> Result<ExitCode>
where
  B: Backend,
  FloatElem<B>: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<FloatElem<B>>,
  Exp1: Distribution<FloatElem<B>>,
  Open01: Distribution<FloatElem<B>>,
{
  match params.model.clone() {
    Some(model_path) => {
      let model = BurnModel::<B>::new(&device, &params.model_config);
      let model = model.load_file(
        model_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
      )?;
      let mut predictor = Predictor { model, device };

      // All games share one evaluator: their positions are merged into large
      // forward passes instead of each game evaluating its own tiny batch.
      let (handle, requests) = batch_model::<FloatElem<B>>();
      let games = async {
        let result = play_games(&params, || handle.clone(), rng, &should_stop).await;
        // Close the channel so the evaluator terminates with the last game.
        drop(handle);
        result
      };
      let (games_result, evaluator_result) = futures::join!(games, run_evaluator(&mut predictor, requests));
      evaluator_result?;
      games_result?;
    }
    None => {
      let mut seeder = SmallRng::from_seed(rng.random());
      play_games(
        &params,
        || RandomModel(SmallRng::from_seed(seeder.random())),
        rng,
        &should_stop,
      )
      .await?;
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
  let optimizer = SgdConfig::new()
    .with_weight_decay((params.weight_decay > 0.0).then(|| WeightDecayConfig::new(params.weight_decay)))
    .with_momentum(Some(MomentumConfig::new()))
    .with_gradient_clipping(params.gradient_clipping.map(GradientClippingConfig::Norm))
    .init::<B, BurnModel<_>>();
  let item = Recorder::<B>::load_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &mut params.optimizer.clone(),
  )?;
  let record = Record::from_item::<FullPrecisionSettings>(item, &device);
  let optimizer = optimizer.load_record(record);
  let predictor = Predictor { model, device };
  let mut learner = Learner { predictor, optimizer };

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
      batch.scores,
      batch.captured,
      learning_rate,
    )?;
  }

  learner
    .predictor
    .model
    .save_file(params.model_new, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

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
  let mut model_old = Predictor {
    model: model_old,
    device: device.clone(),
  };

  let model_new = BurnModel::<B>::new(&device, &params.model_config_new);
  let model_new = model_new.load_file(
    params.model_new,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let mut model_new = Predictor {
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
      pit::play(&mut field, player, &mut model_new, &mut model_old, 0, rng).await?
    } else {
      -pit::play(&mut field, player, &mut model_old, &mut model_new, 0, rng).await?
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
      examples += visits.iter().filter(|v| v.1).count() as u32;
    }
  }

  println!("Games: {games}; examples: {examples}");

  Ok(ExitCode::SUCCESS)
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
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let mut model = Predictor { model, device };

  let mut file = File::options().append(true).create(true).open(&params.games_new)?;

  'games: for path in params.games {
    for node in read_games(path)? {
      let node = node?;
      if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
        log::info!("Stopping surprise recalculation");
        break 'games;
      }

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

      let mut position_field = Field::new(width, height, zobrist.clone());
      let mut placed = 0;

      // Recompute the policy surprise (KL divergence from the model's raw policy
      // prior to the visit-count target) and the raw network value for every
      // searched position - cheap searches need the surprise too, since it
      // decides whether they earn training weight. The search value is left
      // untouched: a single root expansion cannot reproduce a search's estimate.
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
        let mut search = Search::<FloatElem<B>>::new(false);
        search
          .mcgs(&mut position_field, player, &mut model, komi_x_2, rng)
          .await?;
        let mut priors = vec![FloatElem::<B>::zero(); position_field.length()];
        search.root_priors(&mut priors);
        current.2 = Search::policy_surprise(&current.0, &priors).to_f64().unwrap();
        if has_value_surprise {
          current.4 = search.raw_value().to_f64().unwrap();
        }
      }

      write_game(&mut file, &field, &visits, komi_x_2)?;
    }
  }

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
    Action::Play(params) => futures::executor::block_on(play::<B, _>(params, device, &mut rng, should_stop)),
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
