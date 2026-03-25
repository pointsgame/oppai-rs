mod config;

use anyhow::{Error, Result};
use burn::{
  backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice},
  module::Module,
  optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig},
  record::{DefaultFileRecorder, FullPrecisionSettings, Record, Recorder},
  tensor::backend::{AutodiffBackend, Backend},
};
use config::{Action, Backend as ConfigBackend, Config, InitParams, PitParams, PlayParams, TrainParams, cli_parse};
use either::Either;
use num_traits::Float;
use oppai_field::{
  any_field::AnyField,
  field::{Field, length},
  player::Player,
  zobrist::Zobrist,
};
use oppai_sgf::{from_sgf, to_sgf};
use oppai_zero::{
  episode::{self, episode},
  examples::Examples,
  model::TrainableModel,
  opening::opening,
  pit,
  random_model::RandomModel,
};
use oppai_zero_burn::model::{Learner, Model as BurnModel, Predictor};
use oppai_zero_sgf::{sgf_to_visits, visits_to_sgf};
use rand::{Rng, RngExt, SeedableRng, distr::uniform::SampleUniform, make_rng, rngs::SmallRng};
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use sgf_parse::{GameTree, SimpleText, serialize, unknown_game::Prop};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  fs::{self, File},
  io::Write,
  iter::{self, Sum},
  process::ExitCode,
  sync::{Arc, atomic::AtomicBool},
};

fn init<B>(params: InitParams, device: B::Device) -> Result<ExitCode>
where
  B: AutodiffBackend,
{
  let model = BurnModel::<B>::new(&device);
  model.save_file(params.model, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let optimizer = SgdConfig::new()
    .with_weight_decay(Some(WeightDecayConfig::new(0.00003)))
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

fn play<B, R: Rng>(params: PlayParams, device: B::Device, rng: &mut R) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let mut model = match params.model {
    Some(model_path) => {
      let model = BurnModel::<B>::new(&device);
      let model = model.load_file(
        model_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
      )?;
      Either::Right(Predictor { model, device })
    }
    None => Either::Left(RandomModel(SmallRng::from_seed(rng.random()))),
  };

  let mut file = File::options().append(true).create(true).open(&params.game)?;

  for _ in 0..params.count {
    let width = params.width[rng.random_range(0..params.width.len())];
    let height = params.height[rng.random_range(0..params.height.len())];
    let komi_x_2 = params.komi_x_2[rng.random_range(0..params.komi_x_2.len())];
    let mut player = Player::Red;
    let mut field = Field::new_from_rng(width, height, rng);

    let op = opening(width, height, rng);
    for (x, y) in op {
      let pos = field.to_pos(x, y);
      assert!(field.put_point(pos, player));
      field.update_grounded();
      player = player.next();
    }

    let visits = episode(&mut field, player, &mut model, komi_x_2, rng)
      .map_err(|e| e.either(|()| anyhow::anyhow!("random model failed"), Error::from))?;

    let field = field.into();
    if let Some(mut node) = to_sgf(&field) {
      visits_to_sgf(&mut node, &visits, field.field().stride, field.field().moves_count());
      let score = field.field().score(Player::Red);
      node.properties.push(Prop::RE(match score.cmp(&0) {
        Ordering::Equal => "0".into(),
        Ordering::Greater => SimpleText {
          text: format!("W+{}", score),
        },
        Ordering::Less => SimpleText {
          text: format!("B+{}", score.abs()),
        },
      }));
      node
        .properties
        .push(Prop::Unknown("KM".into(), vec![(komi_x_2 as f32 / 2.0).to_string()]));
      let sgf = serialize(iter::once(&GameTree::Unknown(node)));
      writeln!(&mut file, "{sgf}")?;
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
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let optimizer = SgdConfig::new()
    .with_weight_decay(Some(WeightDecayConfig::new(0.00003)))
    .with_momentum(Some(MomentumConfig::new()))
    .init::<B, BurnModel<_>>();
  let item = Recorder::<B>::load_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &mut params.optimizer.clone(),
  )?;
  let record = Record::from_item::<FullPrecisionSettings>(item, &device);
  let optimizer = optimizer.load_record(record);
  let predictor = Predictor { model, device };
  let mut learner = Learner {
    predictor,
    optimizer,
    lr: params.learning_rate,
  };

  let mut examples: Examples<<B as Backend>::FloatElem> = Default::default();
  for path in params.games {
    let sgf = fs::read_to_string(path)?;
    let trees = sgf_parse::parse(&sgf)?;
    for node in trees.iter().filter_map(|tree| match tree {
      GameTree::Unknown(node) => Some(node),
      GameTree::GoGame(_) => None,
    }) {
      let field = from_sgf::<Field, _>(node, rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
      let visits = sgf_to_visits(node, field.stride);
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

      examples = examples
        + episode::examples(
          params.width,
          params.height,
          field.width(),
          field.height(),
          komi_x_2,
          field.zobrist_arc(),
          &visits,
          &field.colored_moves().collect::<Vec<_>>(),
        );
    }
  }

  examples.shuffle(rng);
  let batches_count = examples.batches_count(params.batch_size);
  for (i, batch) in examples.batches(params.batch_size).enumerate().skip(params.skip) {
    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
      log::info!("Stopping training after {} batches", i);
      break;
    }
    if i.is_multiple_of(64) {
      log::info!("Batch {} out of {}", i, batches_count);
    }
    learner = learner.train(
      batch.inputs,
      batch.global,
      batch.policies,
      batch.opponent_policies,
      batch.values,
      batch.scores,
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

fn pit<B, R: Rng>(params: PitParams, device: B::Device, rng: &mut R) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model_old = BurnModel::<B>::new(&device);
  let model_old = model_old.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let mut model_old = Predictor {
    model: model_old,
    device: device.clone(),
  };

  let model_new = BurnModel::<B>::new(&device);
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
      *params.width.iter().max().unwrap(),
      *params.height.iter().max().unwrap(),
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
    let result = if i.is_multiple_of(2) {
      pit::play(&mut field, player, &mut model_new, &mut model_old, 0, rng)?
    } else {
      -pit::play(&mut field, player, &mut model_old, &mut model_new, 0, rng)?
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
      writeln!(&mut file, "{sgf}").unwrap();
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

fn run<B>(config: Config, action: Action, device: B::Device, should_stop: Arc<AtomicBool>) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let mut rng = config.seed.map_or_else(make_rng, SmallRng::seed_from_u64);

  match action {
    Action::Init(params) => init::<Autodiff<B>>(params, device),
    Action::Play(params) => play::<B, _>(params, device, &mut rng),
    Action::Train(params) => train::<Autodiff<B>, _>(params, device, &mut rng, should_stop),
    Action::Pit(params) => pit::<B, _>(params, device, &mut rng),
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
    ConfigBackend::Ndarray => run::<NdArray>(config, action, NdArrayDevice::Cpu, should_stop),
    ConfigBackend::Wgpu => run::<Wgpu>(config, action, WgpuDevice::DefaultDevice, should_stop),
  }
}
