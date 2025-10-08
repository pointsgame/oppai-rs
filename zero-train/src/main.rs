#![allow(clippy::too_many_arguments)]

mod config;

use anyhow::Result;
use burn::{
  backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice},
  module::Module,
  optim::{Optimizer, SgdConfig},
  record::{DefaultFileRecorder, FullPrecisionSettings, Record, Recorder},
  tensor::backend::{AutodiffBackend, Backend},
};
use config::{Action, Backend as ConfigBackend, Config, cli_parse};
use num_traits::Float;
use oppai_field::{
  any_field::AnyField,
  field::{Field, length},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_sgf::{
  from_sgf, to_sgf,
  visits::{sgf_to_visits, visits_to_sgf},
};
use oppai_zero::{
  episode::{self, episode},
  examples::Examples,
  model::TrainableModel,
  pit,
};
use oppai_zero_burn::model::{Learner, Model as BurnModel, Predictor};
use rand::{SeedableRng, distr::uniform::SampleUniform, rngs::SmallRng};
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use sgf_parse::{GameTree, SimpleText, serialize, unknown_game::Prop};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  fs,
  iter::{self, Sum},
  path::PathBuf,
  process::ExitCode,
  sync::Arc,
};

fn init<B>(model_path: PathBuf, optimizer_path: PathBuf, device: B::Device) -> Result<ExitCode>
where
  B: AutodiffBackend,
{
  let model = BurnModel::<B>::new(&device);
  model.save_file(model_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let optimizer = SgdConfig::new().init::<B, BurnModel<_>>();
  let record = optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    optimizer_path,
  )?;

  Ok(ExitCode::SUCCESS)
}

fn play<B>(config: Config, model_path: PathBuf, game_path: PathBuf, device: B::Device) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    model_path,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let predictor = Predictor { model, device };

  let player = Player::Red;

  let mut rng = SmallRng::from_os_rng();
  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, &mut rng));
  let mut field = Field::new(config.width, config.height, zobrist);

  for (pos, player) in InitialPosition::Cross.points(config.width, config.height, player) {
    // TODO: random shift
    field.put_point(pos, player);
  }

  let visits = episode(&mut field, player, &predictor, &mut rng)?;

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
    let sgf = serialize(iter::once(&GameTree::Unknown(node)));
    fs::write(game_path, sgf)?;
  }

  Ok(ExitCode::SUCCESS)
}

fn train<B>(
  model_path: PathBuf,
  mut optimizer_path: PathBuf,
  model_new_path: PathBuf,
  optimizer_new_path: PathBuf,
  games_paths: Vec<PathBuf>,
  batch_size: usize,
  epochs: usize,
  device: B::Device,
) -> Result<ExitCode>
where
  B: AutodiffBackend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    model_path,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let optimizer = SgdConfig::new().init::<B, BurnModel<_>>();
  let item = Recorder::<B>::load_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &mut optimizer_path,
  )?;
  let record = Record::from_item::<FullPrecisionSettings>(item, &device);
  let optimizer = optimizer.load_record(record);
  let predictor = Predictor { model, device };
  let mut learner = Learner { predictor, optimizer };

  let mut rng = SmallRng::from_os_rng();
  let mut examples: Examples<<B as Backend>::FloatElem> = Default::default();
  for path in games_paths {
    let sgf = fs::read_to_string(path)?;
    let trees = sgf_parse::parse(&sgf)?;
    let node = trees
      .iter()
      .find_map(|tree| match tree {
        GameTree::Unknown(node) => Some(node),
        GameTree::GoGame(_) => None,
      })
      .ok_or(anyhow::anyhow!("no sgf tree"))?;
    let field = from_sgf::<Field, _>(node, &mut rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
    let visits = sgf_to_visits(node, field.stride);

    examples = examples
      + episode::examples(
        field.width(),
        field.height(),
        field.zobrist_arc(),
        &visits,
        &field.colored_moves().collect::<Vec<_>>(),
      );
  }

  let mut rng = SmallRng::from_os_rng();

  for epoch in 0..epochs {
    log::info!("Training {} epoch", epoch);
    examples.shuffle(&mut rng);
    for (inputs, policies, values) in examples.batches(batch_size) {
      learner = learner.train(inputs, policies, values)?;
    }
  }

  learner
    .predictor
    .model
    .save_file(model_new_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let record = learner.optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    optimizer_new_path,
  )?;

  Ok(ExitCode::SUCCESS)
}

fn pit<B>(config: Config, model_path: PathBuf, model_new_path: PathBuf, device: B::Device) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    model_path,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let predictor = Predictor {
    model,
    device: device.clone(),
  };

  let model_new = BurnModel::<B>::new(&device);
  let model_new = model_new.load_file(
    model_new_path,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let predictor_new = Predictor {
    model: model_new,
    device,
  };

  let player = Player::Red;

  let mut rng = SmallRng::from_os_rng();
  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, &mut rng));
  let field = Field::new(config.width, config.height, zobrist);

  let result = if pit::pit(&field, player, &predictor_new, &predictor, &mut rng)? {
    ExitCode::SUCCESS
  } else {
    2.into()
  };

  Ok(result)
}

fn run<B>(config: Config, action: Action, device: B::Device) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  match action {
    Action::Init { model, optimizer } => init::<Autodiff<B>>(model, optimizer, device),
    Action::Play { model, game } => play::<B>(config, model, game, device),
    Action::Train {
      model,
      optimizer,
      model_new,
      optimizer_new,
      games,
      batch_size,
      epochs,
    } => train::<Autodiff<B>>(
      model,
      optimizer,
      model_new,
      optimizer_new,
      games,
      batch_size,
      epochs,
      device,
    ),
    Action::Pit { model, model_new } => pit::<B>(config, model, model_new, device),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (config, action) = cli_parse();

  match config.backend {
    ConfigBackend::Ndarray => run::<NdArray>(config, action, NdArrayDevice::Cpu),
    ConfigBackend::Wgpu => run::<Wgpu>(config, action, WgpuDevice::DefaultDevice),
  }
}
