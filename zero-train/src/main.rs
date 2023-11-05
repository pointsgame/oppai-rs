#![allow(clippy::too_many_arguments)]

mod config;

use anyhow::Result;
use burn::{
  autodiff::ADBackendDecorator,
  backend::{NdArrayBackend, WgpuBackend},
  module::Module,
  optim::{AdamWConfig, Optimizer},
  record::{DefaultFileRecorder, FullPrecisionSettings, Record, Recorder},
  tensor::backend::{ADBackend, Backend},
};
use config::{cli_parse, Action, Backend as ConfigBackend, Config};
use num_traits::Float;
use oppai_field::{
  any_field::AnyField,
  field::{length, Field},
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
use oppai_zero_burn::model::{Learner, Model as BurnModel};
use rand::{distributions::uniform::SampleUniform, rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use sgf_parse::{serialize, unknown_game::Prop, GameTree, SimpleText};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  fs,
  iter::{self, Sum},
  path::PathBuf,
  process::ExitCode,
  sync::Arc,
};

fn init<B>(config: Config, model_path: PathBuf, optimizer_path: PathBuf) -> Result<ExitCode>
where
  B: ADBackend,
{
  let model = BurnModel::<B>::new(config.width, config.height);
  model.save_file(model_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let optimizer = AdamWConfig::new().init::<B, BurnModel<_>>();
  let record = optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  DefaultFileRecorder::<FullPrecisionSettings>::new().save_item(item, optimizer_path)?;

  Ok(ExitCode::SUCCESS)
}

fn play<B>(config: Config, model_path: PathBuf, game_path: PathBuf) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let model = BurnModel::<B>::new(config.width, config.height);
  let model = model.load_file(model_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let player = Player::Red;

  let mut rng = SmallRng::from_entropy();
  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, &mut rng));
  let mut field = Field::new(config.width, config.height, zobrist);

  for (pos, player) in InitialPosition::Cross.points(config.width, config.height, player) {
    // TODO: random shift
    field.put_point(pos, player);
  }

  let visits = episode(&mut field, player, &model, &mut rng)?;

  let field = field.into();
  if let Some(mut node) = to_sgf(&field) {
    visits_to_sgf(&mut node, &visits, field.field().width(), field.field().moves_count());
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
  config: Config,
  model_path: PathBuf,
  optimizer_path: PathBuf,
  model_new_path: PathBuf,
  optimizer_new_path: PathBuf,
  games_paths: Vec<PathBuf>,
  batch_size: usize,
  epochs: usize,
) -> Result<ExitCode>
where
  B: ADBackend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(config.width, config.height);
  let model = model.load_file(model_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;
  let optimizer = AdamWConfig::new().init::<B, BurnModel<_>>();
  let item = DefaultFileRecorder::<FullPrecisionSettings>::new().load_item(optimizer_path)?;
  let record = Record::from_item::<FullPrecisionSettings>(item);
  let optimizer = optimizer.load_record(record);
  let mut learner = Learner { model, optimizer };

  let mut rng = SmallRng::from_entropy();
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
    let visits = sgf_to_visits(node, field.width());

    examples = examples
      + episode::examples(
        field.width(),
        field.height(),
        field.zobrist_arc(),
        &visits,
        &field.colored_moves().collect::<Vec<_>>(),
      );
  }

  let mut rng = SmallRng::from_entropy();

  for epoch in 0..epochs {
    log::info!("Training {} epoch", epoch);
    examples.shuffle(&mut rng);
    for (inputs, policies, values) in examples.batches(batch_size) {
      learner = learner.train(inputs, policies, values)?;
    }
  }

  learner
    .model
    .save_file(model_new_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let record = learner.optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  DefaultFileRecorder::<FullPrecisionSettings>::new().save_item(item, optimizer_new_path)?;

  Ok(ExitCode::SUCCESS)
}

fn pit<B>(config: Config, model_path: PathBuf, model_new_path: PathBuf) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(config.width, config.height);
  let model = model.load_file(model_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let model_new = BurnModel::<B>::new(config.width, config.height);
  let model_new = model_new.load_file(model_new_path, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let player = Player::Red;

  let mut rng = SmallRng::from_entropy();
  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, &mut rng));
  let field = Field::new(config.width, config.height, zobrist);

  let result = if pit::pit(&field, player, &model_new, &model, &mut rng)? {
    ExitCode::SUCCESS
  } else {
    2.into()
  };

  Ok(result)
}

fn run<B>(config: Config, action: Action) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  match action {
    Action::Init { model, optimizer } => init::<ADBackendDecorator<B>>(config, model, optimizer),
    Action::Play { model, game } => play::<B>(config, model, game),
    Action::Train {
      model,
      optimizer,
      model_new,
      optimizer_new,
      games,
      batch_size,
      epochs,
    } => train::<ADBackendDecorator<B>>(
      config,
      model,
      optimizer,
      model_new,
      optimizer_new,
      games,
      batch_size,
      epochs,
    ),
    Action::Pit { model, model_new } => pit::<B>(config, model, model_new),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (config, action) = cli_parse();

  match config.backend {
    ConfigBackend::Ndarray => run::<NdArrayBackend>(config, action),
    ConfigBackend::Wgpu => run::<WgpuBackend>(config, action),
  }
}
