mod config;
mod visits_sgf;

use anyhow::Result;
use config::{cli_parse, Action, Config};
use num_traits::Float;
use numpy::Element;
use oppai_field::{
  any_field::AnyField,
  field::{length, Field},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_sgf::{from_sgf, to_sgf};
use oppai_zero::{
  episode::{self, episode},
  examples::Examples,
  field_features::CHANNELS,
  model::TrainableModel,
  pit,
};
use oppai_zero_torch::model::{DType, PyModel};
use pyo3::{types::IntoPyDict, Python};
use rand::{distributions::uniform::SampleUniform, rngs::SmallRng, SeedableRng};
use sgf_parse::{serialize, GameTree};
use std::{
  borrow::Cow,
  fmt::{Debug, Display},
  fs,
  iter::{self, Sum},
  path::PathBuf,
  process::ExitCode,
  sync::Arc,
};
use visits_sgf::{sgf_to_visits, visits_to_sgf};

fn init<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  model_path: PathBuf,
) -> Result<ExitCode> {
  let model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.save(model_path)?;

  Ok(ExitCode::SUCCESS)
}

fn play<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  model_path: PathBuf,
  game_path: PathBuf,
) -> Result<ExitCode> {
  let mut model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.load(model_path)?;
  model.to_device(Cow::Owned(config.device))?;

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
    let sgf = serialize(iter::once(&GameTree::Unknown(node)));
    fs::write(game_path, sgf)?;
  }

  Ok(ExitCode::SUCCESS)
}

fn train<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  model_path: PathBuf,
  model_new_path: PathBuf,
  games_paths: Vec<PathBuf>,
) -> Result<ExitCode> {
  let mut model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.load(model_path)?;
  model.to_device(Cow::Owned(config.device))?;

  let mut rng = SmallRng::from_entropy();
  let mut examples: Examples<N> = Default::default();
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
      + episode::examples::<N>(
        field.width(),
        field.height(),
        field.zobrist_arc(),
        &visits,
        &field.colored_moves().collect::<Vec<_>>(),
      );
  }

  let mut rng = SmallRng::from_entropy();

  for _ in 0..20 {
    examples.shuffle(&mut rng);
    for (inputs, policies, values) in examples.batches(1024) {
      model.train(inputs, policies, values)?;
    }
  }

  model.save(model_new_path)?;

  Ok(ExitCode::SUCCESS)
}

fn pit<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  model_path: PathBuf,
  model_new_path: PathBuf,
) -> Result<ExitCode> {
  let mut model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.load(model_path)?;
  model.to_device(Cow::Owned(config.device.clone()))?;

  let mut model_new = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model_new.load(model_new_path)?;
  model_new.to_device(Cow::Owned(config.device))?;

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

fn run<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  action: Action,
) -> Result<ExitCode> {
  if let Some(ref library) = config.library {
    Python::with_gil(|py| {
      let locals = [("torch", py.import("torch")?)].into_py_dict(py);
      locals.set_item("library", library)?;

      py.run("torch.ops.load_library(library)", None, Some(locals))
    })?;
  }

  match action {
    Action::Init { model } => init::<N>(config, model),
    Action::Play { model, game } => play::<N>(config, model, game),
    Action::Train {
      model,
      model_new,
      games,
    } => train::<N>(config, model, model_new, games),
    Action::Pit { model, model_new } => pit::<N>(config, model, model_new),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (config, action) = cli_parse();

  if config.double {
    run::<f64>(config, action)
  } else {
    run::<f32>(config, action)
  }
}
