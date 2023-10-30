mod config;

use anyhow::Result;
use config::{cli_parse, Action, Config};
use num_traits::Float;
use numpy::Element;
use oppai_field::{
  any_field::AnyField,
  field::{length, to_x, to_y, Field},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_sgf::{from_coordinate, to_sgf};
use oppai_zero::{
  episode::{episode, examples, Visits},
  examples::Examples,
  field_features::CHANNELS,
  model::TrainableModel,
  pit,
};
use oppai_zero_torch::model::{DType, PyModel};
use pyo3::{types::IntoPyDict, Python};
use rand::{distributions::uniform::SampleUniform, rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use sgf_parse::{serialize, unknown_game::Prop, GameTree, SgfNode};
use std::{
  borrow::Cow,
  fmt::{Debug, Display},
  fs::{self, File},
  iter::{self, Sum},
  path::PathBuf,
  process::ExitCode,
  sync::Arc,
};

fn visits_to_sgf(mut node: &mut SgfNode<Prop>, visits: &[Visits], width: u32, moves_count: usize) {
  for _ in 0..moves_count - visits.len() {
    node = &mut node.children[0];
  }

  for Visits(visits) in visits {
    node = &mut node.children[0];

    node.properties.push(Prop::Unknown(
      "ZR".into(),
      visits
        .iter()
        .map(|&(pos, visits)| {
          format!(
            "{}{}{}",
            from_coordinate(to_x(width, pos) as u8) as char,
            from_coordinate(to_y(width, pos) as u8) as char,
            visits,
          )
        })
        .collect(),
    ));
  }
}

fn init<N: Float + Sum + SampleUniform + DType + Element + Display + Debug>(
  config: Config,
  model_path: PathBuf,
) -> Result<ExitCode> {
  let model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.save(model_path)?;

  Ok(ExitCode::SUCCESS)
}

fn play<N: Float + Sum + SampleUniform + DType + Element + Display + Debug + Serialize>(
  config: Config,
  model_path: PathBuf,
  game_path: PathBuf,
  sgf_path: Option<PathBuf>,
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
  let examples = examples::<N>(
    field.width(),
    field.height(),
    field.zobrist_arc(),
    &visits,
    &field.colored_moves().collect::<Vec<_>>(),
  );

  let field = field.into();
  if let Some(sgf_path) = sgf_path {
    if let Some(mut node) = to_sgf(&field) {
      visits_to_sgf(&mut node, &visits, field.field().width(), field.field().moves_count());
      let sgf = serialize(iter::once(&GameTree::Unknown(node)));
      fs::write(sgf_path, sgf)?;
    }
  }

  let mut file = File::create(game_path)?;
  ciborium::into_writer(&examples, &mut file)?;

  Ok(ExitCode::SUCCESS)
}

fn train<N: Float + Sum + SampleUniform + DType + Element + Display + Debug + for<'de> Deserialize<'de>>(
  config: Config,
  model_path: PathBuf,
  model_new_path: PathBuf,
  games_paths: Vec<PathBuf>,
) -> Result<ExitCode> {
  let mut model = PyModel::<N>::new(config.width, config.height, CHANNELS as u32)?;
  model.load(model_path)?;
  model.to_device(Cow::Owned(config.device))?;

  let mut examples: Examples<N> = Default::default();
  for path in games_paths {
    let mut file = File::open(path)?;
    examples = examples + ciborium::from_reader(&mut file)?;
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

fn run<N: Float + Sum + SampleUniform + DType + Element + Display + Debug + Serialize + for<'de> Deserialize<'de>>(
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
    Action::Play { model, game, sgf } => play::<N>(config, model, game, sgf),
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
