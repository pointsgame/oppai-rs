#![allow(clippy::too_many_arguments)]

mod config;

use anyhow::{Result, anyhow};
use config::{Action, Backend as ConfigBackend, Config, cli_parse};
use either::Either;
use oppai_field::{
  any_field::AnyField,
  field::{Field, length},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_sgf::{to_sgf, visits::visits_to_sgf};
use oppai_zero::{episode::episode, examples::Examples, model::TrainableModel, pit, random_model::RandomModel};
use oppai_zero_torch::model::PyModel;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use sgf_parse::{GameTree, SimpleText, serialize, unknown_game::Prop};
use std::{cmp::Ordering, fs, iter, path::PathBuf, process::ExitCode, sync::Arc};

fn init(model_path: PathBuf, device: &str) -> Result<ExitCode> {
  let mut model = PyModel::<f32>::new(oppai_zero::field_features::CHANNELS as u32, 0.0)?;
  model.to_device(device.to_string().into())?;
  model.save(model_path)?;
  Ok(ExitCode::SUCCESS)
}

fn play<R: Rng>(
  config: Config,
  model_path: Option<PathBuf>,
  game_path: PathBuf,
  device: &str,
  rng: &mut R,
) -> Result<ExitCode> {
  let mut model = match model_path {
    Some(path) => {
      let mut model = PyModel::<f32>::new(oppai_zero::field_features::CHANNELS as u32, 0.0)?;
      model.to_device(device.to_string().into())?;
      model.load(path)?;
      Either::Right(model)
    }
    None => Either::Left(RandomModel(SmallRng::from_seed(rng.random()))),
  };

  let player = Player::Red;

  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, rng));
  let mut field = Field::new(config.width, config.height, zobrist);

  for (pos, player) in InitialPosition::Cross.points(config.width, config.height, player) {
    field.put_point(pos, player);
  }

  let visits = episode(&mut field, player, &mut model, rng)
    .map_err(|e| e.either(|()| anyhow!("random model failed"), |e| anyhow!(e)))?;

  let field_wrapper = field.into();
  if let Some(mut node) = to_sgf(&field_wrapper) {
    visits_to_sgf(
      &mut node,
      &visits,
      field_wrapper.field().stride,
      field_wrapper.field().moves_count(),
    );
    let score = field_wrapper.field().score(Player::Red);
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

fn train<R: Rng>(
  model_path: PathBuf,
  model_new_path: PathBuf,
  games_paths: Vec<PathBuf>,
  lr: f64,
  batch_size: usize,
  epochs: usize,
  device: &str,
  rng: &mut R,
) -> Result<ExitCode> {
  let mut model = PyModel::<f32>::new(oppai_zero::field_features::CHANNELS as u32, lr)?;
  model.to_device(device.to_string().into())?;
  model.load(model_path)?;

  let mut examples: Examples<f32> = Default::default();

  for path in games_paths {
    let content = fs::read_to_string(path)?;
    let trees = sgf_parse::parse(&content)?;
    for node in trees.iter().filter_map(|tree| match tree {
      GameTree::Unknown(node) => Some(node),
      GameTree::GoGame(_) => None,
    }) {
      let field = oppai_sgf::from_sgf::<Field, _>(node, rng).ok_or(anyhow!("invalid sgf"))?;
      let visits = oppai_sgf::visits::sgf_to_visits(node, field.stride);

      examples = examples
        + oppai_zero::episode::examples(
          field.width(),
          field.height(),
          field.zobrist_arc(),
          &visits,
          &field.colored_moves().collect::<Vec<_>>(),
        );
    }
  }

  for epoch in 0..epochs {
    log::info!("Training {} epoch", epoch);
    examples.shuffle(rng);
    for (inputs, policies, values) in examples.batches(batch_size) {
      model = model.train(inputs, policies, values)?;
    }
  }

  model.save(model_new_path)?;

  Ok(ExitCode::SUCCESS)
}

fn pit<R: Rng>(
  config: Config,
  model_path: PathBuf,
  model_new_path: PathBuf,
  device: &str,
  rng: &mut R,
) -> Result<ExitCode> {
  let mut model = PyModel::<f32>::new(oppai_zero::field_features::CHANNELS as u32, 0.0)?;
  model.to_device(device.to_string().into())?;
  model.load(model_path)?;

  let mut model_new = PyModel::<f32>::new(oppai_zero::field_features::CHANNELS as u32, 0.0)?;
  model_new.to_device(device.to_string().into())?;
  model_new.load(model_new_path)?;

  let player = Player::Red;
  let zobrist = Arc::new(Zobrist::new(
    oppai_field::field::length(config.width, config.height) * 2,
    rng,
  ));
  let field = Field::new(config.width, config.height, zobrist);

  // Pit logic: New model (challenger) vs Old model (champion)
  if pit::pit(&field, player, &mut model_new, &mut model, rng)? {
    Ok(ExitCode::SUCCESS)
  } else {
    // Return 2 to indicate challenger failed to beat champion significantly
    Ok(ExitCode::from(2))
  }
}

fn run(config: Config, action: Action, device: &str) -> Result<ExitCode> {
  let mut rng = config.seed.map_or_else(SmallRng::from_os_rng, SmallRng::seed_from_u64);

  match action {
    Action::Init { model } => init(model, device),
    Action::Play { model, game } => play(config, model, game, device, &mut rng),
    Action::Train {
      model,
      model_new,
      games,
      learning_rate,
      batch_size,
      epochs,
    } => train(
      model,
      model_new,
      games,
      learning_rate,
      batch_size,
      epochs,
      device,
      &mut rng,
    ),
    Action::Pit { model, model_new } => pit(config, model, model_new, device, &mut rng),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (config, action) = cli_parse();

  let device = match config.backend {
    ConfigBackend::Ndarray => "cpu",
    ConfigBackend::Wgpu => "cuda",
  };

  run(config, action, device)
}
