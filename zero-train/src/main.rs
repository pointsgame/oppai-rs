mod config;

use std::{path::PathBuf, sync::Arc};

use config::cli_parse;
use oppai_field::{
  field::{length, Field},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_zero::self_play::self_play;
use oppai_zero_torch::model::PyModel;
use pyo3::PyResult;
use rand::{rngs::SmallRng, SeedableRng};

fn main() -> PyResult<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let config = cli_parse();
  let player = Player::Red;

  let mut rng = SmallRng::from_entropy();
  let zobrist = Arc::new(Zobrist::new(length(config.width, config.height) * 2, &mut rng));
  let mut field = Field::new(config.width, config.height, zobrist);

  for (pos, player) in InitialPosition::Cross.points(config.width, config.height, player) {
    // TODO: random shift
    field.put_point(pos, player);
  }

  let path = PathBuf::from("model.pt");
  let exists = path.exists();
  if exists {
    log::info!("Loading the model from {}", path.display());
  }
  let model = PyModel::new(path, config.width, config.height, 4)?;
  if exists {
    model.load()?;
  }
  self_play(&field, player, model, &mut rng)
}
