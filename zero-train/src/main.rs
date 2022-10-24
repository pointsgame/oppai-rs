mod config;

use std::{borrow::Cow, path::PathBuf, sync::Arc};

use config::cli_parse;
use oppai_field::{
  field::{length, Field},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_zero::self_play::self_play;
use oppai_zero_torch::model::PyModel;
use pyo3::{types::IntoPyDict, PyResult, Python};
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

  if let Some(library) = config.library {
    Python::with_gil(|py| {
      let locals = [("torch", py.import("torch")?)].into_py_dict(py);
      locals.set_item("library", library)?;

      py.run("torch.ops.load_library(library)", None, Some(locals))
    })?;
  }

  let path = PathBuf::from("model.pt");
  let exists = path.exists();
  if exists {
    log::info!("Loading the model from {}", path.display());
  }
  let mut model = PyModel::new::<f64>(path, config.width, config.height, 4)?;
  if exists {
    model.load()?;
  }
  model.to_device(Cow::Owned(config.device))?;
  self_play::<f64, _, _>(&field, player, model, &mut rng)
}
