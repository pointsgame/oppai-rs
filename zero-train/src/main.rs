use std::sync::Arc;

use oppai_field::{
  field::{length, Field},
  player::Player,
  zobrist::Zobrist,
};
use oppai_initial::initial::InitialPosition;
use oppai_zero::self_play::self_play;
use oppai_zero_torch::model::PyModel;
use rand::{rngs::SmallRng, SeedableRng};

fn main() {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  pyo3::prepare_freethreaded_python();

  let width = 16;
  let height = 16;
  let player = Player::Red;

  let mut rng = SmallRng::from_entropy();
  let zobrist = Arc::new(Zobrist::new(length(width, height) * 2, &mut rng));
  let mut field = Field::new(width, height, zobrist);

  for (pos, player) in InitialPosition::Cross.points(width, height, player) {
    // TODO: random shift
    field.put_point(pos, player);
  }

  let model = PyModel::new(width, height, 4).unwrap();
  self_play(&field, player, &model, &mut rng).unwrap();
}
