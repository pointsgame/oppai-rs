use clap::{Arg, Command, crate_authors, crate_description, crate_name, crate_version, value_parser};
use std::path::PathBuf;
use strum::{EnumString, VariantNames};

pub struct InitParams {
  pub model: PathBuf,
  pub optimizer: PathBuf,
}

pub struct PlayParams {
  pub width: u32,
  pub height: u32,
  pub komi_x_2: i32,
  pub model: Option<PathBuf>,
  pub game: PathBuf,
}

pub struct TrainParams {
  pub width: u32,
  pub height: u32,
  pub model: PathBuf,
  pub optimizer: PathBuf,
  pub model_new: PathBuf,
  pub optimizer_new: PathBuf,
  pub games: Vec<PathBuf>,
  pub learning_rate: f64,
  pub batch_size: usize,
  pub epochs: usize,
}

pub struct PitParams {
  pub width: u32,
  pub height: u32,
  pub model: PathBuf,
  pub model_new: PathBuf,
  pub games: Option<PathBuf>,
}

pub enum Action {
  Init(InitParams),
  Play(PlayParams),
  Train(TrainParams),
  Pit(PitParams),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum Backend {
  Wgpu,
  Ndarray,
}

pub struct Config {
  pub backend: Backend,
  pub seed: Option<u64>,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      backend: Backend::Wgpu,
      seed: None,
    }
  }
}

fn width_arg() -> Arg {
  Arg::new("width")
    .long("width")
    .help("Field width")
    .num_args(1)
    .value_parser(value_parser!(u32))
    .default_value("16")
}

fn height_arg() -> Arg {
  Arg::new("height")
    .long("height")
    .help("Field height")
    .num_args(1)
    .value_parser(value_parser!(u32))
    .default_value("16")
}

fn model_arg() -> Arg {
  Arg::new("model")
    .long("model")
    .short('m')
    .help("Model path")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
    .required(true)
}

fn optimizer_arg() -> Arg {
  Arg::new("optimizer")
    .long("optimizer")
    .short('o')
    .help("Optimizer state path")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
    .required(true)
}

fn model_new_arg() -> Arg {
  Arg::new("model-new")
    .long("model-new")
    .short('n')
    .help("Trained model path")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
    .required(true)
}

fn optimizer_new_arg() -> Arg {
  Arg::new("optimizer-new")
    .long("optimizer-new")
    .short('p')
    .help("New optimizer state path")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
    .required(true)
}

pub fn cli_parse() -> (Config, Action) {
  let init = Command::new("init")
    .about("Initialize the neural network")
    .arg(model_arg())
    .arg(optimizer_arg());
  let play = Command::new("play")
    .about("Self-play a single game")
    .arg(width_arg())
    .arg(height_arg())
    .arg(
      Arg::new("komi-x2")
        .long("komi-x2")
        .help("Komi multiplied by 2 (to allow half-integer komi values)")
        .num_args(1)
        .value_parser(value_parser!(i32))
        .default_value("0"),
    )
    .arg(model_arg().required(false))
    .arg(
      Arg::new("game")
        .long("game")
        .short('g')
        .help("Path where to save the played game")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    );
  let train = Command::new("train")
    .about("Train the neural network")
    .arg(width_arg())
    .arg(height_arg())
    .arg(model_arg())
    .arg(optimizer_arg())
    .arg(model_new_arg())
    .arg(optimizer_new_arg())
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Paths to the played games")
        .num_args(1..)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("learning-rate")
        .long("learning-rate")
        .short('l')
        .help("Learning rate")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0.00001"),
    )
    .arg(
      Arg::new("batch-size")
        .long("batch-size")
        .short('b')
        .help("Batch size")
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("512"),
    )
    .arg(
      Arg::new("epochs")
        .long("epochs")
        .short('e')
        .help("Number of epochs to train")
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("1"),
    );
  let pit = Command::new("pit")
    .about("Pit one neural network against another")
    .arg(width_arg())
    .arg(height_arg())
    .arg(model_arg())
    .arg(model_new_arg())
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Path where to save the played games")
        .num_args(1)
        .value_parser(value_parser!(PathBuf)),
    );

  let matches = Command::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .subcommand(init)
    .subcommand(play)
    .subcommand(train)
    .subcommand(pit)
    .subcommand_required(true)
    .arg(
      Arg::new("backend")
        .long("backend")
        .help("Backend to use")
        .num_args(1)
        .value_parser(value_parser!(Backend))
        .default_value("Wgpu"),
    )
    .arg(
      Arg::new("seed")
        .long("seed")
        .help("Random seed")
        .num_args(1)
        .value_parser(value_parser!(u64)),
    )
    .get_matches();

  let backend = matches.get_one("backend").copied().unwrap();
  let seed = matches.get_one("seed").copied();

  let config = Config { backend, seed };

  let action = match matches.subcommand() {
    Some(("init", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      Action::Init(InitParams { model, optimizer })
    }
    Some(("play", matches)) => {
      let width = matches.get_one("width").copied().unwrap();
      let height = matches.get_one("height").copied().unwrap();
      let komi_x_2 = matches.get_one("komi-x2").copied().unwrap();
      let model = matches.get_one("model").cloned();
      let game = matches.get_one("game").cloned().unwrap();
      Action::Play(PlayParams {
        width,
        height,
        komi_x_2,
        model,
        game,
      })
    }
    Some(("train", matches)) => {
      let width = matches.get_one("width").copied().unwrap();
      let height = matches.get_one("height").copied().unwrap();
      let model = matches.get_one("model").cloned().unwrap();
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let optimizer_new = matches.get_one("optimizer-new").cloned().unwrap();
      let games = matches.get_many("games").unwrap().cloned().collect();
      let learning_rate = matches.get_one("learning-rate").cloned().unwrap();
      let batch_size = matches.get_one("batch-size").cloned().unwrap();
      let epochs = matches.get_one("epochs").cloned().unwrap();
      Action::Train(TrainParams {
        width,
        height,
        model,
        optimizer,
        model_new,
        optimizer_new,
        games,
        learning_rate,
        batch_size,
        epochs,
      })
    }
    Some(("pit", matches)) => {
      let width = matches.get_one("width").copied().unwrap();
      let height = matches.get_one("height").copied().unwrap();
      let model = matches.get_one("model").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let games = matches.get_one("games").cloned();
      Action::Pit(PitParams {
        width,
        height,
        model,
        model_new,
        games,
      })
    }
    _ => panic!("no subcommand"),
  };

  (config, action)
}
