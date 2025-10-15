use clap::{Arg, Command, crate_authors, crate_description, crate_name, crate_version, value_parser};
use std::path::PathBuf;
use strum::{EnumString, VariantNames};

pub enum Action {
  Init {
    model: PathBuf,
    optimizer: PathBuf,
  },
  Play {
    model: Option<PathBuf>,
    game: PathBuf,
  },
  Train {
    model: PathBuf,
    optimizer: PathBuf,
    model_new: PathBuf,
    optimizer_new: PathBuf,
    games: Vec<PathBuf>,
    batch_size: usize,
    epochs: usize,
  },
  Pit {
    model: PathBuf,
    model_new: PathBuf,
  },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum Backend {
  Wgpu,
  Ndarray,
}

pub struct Config {
  pub width: u32,
  pub height: u32,
  pub backend: Backend,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 20,
      height: 20,
      backend: Backend::Wgpu,
    }
  }
}

pub fn cli_parse() -> (Config, Action) {
  let init = Command::new("init")
    .about("Initialize the neural network")
    .arg(
      Arg::new("model")
        .long("model")
        .short('m')
        .help("Model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("optimizer")
        .long("optimizer")
        .short('o')
        .help("Optimizer state path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    );
  let play = Command::new("play")
    .about("Self-play a single game")
    .arg(
      Arg::new("model")
        .long("model")
        .short('m')
        .help("Model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf)),
    )
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
    .arg(
      Arg::new("model")
        .long("model")
        .short('m')
        .help("Model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("optimizer")
        .long("optimizer")
        .short('o')
        .help("Optimizer state path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("model-new")
        .long("model-new")
        .short('n')
        .help("Trained model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("optimizer-new")
        .long("optimizer-new")
        .short('m')
        .help("New optimizer state path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Paths the played games")
        .num_args(1..)
        .value_parser(value_parser!(PathBuf))
        .required(true),
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
        .default_value("10"),
    );
  let pit = Command::new("pit")
    .about("Pit one neural network against another")
    .arg(
      Arg::new("model")
        .long("model")
        .short('m')
        .help("Model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("model-new")
        .long("model-new")
        .short('n')
        .help("Trained model path")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
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
      Arg::new("width")
        .long("width")
        .help("Field width")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("16"),
    )
    .arg(
      Arg::new("height")
        .long("height")
        .help("Field height")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("16"),
    )
    .arg(
      Arg::new("backend")
        .long("backend")
        .help("Backend to use")
        .num_args(1)
        .value_parser(value_parser!(Backend))
        .default_value("Wgpu"),
    )
    .get_matches();

  let width = matches.get_one("width").copied().unwrap();
  let height = matches.get_one("height").copied().unwrap();
  let backend = matches.get_one("backend").copied().unwrap();

  let config = Config { width, height, backend };

  let action = match matches.subcommand() {
    Some(("init", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      Action::Init { model, optimizer }
    }
    Some(("play", matches)) => {
      let model = matches.get_one("model").cloned();
      let game = matches.get_one("game").cloned().unwrap();
      Action::Play { model, game }
    }
    Some(("train", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let optimizer_new = matches.get_one("optimizer-new").cloned().unwrap();
      let games = matches.get_many("games").unwrap().cloned().collect();
      let batch_size = matches.get_one("batch-size").cloned().unwrap();
      let epochs = matches.get_one("epochs").cloned().unwrap();
      Action::Train {
        model,
        optimizer,
        model_new,
        optimizer_new,
        games,
        batch_size,
        epochs,
      }
    }
    Some(("pit", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      Action::Pit { model, model_new }
    }
    _ => panic!("no subcommand"),
  };

  (config, action)
}
