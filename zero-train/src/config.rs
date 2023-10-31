use clap::{crate_authors, crate_description, crate_name, crate_version, value_parser, Arg, ArgAction, Command};
use std::path::PathBuf;

pub enum Action {
  Init {
    model: PathBuf,
  },
  Play {
    model: PathBuf,
    game: PathBuf,
  },
  Train {
    model: PathBuf,
    model_new: PathBuf,
    games: Vec<PathBuf>,
  },
  Pit {
    model: PathBuf,
    model_new: PathBuf,
  },
}

pub struct Config {
  pub width: u32,
  pub height: u32,
  pub device: String,
  pub library: Option<String>,
  pub double: bool,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 20,
      height: 20,
      device: "cpu".to_string(),
      library: None,
      double: false,
    }
  }
}

pub fn cli_parse() -> (Config, Action) {
  let init = Command::new("init").about("Initialize the neural network").arg(
    Arg::new("model")
      .long("model")
      .short('m')
      .help("Model path")
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
        .value_parser(value_parser!(PathBuf))
        .required(true),
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
      Arg::new("model-new")
        .long("model-new")
        .short('n')
        .help("Trained model path")
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
      Arg::new("device")
        .long("device")
        .help("Device to run pytorch network")
        .num_args(1)
        .default_value("cpu"),
    )
    .arg(
      Arg::new("library")
        .long("library")
        .help("Load pytorch dynamic library")
        .num_args(1),
    )
    .arg(
      Arg::new("double")
        .long("double")
        .help("Use double precision type (float64) for calculations")
        .action(ArgAction::SetTrue),
    )
    .get_matches();

  let width = matches.get_one("width").copied().unwrap();
  let height = matches.get_one("height").copied().unwrap();
  let device = matches.get_one("device").cloned().unwrap();
  let library = matches.get_one("library").cloned();
  let double = matches.get_flag("double");

  let config = Config {
    width,
    height,
    device,
    library,
    double,
  };

  let action = match matches.subcommand() {
    Some(("init", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      Action::Init { model }
    }
    Some(("play", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let game = matches.get_one("game").cloned().unwrap();
      Action::Play { model, game }
    }
    Some(("train", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let games = matches.get_many("games").unwrap().cloned().collect();
      Action::Train {
        model,
        model_new,
        games,
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
