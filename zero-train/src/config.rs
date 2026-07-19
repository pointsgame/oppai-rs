use clap::{Arg, Command, crate_authors, crate_description, crate_name, crate_version, value_parser};
use oppai_zero_burn::model::ModelConfig;
use std::path::PathBuf;
use strum::{EnumString, VariantNames};

pub struct InitParams {
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub optimizer: PathBuf,
  pub weight_decay: f32,
}

pub struct PlayParams {
  pub width: Vec<u32>,
  pub height: Vec<u32>,
  pub komi_x_2: Vec<i32>,
  pub model: Option<PathBuf>,
  pub model_config: ModelConfig,
  pub games: PathBuf,
  pub count: usize,
  pub parallel_games: usize,
}

pub struct TrainParams {
  pub width: u32,
  pub height: u32,
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub optimizer: PathBuf,
  pub model_new: PathBuf,
  pub optimizer_new: PathBuf,
  pub games: Vec<PathBuf>,
  pub learning_rate_start: f64,
  pub learning_rate_end: f64,
  pub weight_decay: f32,
  pub gradient_clipping: Option<f32>,
  pub batch_size: usize,
  pub skip: usize,
  pub ignore_surprise: bool,
}

pub struct PitParams {
  pub width: Vec<u32>,
  pub height: Vec<u32>,
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub model_new: PathBuf,
  pub model_config_new: ModelConfig,
  pub games: Option<PathBuf>,
  pub count: u64,
  pub win_rate_threshold: f64,
}

pub struct CountParams {
  pub games: Vec<PathBuf>,
}

pub struct RecalcParams {
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub games: Vec<PathBuf>,
  pub games_new: PathBuf,
}

pub enum Action {
  Init(InitParams),
  Play(PlayParams),
  Train(TrainParams),
  Pit(PitParams),
  Count(CountParams),
  Recalc(RecalcParams),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum Backend {
  #[cfg(feature = "cuda")]
  Cuda,
  #[cfg(feature = "flex")]
  Flex,
  #[cfg(feature = "ndarray")]
  Ndarray,
  #[cfg(feature = "rocm")]
  Rocm,
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  Wgpu,
}

pub struct Config {
  pub backend: Backend,
  pub device_type: u16,
  pub device_id: u16,
  pub seed: Option<u64>,
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

fn model_config_arg() -> Arg {
  Arg::new("model-config")
    .long("model-config")
    .help("Path to a JSON file with the model architecture configuration")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
}

fn model_config_new_arg() -> Arg {
  Arg::new("model-config-new")
    .long("model-config-new")
    .help("Path to a JSON file with the new model architecture configuration")
    .num_args(1)
    .value_parser(value_parser!(PathBuf))
}

fn parse_model_config(matches: &clap::ArgMatches, name: &str) -> ModelConfig {
  matches
    .get_one::<PathBuf>(name)
    .map_or_else(ModelConfig::default, |path| {
      ModelConfig::from_file(path).expect("failed to load the model config file")
    })
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

fn weight_decay_arg() -> Arg {
  Arg::new("weight-decay")
    .long("weight-decay")
    .short('w')
    .help("Weight decay (L2 penalty; 0 disables it)")
    .num_args(1)
    .value_parser(value_parser!(f32))
    .default_value("0.000000004")
}

pub fn cli_parse() -> (Config, Action) {
  let init = Command::new("init")
    .about("Initialize the neural network")
    .arg(model_arg())
    .arg(model_config_arg())
    .arg(optimizer_arg())
    .arg(weight_decay_arg());
  let play = Command::new("play")
    .about("Self-play a single game")
    .arg(width_arg().num_args(1..))
    .arg(height_arg().num_args(1..))
    .arg(
      Arg::new("komi-x2")
        .long("komi-x2")
        .help("Komi multiplied by 2 (to allow half-integer komi values)")
        .num_args(1..)
        .value_parser(value_parser!(i32))
        .default_value("0")
        .allow_hyphen_values(true),
    )
    .arg(model_arg().required(false))
    .arg(model_config_arg())
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Path where to save the played games")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .required(true),
    )
    .arg(
      Arg::new("count")
        .long("count")
        .short('c')
        .help("Number of games to play")
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("1"),
    )
    .arg(
      Arg::new("parallel-games")
        .long("parallel-games")
        .help("How many games to play concurrently, merging their positions into shared forward passes")
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("32"),
    );
  let train = Command::new("train")
    .about("Train the neural network")
    .arg(width_arg())
    .arg(height_arg())
    .arg(model_arg())
    .arg(model_config_arg())
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
      Arg::new("learning-rate-start")
        .long("learning-rate-start")
        .short('l')
        .help("Learning rate at the first batch (kept low to warm up momentum)")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0.00001"),
    )
    .arg(
      Arg::new("learning-rate-end")
        .long("learning-rate-end")
        .short('e')
        .help("Learning rate at the last batch")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0.0001"),
    )
    .arg(weight_decay_arg())
    .arg(
      Arg::new("gradient-clipping")
        .long("gradient-clipping")
        .short('c')
        .help("Clip each parameter's gradient L2 norm to this value")
        .num_args(1)
        .value_parser(value_parser!(f32)),
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
      Arg::new("skip")
        .long("skip")
        .short('s')
        .help("Skip the first N batches")
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("0"),
    )
    .arg(
      Arg::new("ignore-surprise")
        .long("ignore-surprise")
        .help("Ignore policy surprise values when weighting training samples")
        .num_args(0)
        .action(clap::ArgAction::SetTrue),
    );
  let pit = Command::new("pit")
    .about("Pit one neural network against another")
    .arg(width_arg().num_args(1..))
    .arg(height_arg().num_args(1..))
    .arg(model_arg())
    .arg(model_config_arg())
    .arg(model_new_arg())
    .arg(model_config_new_arg())
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Path where to save the played games")
        .num_args(1)
        .value_parser(value_parser!(PathBuf)),
    )
    .arg(
      Arg::new("count")
        .long("count")
        .short('c')
        .help("Number of games to play per side (total games will be twice this)")
        .num_args(1)
        .value_parser(value_parser!(u64))
        .default_value("50"),
    )
    .arg(
      Arg::new("win-rate-threshold")
        .long("win-rate-threshold")
        .short('t')
        .help("Win rate threshold to accept the new model")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0.55"),
    );
  let count = Command::new("count").about("Count games and trainable examples").arg(
    Arg::new("games")
      .long("games")
      .short('g')
      .help("Paths to the played games")
      .num_args(1..)
      .value_parser(value_parser!(PathBuf))
      .required(true),
  );
  let recalc = Command::new("recalc-surprise")
    .about("Recalculate policy surprise and raw network value for games using a model")
    .arg(model_arg())
    .arg(model_config_arg())
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
      Arg::new("games-new")
        .long("games-new")
        .short('n')
        .help("Path where to save the games with recalculated surprise")
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
    .subcommand(count)
    .subcommand(recalc)
    .subcommand_required(true)
    .arg(
      Arg::new("backend")
        .long("backend")
        .help("Backend to use")
        .num_args(1)
        .value_parser(value_parser!(Backend))
        .required(true),
    )
    .arg(
      Arg::new("device-type")
        .long("device-type")
        .help("Device type id used to construct the backend device")
        .num_args(1)
        .value_parser(value_parser!(u16))
        .default_value("0"),
    )
    .arg(
      Arg::new("device-id")
        .long("device-id")
        .help("Device index id used to construct the backend device")
        .num_args(1)
        .value_parser(value_parser!(u16))
        .default_value("0"),
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
  let device_type = matches.get_one("device-type").copied().unwrap();
  let device_id = matches.get_one("device-id").copied().unwrap();
  let seed = matches.get_one("seed").copied();

  let config = Config {
    backend,
    device_type,
    device_id,
    seed,
  };

  let action = match matches.subcommand() {
    Some(("init", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let model_config = parse_model_config(matches, "model-config");
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      let weight_decay = matches.get_one("weight-decay").copied().unwrap();
      Action::Init(InitParams {
        model,
        model_config,
        optimizer,
        weight_decay,
      })
    }
    Some(("play", matches)) => {
      let width = matches.get_many("width").unwrap().copied().collect();
      let height = matches.get_many("height").unwrap().copied().collect();
      let komi_x_2 = matches.get_many("komi-x2").unwrap().copied().collect();
      let model = matches.get_one("model").cloned();
      let model_config = parse_model_config(matches, "model-config");
      let games = matches.get_one("games").cloned().unwrap();
      let count = matches.get_one("count").copied().unwrap();
      let parallel_games = matches.get_one("parallel-games").copied().unwrap();
      Action::Play(PlayParams {
        width,
        height,
        komi_x_2,
        model,
        model_config,
        games,
        count,
        parallel_games,
      })
    }
    Some(("train", matches)) => {
      let width = matches.get_one("width").copied().unwrap();
      let height = matches.get_one("height").copied().unwrap();
      let model = matches.get_one("model").cloned().unwrap();
      let model_config = parse_model_config(matches, "model-config");
      let optimizer = matches.get_one("optimizer").cloned().unwrap();
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let optimizer_new = matches.get_one("optimizer-new").cloned().unwrap();
      let games = matches.get_many("games").unwrap().cloned().collect();
      let learning_rate_start = matches.get_one("learning-rate-start").cloned().unwrap();
      let learning_rate_end = matches.get_one("learning-rate-end").cloned().unwrap();
      let weight_decay = matches.get_one("weight-decay").copied().unwrap();
      let gradient_clipping = matches.get_one::<f32>("gradient-clipping").copied();
      let batch_size = matches.get_one("batch-size").cloned().unwrap();
      let skip = matches.get_one("skip").copied().unwrap();
      let ignore_surprise = matches.get_flag("ignore-surprise");
      Action::Train(TrainParams {
        width,
        height,
        model,
        model_config,
        optimizer,
        model_new,
        optimizer_new,
        games,
        learning_rate_start,
        learning_rate_end,
        weight_decay,
        gradient_clipping,
        batch_size,
        skip,
        ignore_surprise,
      })
    }
    Some(("pit", matches)) => {
      let width = matches.get_many("width").unwrap().copied().collect();
      let height = matches.get_many("height").unwrap().copied().collect();
      let model = matches.get_one("model").cloned().unwrap();
      let model_config = parse_model_config(matches, "model-config");
      let model_new = matches.get_one("model-new").cloned().unwrap();
      let model_config_new = parse_model_config(matches, "model-config-new");
      let games = matches.get_one("games").cloned();
      let count = matches.get_one("count").copied().unwrap();
      let win_rate_threshold = matches.get_one("win-rate-threshold").copied().unwrap();
      Action::Pit(PitParams {
        width,
        height,
        model,
        model_config,
        model_new,
        model_config_new,
        games,
        count,
        win_rate_threshold,
      })
    }
    Some(("count", matches)) => {
      let games = matches.get_many("games").unwrap().cloned().collect();
      Action::Count(CountParams { games })
    }
    Some(("recalc-surprise", matches)) => {
      let model = matches.get_one("model").cloned().unwrap();
      let model_config = parse_model_config(matches, "model-config");
      let games = matches.get_many("games").unwrap().cloned().collect();
      let games_new = matches.get_one("games-new").cloned().unwrap();
      Action::Recalc(RecalcParams {
        model,
        model_config,
        games,
        games_new,
      })
    }
    _ => panic!("no subcommand"),
  };

  (config, action)
}
