use burn::tensor::Device;
use clap::{Arg, Command, crate_authors, crate_description, crate_name, crate_version, value_parser};
use oppai_zero_burn::{
  device::{DEVICE_HELP, parse_device},
  model::ModelConfig,
  optimizer::{Lookahead, OptimizerConfig, OptimizerKind},
};
use std::path::PathBuf;

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
  pub threads: usize,
  pub batch_games: usize,
  pub in_flight_passes: usize,
}

pub struct TrainParams {
  pub width: u32,
  pub height: u32,
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub optimizer: PathBuf,
  pub model_new: PathBuf,
  pub optimizer_new: PathBuf,
  pub model_swa: Option<PathBuf>,
  pub model_swa_new: Option<PathBuf>,
  pub swa_scale: f64,
  pub swa_period: Option<usize>,
  pub games: Vec<PathBuf>,
  pub learning_rate_start: f64,
  pub learning_rate_end: f64,
  pub optimizer_config: OptimizerConfig,
  pub gradient_clipping: Option<f32>,
  pub mixed_precision: bool,
  pub gradient_checkpointing: bool,
  pub batch_size: usize,
  pub skip: usize,
  pub train_examples: Option<usize>,
  pub ignore_surprise: bool,
  pub validation_fraction: f64,
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

pub struct BenchParams {
  pub model: Option<PathBuf>,
  pub model_config: ModelConfig,
  pub width: u32,
  pub height: u32,
  pub batch_sizes: Vec<usize>,
  pub seconds: f64,
  pub in_flight_passes: usize,
}

pub struct CountParams {
  pub games: Vec<PathBuf>,
}

pub struct RecalcParams {
  pub model: PathBuf,
  pub model_config: ModelConfig,
  pub games: Vec<PathBuf>,
  pub games_new: PathBuf,
  pub parallel_games: usize,
  pub in_flight_passes: usize,
}

pub enum Action {
  Init(InitParams),
  Play(PlayParams),
  Train(TrainParams),
  Pit(PitParams),
  Count(CountParams),
  Recalc(RecalcParams),
  Bench(BenchParams),
  Devices,
}

pub struct Config {
  pub device: Device,
  pub seed: Option<u64>,
  pub half: bool,
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

fn parallel_games_arg(help: &'static str) -> Arg {
  Arg::new("parallel-games")
    .long("parallel-games")
    .help(help)
    .num_args(1)
    .value_parser(value_parser!(usize))
    .default_value("32")
}

fn threads_arg() -> Arg {
  Arg::new("threads")
    .long("threads")
    .help(
      "How many OS threads play games; the concurrent games are split between them. Defaults to one per \
       physical core, since the games are CPU bound between forward passes and hyperthreads would only contend \
       for the same execution units",
    )
    .num_args(1)
    .value_parser(value_parser!(usize))
}

fn batch_games_arg() -> Arg {
  Arg::new("batch-games")
    .long("batch-games")
    .help(
      "How many games' positions are enough to dispatch a forward pass, leaving the rest to keep selecting \
       meanwhile. Capped at the games actually in flight, so a value at or above the concurrent games waits for \
       all of them and overlaps nothing",
    )
    .num_args(1)
    .value_parser(value_parser!(usize))
    .default_value("16")
}

fn in_flight_passes_arg() -> Arg {
  Arg::new("in-flight-passes")
    .long("in-flight-passes")
    .help(
      "How many forward passes may be under way at once. The device idles while a finished pass is read back and \
       the next batch is merged, which a second pass in flight covers; more than that hides little and only makes \
       the batches smaller, since a pass dispatched earlier is a pass fewer games make it into",
    )
    .num_args(1)
    .value_parser(value_parser!(usize))
    .default_value("2")
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
    .arg(parallel_games_arg(
      "How many games to play concurrently, merging their positions into shared forward passes",
    ))
    .arg(threads_arg())
    .arg(batch_games_arg())
    .arg(in_flight_passes_arg());
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
      Arg::new("model-swa")
        .long("model-swa")
        .help("Path to the SWA (averaged) model weights to continue from; when omitted the average starts from the loaded model")
        .num_args(1)
        .value_parser(value_parser!(PathBuf)),
    )
    .arg(
      Arg::new("model-swa-new")
        .long("model-swa-new")
        .help("Path where to save the updated SWA model weights; the average is maintained only when this is set. Export this model for self-play and pitting")
        .num_args(1)
        .value_parser(value_parser!(PathBuf)),
    )
    .arg(
      Arg::new("swa-scale")
        .long("swa-scale")
        .help("Number of snapshots averaged in expectation by the SWA exponential moving average")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("8"),
    )
    .arg(
      Arg::new("swa-period")
        .long("swa-period")
        .help("How many batches between SWA snapshots (default: half of the total batch count)")
        .num_args(1)
        .value_parser(value_parser!(usize)),
    )
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
        .help("Learning rate at the first batch, kept low to warm up momentum [default: a tenth of the last one]")
        .num_args(1)
        .value_parser(value_parser!(f64)),
    )
    .arg(
      Arg::new("learning-rate-end")
        .long("learning-rate-end")
        .short('e')
        .help(
          "Learning rate at the last batch [default: 0.0001 for sgd, whose learning rate is that of a loss averaged \
           over the batch and has to grow linearly with the batch size; 0.00004 * sqrt(batch size / 256) for adamw \
           and muon, whose steps do not depend on the scale of the gradient]",
        )
        .num_args(1)
        .value_parser(value_parser!(f64)),
    )
    .arg(
      Arg::new("optimizer-kind")
        .long("optimizer-kind")
        .help(
          "Update rule: sgd - with momentum; adamw - Adam with decoupled weight decay; muon - orthogonalized \
           momentum for the weights of the hidden layers, convolution filters taken as matrices of output channels \
           by input patches, and adamw for every other parameter. Changing it starts the optimizer state afresh",
        )
        .num_args(1)
        .value_parser(OptimizerKind::NAMES)
        .default_value("sgd"),
    )
    .arg(
      Arg::new("weight-decay")
        .long("weight-decay")
        .short('w')
        .help(
          "Weight decay of the weights of the hidden layers, the input layer and the heads; the attention's \
           projections get half of it and the biases, offsets and rotary frequencies nearly none. With sgd it is an \
           L2 penalty, with adamw and muon the weights shrink by learning rate * weight decay every step \
           [default: 0.000000004 for sgd, 0.005 * sqrt(batch size / 256) for adamw and muon]",
        )
        .num_args(1)
        .value_parser(value_parser!(f64)),
    )
    .arg(
      Arg::new("lookahead-steps")
        .long("lookahead-steps")
        .help("Steps of a lookahead cycle: after this many steps, the weights move back toward where the cycle began")
        .num_args(1)
        .value_parser(value_parser!(u64).range(1..))
        .default_value("6"),
    )
    .arg(
      Arg::new("lookahead-alpha")
        .long("lookahead-alpha")
        .help(
          "How far toward where its steps took them the weights move at the end of a lookahead cycle. The steps \
           within a cycle are made this many times longer, so that a cycle covers the same distance. 1 disables \
           lookahead",
        )
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0.5"),
    )
    .arg(
      Arg::new("gradient-clipping")
        .long("gradient-clipping")
        .short('c')
        .help(
          "Clip the L2 norm of the whole gradient - every parameter's gradient taken as one vector - to this value. \
           The loss is a mean over the batch, so a bound stated for a loss summed over the batch has to be divided \
           by the batch size",
        )
        .num_args(1)
        .value_parser(value_parser!(f32)),
    )
    .arg(
      Arg::new("mixed-precision")
        .long("mixed-precision")
        .help(
          "Compute the convolutions and the transformer's projections in brain floating point while keeping the \
           weights, the optimizer and everything else in single precision. They cannot use the matrix units of many \
           GPUs in single precision, so this makes training many times faster",
        )
        .action(clap::ArgAction::SetTrue),
    )
    .arg(
      Arg::new("gradient-checkpointing")
        .long("gradient-checkpointing")
        .help(
          "Recompute the outputs of the cheap operations during the backward pass instead of keeping them from the \
           forward one. Training memory is mostly what the forward pass keeps, and this cuts it by about a quarter \
           for the convolutional network and more for the transformer, whose attention keeps several board area \
           squared sized tensors per layer",
        )
        .action(clap::ArgAction::SetTrue),
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
      Arg::new("train-examples")
        .long("train-examples")
        .help(
          "Train on only this many of the examples, drawn at random, instead of one pass over all of them. Lets \
           the window of games trained on grow without the training growing with it [default: all of them]",
        )
        .num_args(1)
        .value_parser(value_parser!(usize)),
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
    )
    .arg(
      Arg::new("validation-fraction")
        .long("validation-fraction")
        .help(
          "Fraction of the games held out of training to measure the loss on, before and after the training. \
           Which games are held out depends only on the games themselves, so a game stays held out in every \
           training run; 0 trains on all of them",
        )
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("0"),
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
  let devices = Command::new("devices").about("List the compute devices that can be passed to --device");
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
    )
    .arg(parallel_games_arg(
      "How many games to recalculate concurrently, merging their positions into shared forward passes",
    ))
    .arg(in_flight_passes_arg());

  let bench = Command::new("bench")
    .about("Measure forward passes per second on the device, without any search")
    .arg(model_arg().required(false))
    .arg(model_config_arg())
    .arg(width_arg())
    .arg(height_arg())
    .arg(
      Arg::new("batch-sizes")
        .long("batch-sizes")
        .help("Positions per forward pass to measure, space separated")
        .num_args(1..)
        .value_parser(value_parser!(usize))
        .default_values(["32", "64", "128", "256", "512"]),
    )
    .arg(
      Arg::new("seconds")
        .long("seconds")
        .help("How long to measure each batch size, after a warm-up of the same length")
        .num_args(1)
        .value_parser(value_parser!(f64))
        .default_value("10"),
    )
    .arg(in_flight_passes_arg());

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
    .subcommand(bench)
    .subcommand(devices)
    .subcommand_required(true)
    .arg(
      Arg::new("device")
        .long("device")
        .help(DEVICE_HELP)
        .num_args(1)
        .value_parser(parse_device)
        .default_value("auto"),
    )
    .arg(
      Arg::new("seed")
        .long("seed")
        .help("Random seed")
        .num_args(1)
        .value_parser(value_parser!(u64)),
    )
    .arg(
      Arg::new("half")
        .long("half")
        .help(
          "Evaluate the network in half precision when only evaluating it - in play, pit, recalc-surprise and \
           bench. Convolutions in single precision cannot use the matrix units of many GPUs, so this is often \
           many times faster; training always runs in single precision",
        )
        .action(clap::ArgAction::SetTrue),
    )
    .get_matches();

  let device = matches.get_one::<Device>("device").cloned().unwrap();
  let seed = matches.get_one("seed").copied();

  let half = matches.get_flag("half");

  let config = Config { device, seed, half };

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
      let threads = matches
        .get_one("threads")
        .copied()
        .unwrap_or_else(num_cpus::get_physical);
      let batch_games = matches.get_one("batch-games").copied().unwrap();
      let in_flight_passes = matches.get_one("in-flight-passes").copied().unwrap();
      Action::Play(PlayParams {
        width,
        height,
        komi_x_2,
        model,
        model_config,
        games,
        count,
        parallel_games,
        threads,
        batch_games,
        in_flight_passes,
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
      let model_swa = matches.get_one("model-swa").cloned();
      let model_swa_new = matches.get_one("model-swa-new").cloned();
      let swa_scale = matches.get_one("swa-scale").copied().unwrap();
      let swa_period = matches.get_one("swa-period").copied();
      let games = matches.get_many("games").unwrap().cloned().collect();
      let gradient_clipping = matches.get_one::<f32>("gradient-clipping").copied();
      let mixed_precision = matches.get_flag("mixed-precision");
      let gradient_checkpointing = matches.get_flag("gradient-checkpointing");
      let batch_size = matches.get_one("batch-size").cloned().unwrap();
      let kind = matches
        .get_one::<String>("optimizer-kind")
        .unwrap()
        .parse::<OptimizerKind>()
        .unwrap();
      let learning_rate_end = matches
        .get_one("learning-rate-end")
        .copied()
        .unwrap_or_else(|| kind.default_learning_rate(batch_size));
      let learning_rate_start = matches
        .get_one("learning-rate-start")
        .copied()
        .unwrap_or(learning_rate_end / 10.0);
      let alpha = matches.get_one::<f64>("lookahead-alpha").copied().unwrap();
      if !(alpha > 0.0 && alpha <= 1.0) {
        clap::Error::raw(
          clap::error::ErrorKind::ValueValidation,
          "--lookahead-alpha has to be in (0, 1]\n",
        )
        .exit();
      }
      let optimizer_config = OptimizerConfig {
        kind,
        weight_decay: matches
          .get_one("weight-decay")
          .copied()
          .unwrap_or_else(|| kind.default_weight_decay(batch_size)),
        lookahead: (alpha < 1.0).then(|| Lookahead {
          steps: matches.get_one::<u64>("lookahead-steps").copied().unwrap() as usize,
          alpha,
        }),
        batch_size,
      };
      let skip = matches.get_one("skip").copied().unwrap();
      let train_examples = matches.get_one("train-examples").copied();
      let ignore_surprise = matches.get_flag("ignore-surprise");
      let validation_fraction = matches.get_one("validation-fraction").copied().unwrap();
      Action::Train(TrainParams {
        width,
        height,
        model,
        model_config,
        optimizer,
        model_new,
        optimizer_new,
        model_swa,
        model_swa_new,
        swa_scale,
        swa_period,
        games,
        learning_rate_start,
        learning_rate_end,
        optimizer_config,
        gradient_clipping,
        mixed_precision,
        gradient_checkpointing,
        batch_size,
        skip,
        train_examples,
        ignore_surprise,
        validation_fraction,
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
      let parallel_games = matches.get_one("parallel-games").copied().unwrap();
      let in_flight_passes = matches.get_one("in-flight-passes").copied().unwrap();
      Action::Recalc(RecalcParams {
        model,
        model_config,
        games,
        games_new,
        parallel_games,
        in_flight_passes,
      })
    }
    Some(("bench", matches)) => Action::Bench(BenchParams {
      model: matches.get_one("model").cloned(),
      model_config: parse_model_config(matches, "model-config"),
      width: matches.get_one("width").copied().unwrap(),
      height: matches.get_one("height").copied().unwrap(),
      batch_sizes: matches.get_many("batch-sizes").unwrap().copied().collect(),
      seconds: matches.get_one("seconds").copied().unwrap(),
      in_flight_passes: matches.get_one("in-flight-passes").copied().unwrap(),
    }),
    Some(("devices", _)) => Action::Devices,
    _ => panic!("no subcommand"),
  };

  (config, action)
}
