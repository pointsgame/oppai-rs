use burn::tensor::Device;
use clap::{Arg, Command, value_parser};
use oppai_ais::cli::*;
use oppai_ais::oppai::Config as AIConfig;
use oppai_zero_burn::{
  device::{DEVICE_HELP, parse_device},
  model::ModelConfig,
};
use std::time::Duration;

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub ai: AIConfig,
  pub patterns: Vec<String>,
  pub patterns_cache: Option<String>,
  pub model: Option<String>,
  pub model_config: ModelConfig,
  pub device: Device,
  pub uct_iterations: usize,
  pub minimax_depth: u32,
  pub time_gap: Duration,
}

pub fn cli_parse() -> Config {
  let matches = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .groups(groups())
    .args(args())
    .arg(
      Arg::new("patterns-file")
        .short('p')
        .long("patterns-file")
        .help("Patterns file to use")
        .num_args(1..),
    )
    .arg(
      Arg::new("patterns-cache-file")
        .short('c')
        .long("patterns-cache-file")
        .help("Patterns cache file to use")
        .num_args(1),
    )
    .arg(
      Arg::new("model")
        .short('m')
        .long("model")
        .help("Neural network model file to use for the zero solver")
        .num_args(1),
    )
    .arg(
      Arg::new("model-config")
        .long("model-config")
        .help("Path to a JSON file with the model architecture configuration")
        .num_args(1),
    )
    .arg(
      Arg::new("device")
        .long("device")
        .help(DEVICE_HELP)
        .num_args(1)
        .value_parser(parse_device)
        .default_value("auto"),
    )
    .arg(
      Arg::new("minimax-depth")
        .long("minimax-depth")
        .help(
          "The depth of minimax search tree. Used only for move generation with \
           no time limit",
        )
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("12"),
    )
    .arg(
      Arg::new("uct-iterations")
        .long("uct-iterations")
        .help(
          "The number of UCT iterations. Used only for move generation with \
           no time limit",
        )
        .num_args(1)
        .value_parser(value_parser!(usize))
        .default_value("500000"),
    )
    .arg(
      Arg::new("time-gap")
        .short('g')
        .long("time-gap")
        .help("Time that is given to IO plus internal delay")
        .num_args(1)
        .value_parser(value_parser!(humantime::Duration))
        .default_value("100ms"),
    )
    .get_matches();
  Config {
    ai: parse_config(&matches),
    patterns: matches
      .get_many("patterns-file")
      .map_or_else(Vec::new, |patterns| patterns.cloned().collect()),
    patterns_cache: matches.get_one("patterns-cache-file").cloned(),
    model: matches.get_one("model").cloned(),
    model_config: matches
      .get_one::<String>("model-config")
      .map_or_else(ModelConfig::default, |path| {
        ModelConfig::from_file(path).expect("Failed to load the model config file.")
      }),
    device: matches.get_one::<Device>("device").cloned().unwrap(),
    uct_iterations: matches.get_one("uct-iterations").copied().unwrap(),
    minimax_depth: matches.get_one("minimax-depth").copied().unwrap(),
    time_gap: matches
      .get_one::<humantime::Duration>("time-gap")
      .copied()
      .unwrap()
      .into(),
  }
}
