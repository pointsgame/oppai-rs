use clap::{value_parser, Arg, Command};
use oppai_ais::cli::*;
use oppai_ais::oppai::Config as AIConfig;
use std::time::Duration;

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub ai: AIConfig,
  pub patterns: Vec<String>,
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
    .args(&args())
    .arg(
      Arg::new("patterns-file")
        .short('p')
        .long("patterns-file")
        .help("Patterns file to use")
        .num_args(1..),
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
    uct_iterations: matches.get_one("uct-iterations").copied().unwrap(),
    minimax_depth: matches.get_one("minimax-depth").copied().unwrap(),
    time_gap: matches
      .get_one::<humantime::Duration>("time-gap")
      .copied()
      .unwrap()
      .into(),
  }
}
