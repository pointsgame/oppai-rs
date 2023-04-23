use clap::{value_parser, Arg, Command};
use oppai_bot::cli::*;
use oppai_bot::config::Config as BotConfig;

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub bot: BotConfig,
  pub patterns: Vec<String>,
  pub uct_iterations: usize,
  pub minimax_depth: u32,
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
    .get_matches();
  Config {
    bot: parse_config(&matches),
    patterns: matches
      .get_many("patterns-file")
      .map_or_else(Vec::new, |patterns| patterns.cloned().collect()),
    uct_iterations: matches.get_one("uct-iterations").copied().unwrap(),
    minimax_depth: matches.get_one("minimax-depth").copied().unwrap(),
  }
}
