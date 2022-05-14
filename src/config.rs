use clap::{Arg, Command};
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
    .groups(&groups())
    .args(&args())
    .arg(
      Arg::new("patterns-file")
        .short('p')
        .long("patterns-file")
        .help("Patterns file to use")
        .takes_value(true)
        .multiple_occurrences(true),
    )
    .arg(
      Arg::new("minimax-depth")
        .long("minimax-depth")
        .help(
          "The depth of minimax search tree. Used only for move generation with \
         no time limit",
        )
        .takes_value(true)
        .default_value("12"),
    )
    .arg(
      Arg::new("uct-iterations")
        .long("uct-iterations")
        .help(
          "The number of UCT iterations. Used only for move generation with \
         no time limit",
        )
        .takes_value(true)
        .default_value("500000"),
    )
    .get_matches();
  Config {
    bot: parse_config(&matches),
    patterns: if matches.is_present("patterns-file") {
      matches.values_of_t("patterns-file").unwrap_or_else(|e| e.exit())
    } else {
      Vec::new()
    },
    uct_iterations: matches.value_of_t("uct-iterations").unwrap_or_else(|e| e.exit()),
    minimax_depth: matches.value_of_t("minimax-depth").unwrap_or_else(|e| e.exit()),
  }
}
