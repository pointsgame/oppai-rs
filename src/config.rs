use clap::{App, Arg};
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
  let matches = App::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .groups(&groups())
    .args(&args())
    .arg(
      Arg::with_name("patterns-file")
        .short("p")
        .long("patterns-file")
        .help("Patterns file to use")
        .takes_value(true)
        .multiple(true),
    )
    .arg(
      Arg::with_name("minimax-depth")
        .long("minimax-depth")
        .help(
          "The depth of minimax search tree. Used only for move generation with \
         no time limit",
        )
        .takes_value(true)
        .default_value("12"),
    )
    .arg(
      Arg::with_name("uct-iterations")
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
      clap::values_t!(matches.values_of("patterns-file"), String).unwrap_or_else(|e| e.exit())
    } else {
      Vec::new()
    },
    uct_iterations: clap::value_t!(matches.value_of("uct-iterations"), usize).unwrap_or_else(|e| e.exit()),
    minimax_depth: clap::value_t!(matches.value_of("minimax-depth"), u32).unwrap_or_else(|e| e.exit()),
  }
}
