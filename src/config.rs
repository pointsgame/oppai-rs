use clap::{App, Arg};
use oppai_bot::cli::*;
use oppai_bot::config::Config as BotConfig;

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub bot: BotConfig,
  pub patterns: Vec<String>,
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
    .get_matches();
  Config {
    bot: parse_config(&matches),
    patterns: if matches.is_present("patterns-file") {
      clap::values_t!(matches.values_of("patterns-file"), String).unwrap_or_else(|e| e.exit())
    } else {
      Vec::new()
    },
  }
}
