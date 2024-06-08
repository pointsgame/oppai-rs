#[cfg(feature = "term-render")]
use clap::ArgAction;
use clap::{Arg, Command};

pub struct Config {
  pub ai1: String,
  pub ai2: String,
  pub ai1_args: Vec<String>,
  pub ai2_args: Vec<String>,
  #[cfg(feature = "term-render")]
  pub ascii: bool,
}

pub fn cli_parse() -> Config {
  let command = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .arg(
      Arg::new("ai1")
        .long("ai1")
        .help("First AI to test")
        .num_args(1)
        .required(true),
    )
    .arg(
      Arg::new("ai2")
        .long("ai2")
        .help("Second AI to test")
        .num_args(1)
        .required(true),
    )
    .arg(
      Arg::new("ai1-args")
        .long("ai1-args")
        .help("Args for the first AI, separated by ','")
        .num_args(1..)
        .value_delimiter(','),
    )
    .arg(
      Arg::new("ai2-args")
        .long("ai2-args")
        .help("Args for the first AI, separated by ','")
        .num_args(1..)
        .value_delimiter(','),
    );
  #[cfg(feature = "term-render")]
  let command = command.arg(
    Arg::new("ascii")
      .long("ascii")
      .help("Use ascii graphic for displaying the current field")
      .action(ArgAction::SetTrue),
  );
  let matches = command.get_matches();

  Config {
    ai1: matches.get_one::<String>("ai1").expect("`ai1` is required").to_owned(),
    ai2: matches.get_one::<String>("ai2").expect("`ai2` is required").to_owned(),
    ai1_args: matches
      .get_many::<String>("ai1-args")
      .map(|args| args.cloned().collect())
      .unwrap_or_default(),
    ai2_args: matches
      .get_many::<String>("ai2-args")
      .map(|args| args.cloned().collect())
      .unwrap_or_default(),
    #[cfg(feature = "term-render")]
    ascii: matches.get_flag("ascii"),
  }
}
