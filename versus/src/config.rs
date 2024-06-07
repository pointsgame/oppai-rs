use clap::{Arg, Command};

pub struct Config {
  pub ai1: String,
  pub ai2: String,
}

pub fn cli_parse() -> Config {
  let matches = Command::new(clap::crate_name!())
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
    .get_matches();

  Config {
    ai1: matches.get_one::<String>("ai1").expect("`ai1` is required").to_owned(),
    ai2: matches.get_one::<String>("ai2").expect("`ai2` is required").to_owned(),
  }
}
