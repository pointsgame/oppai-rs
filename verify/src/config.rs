use clap::{Arg, Command, value_parser};

pub struct Config {
  pub worker: String,
  pub worker_args: Vec<String>,
  pub games: u32,
  pub seed: Option<u64>,
}

pub fn cli_parse() -> Config {
  let command = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .arg(
      Arg::new("worker")
        .long("worker")
        .short('w')
        .help("Worker to compare with")
        .num_args(1)
        .required(true),
    )
    .arg(
      Arg::new("worker-args")
        .long("worker-args")
        .short('a')
        .help("Args for the worker, separated by ','")
        .num_args(1..)
        .value_delimiter(','),
    )
    .arg(
      Arg::new("games")
        .long("games")
        .short('g')
        .help("Games count")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("1000000"),
    )
    .arg(
      Arg::new("seed")
        .long("seed")
        .short('s')
        .help("RNG seed")
        .num_args(1)
        .value_parser(value_parser!(u64)),
    );
  let matches = command.get_matches();

  Config {
    worker: matches
      .get_one::<String>("worker")
      .expect("`worker` is required")
      .to_owned(),
    worker_args: matches
      .get_many::<String>("worker-args")
      .map(|args| args.cloned().collect())
      .unwrap_or_default(),
    games: matches.get_one("games").copied().unwrap(),
    seed: matches.get_one("seed").copied(),
  }
}
