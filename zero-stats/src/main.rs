//! Reports for watching a training run: statistics of the games played, the
//! losses of the training, the weights of the inputs of the networks and what
//! the networks predict. Every report takes the files it reads as arguments,
//! so it works with any layout of the run's files and any model config.

mod common;
mod games;
mod horizons;
mod losses;
mod net;
mod weights;

use anyhow::Result;
use clap::{Arg, ArgAction, ArgMatches, Command, crate_authors, crate_version, value_parser};
use oppai_zero_burn::device::DEVICE_HELP;
use std::path::PathBuf;

fn paths(matches: &ArgMatches, id: &str) -> Vec<PathBuf> {
  matches.get_many::<PathBuf>(id).into_iter().flatten().cloned().collect()
}

fn cli() -> Command {
  let files = |id: &'static str, help: &'static str| {
    Arg::new(id)
      .help(help)
      .num_args(1..)
      .required(true)
      .value_parser(value_parser!(PathBuf))
  };
  Command::new("oppai-zero-stats")
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about("Statistics of a training run: games, losses, input weights and network predictions")
    .subcommand_required(true)
    .subcommand(
      Command::new("games")
        .about("Outcomes and lengths of games, how fast decided games end and what drawn games spend their moves on")
        .arg(files("files", "Games files, gzip-compressed or not, one row each"))
        .arg(
          Arg::new("count")
            .long("count")
            .short('c')
            .help("Read at most this many games of each file")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("by-size")
            .long("by-size")
            .help("Also split each file by board size")
            .action(ArgAction::SetTrue),
        ),
    )
    .subcommand(
      Command::new("horizons")
        .about("How far ahead play stays connected to a position, by board size: for picking the horizons of targets")
        .arg(files("files", "Games files, gzip-compressed or not, pooled together"))
        .arg(
          Arg::new("count")
            .long("count")
            .short('c')
            .help("Read at most this many games of each file")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("bucket")
            .long("bucket")
            .help("Width of the board size buckets, in the square root of the area")
            .default_value("4")
            .value_parser(value_parser!(u32).range(1..)),
        )
        .arg(
          Arg::new("recent")
            .long("recent")
            .help("Moves before a position that make its recent play")
            .default_value("4")
            .value_parser(value_parser!(u64).range(1..)),
        )
        .arg(
          Arg::new("radius")
            .long("radius")
            .help("Chebyshev distance from the recent play within which a move counts as near it")
            .default_value("2")
            .value_parser(value_parser!(u32)),
        ),
    )
    .subcommand(
      Command::new("losses")
        .about("Mean losses of every training run of a training log")
        .arg(
          Arg::new("log")
            .help("The log of the training")
            .required(true)
            .value_parser(value_parser!(PathBuf)),
        )
        .arg(Arg::new("components").long("components").help(format!(
          "Comma-separated loss components, or `all` [default: {}]",
          losses::DEFAULT_COMPONENTS
        )))
        .arg(
          Arg::new("last")
            .long("last")
            .short('n')
            .help("Show only the last runs")
            .default_value("10")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("gap")
            .long("gap")
            .help("Minutes without a line of the log after which a new run starts")
            .default_value("10")
            .value_parser(value_parser!(f64)),
        ),
    )
    .subcommand(
      Command::new("weights")
        .about("The size of the weights each input of the networks feeds; no GPU needed")
        .arg(files("models", "Model files"))
        .arg(
          Arg::new("channels")
            .long("channels")
            .help("Show every input plane, not only their groups")
            .action(ArgAction::SetTrue),
        ),
    )
    .subcommand(
      Command::new("net")
        .about("Network predictions on the positions of recorded games: calibration, P(draw), reliance on inputs")
        .arg(files("models", "Model files, all of the config given"))
        .arg(
          Arg::new("config")
            .long("config")
            .help("Model config")
            .required(true)
            .value_parser(value_parser!(PathBuf)),
        )
        .arg(
          Arg::new("games")
            .long("games")
            .short('g')
            .help("Games file to take the positions from")
            .required(true)
            .value_parser(value_parser!(PathBuf)),
        )
        .arg(Arg::new("device").long("device").short('d').help(DEVICE_HELP).default_value("auto"))
        .arg(
          Arg::new("count")
            .long("count")
            .short('c')
            .help("Games to read")
            .default_value("300")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("from")
            .long("from")
            .help("First move of every game to take positions from")
            .default_value("10")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("stride")
            .long("stride")
            .help("Take every this many positions; the final position is always taken")
            .default_value("2")
            .value_parser(value_parser!(usize)),
        )
        .arg(
          Arg::new("blank")
            .long("blank")
            .short('b')
            .help(format!(
              "Input group to blank, repeatable, with groups joined by `+` blanked together, e.g. `-b tactics -b tactics+komi`: {}",
              net::Blank::NAMES
            ))
            .action(ArgAction::Append),
        ),
    )
}

fn main() -> Result<()> {
  env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
  let matches = cli().get_matches();
  match matches.subcommand() {
    Some(("games", m)) => games::run(
      &paths(m, "files"),
      m.get_one::<usize>("count").copied(),
      m.get_flag("by-size"),
    ),
    Some(("horizons", m)) => horizons::run(horizons::HorizonsParams {
      files: paths(m, "files"),
      count: m.get_one::<usize>("count").copied(),
      bucket: *m.get_one::<u32>("bucket").unwrap(),
      recent: *m.get_one::<u64>("recent").unwrap() as usize,
      radius: *m.get_one::<u32>("radius").unwrap(),
    }),
    Some(("losses", m)) => losses::run(
      m.get_one::<PathBuf>("log").unwrap(),
      m.get_one::<String>("components").map(String::as_str),
      *m.get_one::<usize>("last").unwrap(),
      *m.get_one::<f64>("gap").unwrap(),
    ),
    Some(("weights", m)) => weights::run(&paths(m, "models"), m.get_flag("channels")),
    Some(("net", m)) => {
      let games_path = m.get_one::<PathBuf>("games").unwrap();
      let blanks = m
        .get_many::<String>("blank")
        .into_iter()
        .flatten()
        .map(|group| group.split('+').map(net::Blank::parse).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
      net::run(net::NetParams {
        device: m.get_one::<String>("device").unwrap().clone(),
        config: m.get_one::<PathBuf>("config").unwrap().clone(),
        models: paths(m, "models"),
        games: common::read_games(games_path, Some(*m.get_one::<usize>("count").unwrap()))?,
        games_label: common::label(games_path),
        from: *m.get_one::<usize>("from").unwrap(),
        stride: *m.get_one::<usize>("stride").unwrap(),
        blanks,
      })
    }
    _ => unreachable!(),
  }
}
