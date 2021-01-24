use clap::{App, Arg, ArgGroup};
use oppai_bot::config::{Config as BotConfig, Solver};
use oppai_minimax::minimax::{MinimaxConfig, MinimaxType};
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType};
use std::str;
use strum::VariantNames;

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub bot: BotConfig,
  pub patterns: Vec<String>,
}

pub fn cli_parse() -> Config {
  let num_cpus_string = num_cpus::get().to_string();
  let matches = App::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .group(
      ArgGroup::with_name("Minimax")
        .args(&["minimax-type", "rebuild-trajectories"])
        .multiple(true),
    )
    .group(
      ArgGroup::with_name("UCT")
        .args(&[
          "radius",
          "depth",
          "when-create-children",
          "ucb-type",
          "uctk",
          "draw-weight",
          "red",
          "green",
          "komi-type",
          "komi-min-iterations",
        ])
        .multiple(true),
    )
    .arg(
      Arg::with_name("solver")
        .short("s")
        .long("solver")
        .help("Engine type for position estimation and the best move choosing")
        .takes_value(true)
        .possible_values(&Solver::VARIANTS)
        .case_insensitive(true)
        .default_value("Uct"),
    )
    .arg(
      Arg::with_name("time-gap")
        .short("g")
        .long("time-gap")
        .help("Number of milliseconds that is given to IO plus internal delay")
        .takes_value(true)
        .default_value("100"),
    )
    .arg(
      Arg::with_name("threads-count")
        .short("t")
        .long("threads-count")
        .help(
          "Number of threads to use. Best performance is achieved by specifying \
           the number of physical CPU cores on the target computer. Will be determined \
           automatically if not specified, but automatic resolution is prone to errors \
           for multithreaded CPU-s",
        )
        .takes_value(true)
        .default_value(&num_cpus_string),
    )
    .arg(
      Arg::with_name("patterns-file")
        .short("p")
        .long("patterns-file")
        .help("Patterns file to use")
        .takes_value(true)
        .multiple(true),
    )
    .arg(
      Arg::with_name("hash-table-size")
        .long("hash-table-size")
        .help("Count of elements that hash table for Minimax can contain")
        .takes_value(true)
        .default_value("10000"),
    )
    .arg(
      Arg::with_name("minimax-type")
        .long("minimax-type")
        .help("Minimax type")
        .takes_value(true)
        .possible_values(&MinimaxType::VARIANTS)
        .case_insensitive(true)
        .default_value("NegaScout"),
    )
    .arg(
      Arg::with_name("rebuild-trajectories")
        .long("rebuild-trajectories")
        .help(
          "Rebuild trajectories during minimax search. It makes minimax more precise but \
           reduces speed dramatically",
        ),
    )
    .arg(
      Arg::with_name("radius")
        .long("radius")
        .help(
          "Radius for points that will be considered by UCT search algorithm. \
           The initial points are fixed once the UCT search algorithm starts. After \
           that, only points that are close enough to staring ones are considered. \
           Points that are more distant to any of the starting points are discarted",
        )
        .takes_value(true)
        .default_value("3"),
    )
    .arg(
      Arg::with_name("depth")
        .long("depth")
        .help("Maximum depth of the UCT tree")
        .takes_value(true)
        .default_value("8"),
    )
    .arg(
      Arg::with_name("when-create-children")
        .long("when-create-children")
        .help("Child nodes in the UTC tree will be created only after this number of node visits")
        .takes_value(true)
        .default_value("2"),
    )
    .arg(
      Arg::with_name("ucb-type")
        .long("ucb-type")
        .help("Formula of the UCT value")
        .takes_value(true)
        .possible_values(&UcbType::VARIANTS)
        .case_insensitive(true)
        .default_value("Ucb1Tuned"),
    )
    .arg(
      Arg::with_name("uctk")
        .long("uctk")
        .help("UCT constant. Larger values give uniform search. Smaller values give very selective search")
        .takes_value(true)
        .default_value("1.0"),
    )
    .arg(
      Arg::with_name("draw-weight")
        .long("draw-weight")
        .help(
          "Draw weight for UCT formula. Should be fractional number between 0 \
           (weight of the defeat) and 1 (weight of the win). Smaller values give \
           more aggressive game",
        )
        .takes_value(true)
        .default_value("0.4"),
    )
    .arg(
      Arg::with_name("red")
        .long("red")
        .help(
          "Red zone for dynamic komi for UCT. Should be fractional number \
           between 0 and 1. Should also be less than green zone",
        )
        .takes_value(true)
        .default_value("0.45"),
    )
    .arg(
      Arg::with_name("green")
        .long("green")
        .help(
          "Green zone for dynamic komi for UCT. Should be fractional number \
           between 0 and 1. Should also be more than red zone.",
        )
        .takes_value(true)
        .default_value("0.5"),
    )
    .arg(
      Arg::with_name("komi-type")
        .long("komi-type")
        .help("Type of komi evaluation for UTC during the game")
        .takes_value(true)
        .possible_values(&UctKomiType::VARIANTS)
        .case_insensitive(true)
        .default_value("Dynamic"),
    )
    .arg(
      Arg::with_name("komi-min-iterations")
        .long("komi-min-iterations")
        .help("Dynamic komi for UCT will be updated after this number of iterations")
        .takes_value(true)
        .default_value("3000"),
    )
    .get_matches();
  let threads_count = value_t!(matches.value_of("threads-count"), usize).unwrap_or_else(|e| e.exit());
  let uct_config = UctConfig {
    threads_count,
    radius: value_t!(matches.value_of("radius"), u32).unwrap_or_else(|e| e.exit()),
    ucb_type: value_t!(matches.value_of("ucb-type"), UcbType).unwrap_or_else(|e| e.exit()),
    draw_weight: value_t!(matches.value_of("draw-weight"), f64).unwrap_or_else(|e| e.exit()),
    uctk: value_t!(matches.value_of("uctk"), f64).unwrap_or_else(|e| e.exit()),
    when_create_children: value_t!(matches.value_of("when-create-children"), usize).unwrap_or_else(|e| e.exit()),
    depth: value_t!(matches.value_of("depth"), u32).unwrap_or_else(|e| e.exit()),
    komi_type: value_t!(matches.value_of("komi-type"), UctKomiType).unwrap_or_else(|e| e.exit()),
    red: value_t!(matches.value_of("red"), f64).unwrap_or_else(|e| e.exit()),
    green: value_t!(matches.value_of("green"), f64).unwrap_or_else(|e| e.exit()),
    komi_min_iterations: value_t!(matches.value_of("komi-min-iterations"), usize).unwrap_or_else(|e| e.exit()),
  };
  let minimax_config = MinimaxConfig {
    threads_count,
    minimax_type: value_t!(matches.value_of("minimax-type"), MinimaxType).unwrap_or_else(|e| e.exit()),
    hash_table_size: value_t!(matches.value_of("hash-table-size"), usize).unwrap_or_else(|e| e.exit()),
    rebuild_trajectories: matches.is_present("rebuild-trajectories"),
  };
  let bot_config = BotConfig {
    uct: uct_config,
    minimax: minimax_config,
    time_gap: value_t!(matches.value_of("time-gap"), u32).unwrap_or_else(|e| e.exit()),
    solver: value_t!(matches.value_of("solver"), Solver).unwrap_or_else(|e| e.exit()),
  };
  Config {
    bot: bot_config,
    patterns: if matches.is_present("patterns-file") {
      values_t!(matches.values_of("patterns-file"), String).unwrap_or_else(|e| e.exit())
    } else {
      Vec::new()
    },
  }
}
