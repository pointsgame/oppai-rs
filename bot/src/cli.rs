use crate::config::{Config, Solver};
use clap::{value_t, Arg, ArgGroup, ArgMatches};
use oppai_minimax::minimax::{MinimaxConfig, MinimaxType};
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType};
use strum::VariantNames;

pub fn groups() -> [ArgGroup<'static>; 2] {
  [
    ArgGroup::with_name("Minimax")
      .args(&["minimax-type", "rebuild-trajectories"])
      .multiple(true),
    ArgGroup::with_name("UCT")
      .args(&[
        "radius",
        "uct-depth",
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
  ]
}

pub fn args() -> [Arg<'static, 'static>; 20] {
  [
    Arg::with_name("solver")
      .short("s")
      .long("solver")
      .help("Engine type for position estimation and the best move choosing")
      .takes_value(true)
      .possible_values(Solver::VARIANTS)
      .case_insensitive(true)
      .default_value("Uct"),
    Arg::with_name("time-gap")
      .short("g")
      .long("time-gap")
      .help("Time that is given to IO plus internal delay")
      .takes_value(true)
      .default_value("100ms"),
    Arg::with_name("threads-count")
      .short("t")
      .long("threads-count")
      .help(
        "Number of threads to use. Will be determined automatically if not specified: \
         for Minimax number of physical cores will be chosen, for UCT - number of \
         logical cores",
      )
      .takes_value(true),
    Arg::with_name("hash-table-size")
      .long("hash-table-size")
      .help("Count of elements that hash table for Minimax can contain")
      .takes_value(true)
      .default_value("10000"),
    Arg::with_name("minimax-type")
      .long("minimax-type")
      .help("Minimax type")
      .takes_value(true)
      .possible_values(MinimaxType::VARIANTS)
      .case_insensitive(true)
      .default_value("NegaScout"),
    Arg::with_name("rebuild-trajectories")
      .long("rebuild-trajectories")
      .help(
        "Rebuild trajectories during minimax search. It makes minimax more precise but \
         reduces speed dramatically",
      ),
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
    Arg::with_name("uct-depth")
      .long("uct-depth")
      .help("Maximum depth of the UCT tree")
      .takes_value(true)
      .default_value("8"),
    Arg::with_name("when-create-children")
      .long("when-create-children")
      .help("Child nodes in the UTC tree will be created only after this number of node visits")
      .takes_value(true)
      .default_value("2"),
    Arg::with_name("ucb-type")
      .long("ucb-type")
      .help("Formula of the UCT value")
      .takes_value(true)
      .possible_values(UcbType::VARIANTS)
      .case_insensitive(true)
      .default_value("Ucb1Tuned"),
    Arg::with_name("uctk")
      .long("uctk")
      .help(
        "UCT constant. Larger values give uniform search. Smaller values \
         give very selective search",
      )
      .takes_value(true)
      .default_value("1.0"),
    Arg::with_name("draw-weight")
      .long("draw-weight")
      .help(
        "Draw weight for UCT formula. Should be fractional number between 0 \
         (weight of the defeat) and 1 (weight of the win). Smaller values give \
         more aggressive game",
      )
      .takes_value(true)
      .default_value("0.4"),
    Arg::with_name("red")
      .long("red")
      .help(
        "Red zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be less than green zone",
      )
      .takes_value(true)
      .default_value("0.45"),
    Arg::with_name("green")
      .long("green")
      .help(
        "Green zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be more than red zone.",
      )
      .takes_value(true)
      .default_value("0.5"),
    Arg::with_name("komi-type")
      .long("komi-type")
      .help("Type of komi evaluation for UTC during the game")
      .takes_value(true)
      .possible_values(UctKomiType::VARIANTS)
      .case_insensitive(true)
      .default_value("Dynamic"),
    Arg::with_name("komi-min-iterations")
      .long("komi-min-iterations")
      .help("Dynamic komi for UCT will be updated after this number of iterations")
      .takes_value(true)
      .default_value("3000"),
    Arg::with_name("no-ladders-solver")
      .long("no-ladders-solver")
      .help("Disable ladders solver"),
    Arg::with_name("ladders-score-limit")
      .long("ladders-score-limit")
      .help("Score that a ladder should have to be accepted.")
      .takes_value(true)
      .default_value("0"),
    Arg::with_name("ladders-depth-limit")
      .long("ladders-depth-limit")
      .help("Depth that a ladder should have to be accepted.")
      .takes_value(true)
      .default_value("0"),
    Arg::with_name("ladders-time-limit")
      .long("ladders-time-limit")
      .help("Time limit for ladders solving.")
      .takes_value(true)
      .default_value("1s"),
  ]
}

pub fn parse_config(matches: &ArgMatches<'static>) -> Config {
  let threads_count = if matches.is_present("threads-count") {
    Some(value_t!(matches.value_of("threads-count"), usize).unwrap_or_else(|e| e.exit()))
  } else {
    None
  };
  let uct_config = UctConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get),
    radius: value_t!(matches.value_of("radius"), u32).unwrap_or_else(|e| e.exit()),
    ucb_type: value_t!(matches.value_of("ucb-type"), UcbType).unwrap_or_else(|e| e.exit()),
    draw_weight: value_t!(matches.value_of("draw-weight"), f64).unwrap_or_else(|e| e.exit()),
    uctk: value_t!(matches.value_of("uctk"), f64).unwrap_or_else(|e| e.exit()),
    when_create_children: value_t!(matches.value_of("when-create-children"), usize).unwrap_or_else(|e| e.exit()),
    depth: value_t!(matches.value_of("uct-depth"), u32).unwrap_or_else(|e| e.exit()),
    komi_type: value_t!(matches.value_of("komi-type"), UctKomiType).unwrap_or_else(|e| e.exit()),
    red: value_t!(matches.value_of("red"), f64).unwrap_or_else(|e| e.exit()),
    green: value_t!(matches.value_of("green"), f64).unwrap_or_else(|e| e.exit()),
    komi_min_iterations: value_t!(matches.value_of("komi-min-iterations"), usize).unwrap_or_else(|e| e.exit()),
  };
  let minimax_config = MinimaxConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get_physical),
    minimax_type: value_t!(matches.value_of("minimax-type"), MinimaxType).unwrap_or_else(|e| e.exit()),
    hash_table_size: value_t!(matches.value_of("hash-table-size"), usize).unwrap_or_else(|e| e.exit()),
    rebuild_trajectories: matches.is_present("rebuild-trajectories"),
  };
  Config {
    uct: uct_config,
    minimax: minimax_config,
    time_gap: value_t!(matches.value_of("time-gap"), humantime::Duration)
      .unwrap_or_else(|e| e.exit())
      .into(),
    solver: value_t!(matches.value_of("solver"), Solver).unwrap_or_else(|e| e.exit()),
    ladders: !matches.is_present("no-ladders-solver"),
    ladders_score_limit: value_t!(matches.value_of("ladders-score-limit"), u32).unwrap_or_else(|e| e.exit()),
    ladders_depth_limit: value_t!(matches.value_of("ladders-depth-limit"), u32).unwrap_or_else(|e| e.exit()),
    ladders_time_limit: value_t!(matches.value_of("ladders-time-limit"), humantime::Duration)
      .unwrap_or_else(|e| e.exit())
      .into(),
  }
}
