use crate::config::{Config, Solver};
use clap::{Arg, ArgGroup, ArgMatches};
use oppai_minimax::minimax::{MinimaxConfig, MinimaxType};
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType};
use strum::VariantNames;

pub fn groups() -> [ArgGroup<'static>; 2] {
  [
    ArgGroup::new("Minimax")
      .args(&["minimax-type", "rebuild-trajectories"])
      .multiple(true),
    ArgGroup::new("UCT")
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
        "fpu",
      ])
      .multiple(true),
  ]
}

pub fn args() -> [Arg<'static>; 21] {
  [
    Arg::new("solver")
      .short('s')
      .long("solver")
      .help("Engine type for position estimation and the best move choosing")
      .takes_value(true)
      .possible_values(Solver::VARIANTS)
      .ignore_case(true)
      .default_value("Uct"),
    Arg::new("time-gap")
      .short('g')
      .long("time-gap")
      .help("Time that is given to IO plus internal delay")
      .takes_value(true)
      .default_value("100ms"),
    Arg::new("threads-count")
      .short('t')
      .long("threads-count")
      .help(
        "Number of threads to use. Will be determined automatically if not specified: \
         for Minimax number of physical cores will be chosen, for UCT - number of \
         logical cores",
      )
      .takes_value(true),
    Arg::new("hash-table-size")
      .long("hash-table-size")
      .help("Count of elements that hash table for Minimax can contain")
      .takes_value(true)
      .default_value("10000"),
    Arg::new("minimax-type")
      .long("minimax-type")
      .help("Minimax type")
      .takes_value(true)
      .possible_values(MinimaxType::VARIANTS)
      .ignore_case(true)
      .default_value("NegaScout"),
    Arg::new("rebuild-trajectories").long("rebuild-trajectories").help(
      "Rebuild trajectories during minimax search. It makes minimax more precise but \
         reduces speed dramatically",
    ),
    Arg::new("radius")
      .long("radius")
      .help(
        "Radius for points that will be considered by UCT search algorithm. \
         The initial points are fixed once the UCT search algorithm starts. After \
         that, only points that are close enough to staring ones are considered. \
         Points that are more distant to any of the starting points are discarded",
      )
      .takes_value(true)
      .default_value("3"),
    Arg::new("uct-depth")
      .long("uct-depth")
      .help("Maximum depth of the UCT tree")
      .takes_value(true)
      .default_value("8"),
    Arg::new("when-create-children")
      .long("when-create-children")
      .help("Child nodes in the UTC tree will be created only after this number of node visits")
      .takes_value(true)
      .default_value("2"),
    Arg::new("ucb-type")
      .long("ucb-type")
      .help("Formula of the UCT value")
      .takes_value(true)
      .possible_values(UcbType::VARIANTS)
      .ignore_case(true)
      .default_value("Ucb1Tuned"),
    Arg::new("uctk")
      .long("uctk")
      .help(
        "UCT constant. Larger values give uniform search. Smaller values \
         give very selective search",
      )
      .takes_value(true)
      .default_value("1.0"),
    Arg::new("draw-weight")
      .long("draw-weight")
      .help(
        "Draw weight for UCT formula. Should be fractional number between 0 \
         (weight of the defeat) and 1 (weight of the win). Smaller values give \
         more aggressive game",
      )
      .takes_value(true)
      .default_value("0.4"),
    Arg::new("red")
      .long("red")
      .help(
        "Red zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be less than green zone",
      )
      .takes_value(true)
      .default_value("0.45"),
    Arg::new("green")
      .long("green")
      .help(
        "Green zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be more than red zone",
      )
      .takes_value(true)
      .default_value("0.5"),
    Arg::new("komi-type")
      .long("komi-type")
      .help("Type of komi evaluation for UTC during the game")
      .takes_value(true)
      .possible_values(UctKomiType::VARIANTS)
      .ignore_case(true)
      .default_value("Dynamic"),
    Arg::new("komi-min-iterations")
      .long("komi-min-iterations")
      .help("Dynamic komi for UCT will be updated after this number of iterations")
      .takes_value(true)
      .default_value("3000"),
    Arg::new("fpu")
      .long("fpu")
      .help("First-play urgency")
      .takes_value(true)
      .default_value("1.1"),
    Arg::new("no-ladders-solver")
      .long("no-ladders-solver")
      .help("Disable ladders solver"),
    Arg::new("ladders-score-limit")
      .long("ladders-score-limit")
      .help("Score that a ladder should have to be accepted")
      .takes_value(true)
      .default_value("0"),
    Arg::new("ladders-depth-limit")
      .long("ladders-depth-limit")
      .help("Depth that a ladder should have to be accepted")
      .takes_value(true)
      .default_value("0"),
    Arg::new("ladders-time-limit")
      .long("ladders-time-limit")
      .help("Time limit for ladders solving")
      .takes_value(true)
      .default_value("1s"),
  ]
}

pub fn parse_config(matches: &ArgMatches) -> Config {
  let threads_count = if matches.is_present("threads-count") {
    Some(matches.value_of_t("threads-count").unwrap_or_else(|e| e.exit()))
  } else {
    None
  };
  let uct_config = UctConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get),
    radius: matches.value_of_t("radius").unwrap_or_else(|e| e.exit()),
    ucb_type: matches.value_of_t("ucb-type").unwrap_or_else(|e| e.exit()),
    draw_weight: matches.value_of_t("draw-weight").unwrap_or_else(|e| e.exit()),
    uctk: matches.value_of_t("uctk").unwrap_or_else(|e| e.exit()),
    when_create_children: matches.value_of_t("when-create-children").unwrap_or_else(|e| e.exit()),
    depth: matches.value_of_t("uct-depth").unwrap_or_else(|e| e.exit()),
    komi_type: matches.value_of_t("komi-type").unwrap_or_else(|e| e.exit()),
    red: matches.value_of_t("red").unwrap_or_else(|e| e.exit()),
    green: matches.value_of_t("green").unwrap_or_else(|e| e.exit()),
    komi_min_iterations: matches.value_of_t("komi-min-iterations").unwrap_or_else(|e| e.exit()),
    fpu: matches.value_of_t("fpu").unwrap_or_else(|e| e.exit()),
  };
  let minimax_config = MinimaxConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get_physical),
    minimax_type: matches.value_of_t("minimax-type").unwrap_or_else(|e| e.exit()),
    hash_table_size: matches.value_of_t("hash-table-size").unwrap_or_else(|e| e.exit()),
    rebuild_trajectories: matches.is_present("rebuild-trajectories"),
  };
  Config {
    uct: uct_config,
    minimax: minimax_config,
    time_gap: matches
      .value_of_t::<humantime::Duration>("time-gap")
      .unwrap_or_else(|e| e.exit())
      .into(),
    solver: matches.value_of_t("solver").unwrap_or_else(|e| e.exit()),
    ladders: !matches.is_present("no-ladders-solver"),
    ladders_score_limit: matches.value_of_t("ladders-score-limit").unwrap_or_else(|e| e.exit()),
    ladders_depth_limit: matches.value_of_t("ladders-depth-limit").unwrap_or_else(|e| e.exit()),
    ladders_time_limit: matches
      .value_of_t::<humantime::Duration>("ladders-time-limit")
      .unwrap_or_else(|e| e.exit())
      .into(),
  }
}
