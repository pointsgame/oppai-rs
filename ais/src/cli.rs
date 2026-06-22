use crate::oppai::{Config, Solver};
use clap::{Arg, ArgAction, ArgGroup, ArgMatches, value_parser};
use oppai_minimax::minimax::{MinimaxConfig, MinimaxType};
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType};

pub fn groups() -> [ArgGroup; 2] {
  [
    ArgGroup::new("Minimax")
      .args(["minimax-type", "rebuild-trajectories"])
      .multiple(true),
    ArgGroup::new("UCT")
      .args([
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

pub fn args() -> [Arg; 20] {
  [
    Arg::new("solver")
      .short('s')
      .long("solver")
      .help("Engine type for position estimation and the best move choosing")
      .num_args(1)
      .value_parser(value_parser!(Solver))
      .ignore_case(true)
      .default_value("Uct"),
    Arg::new("threads-count")
      .short('t')
      .long("threads-count")
      .help(
        "Number of threads to use. Will be determined automatically if not specified: \
         for Minimax number of physical cores will be chosen, for UCT - number of \
         logical cores",
      )
      .num_args(1)
      .value_parser(value_parser!(usize)),
    Arg::new("hash-table-size")
      .long("hash-table-size")
      .help("Count of elements that hash table for Minimax can contain")
      .num_args(1)
      .value_parser(value_parser!(usize))
      .default_value("10000"),
    Arg::new("minimax-type")
      .long("minimax-type")
      .help("Minimax type")
      .num_args(1)
      .value_parser(value_parser!(MinimaxType))
      .ignore_case(true)
      .default_value("NegaScout"),
    Arg::new("rebuild-trajectories")
      .long("rebuild-trajectories")
      .help(
        "Rebuild trajectories during minimax search. It makes minimax more precise but \
         reduces speed dramatically",
      )
      .action(ArgAction::SetTrue),
    Arg::new("radius")
      .long("radius")
      .help(
        "Radius for points that will be considered by UCT search algorithm. \
         The initial points are fixed once the UCT search algorithm starts. After \
         that, only points that are close enough to staring ones are considered. \
         Points that are more distant to any of the starting points are discarded",
      )
      .num_args(1)
      .value_parser(value_parser!(u32))
      .default_value("3"),
    Arg::new("uct-depth")
      .long("uct-depth")
      .help("Maximum depth of the UCT tree")
      .num_args(1)
      .value_parser(value_parser!(u32))
      .default_value("8"),
    Arg::new("when-create-children")
      .long("when-create-children")
      .help("Child nodes in the UTC tree will be created only after this number of node visits")
      .num_args(1)
      .value_parser(value_parser!(usize))
      .default_value("2"),
    Arg::new("ucb-type")
      .long("ucb-type")
      .help("Formula of the UCT value")
      .num_args(1)
      .value_parser(value_parser!(UcbType))
      .ignore_case(true)
      .default_value("Ucb1Tuned"),
    Arg::new("uctk")
      .long("uctk")
      .help(
        "UCT constant. Larger values give uniform search. Smaller values \
         give very selective search",
      )
      .num_args(1)
      .value_parser(value_parser!(f64))
      .default_value("1.0"),
    Arg::new("draw-weight")
      .long("draw-weight")
      .help(
        "Draw weight for UCT formula. Should be fractional number between 0 \
         (weight of the defeat) and 1 (weight of the win). Smaller values give \
         more aggressive game",
      )
      .num_args(1)
      .value_parser(value_parser!(f64))
      .default_value("0.4"),
    Arg::new("red")
      .long("red")
      .help(
        "Red zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be less than green zone",
      )
      .num_args(1)
      .value_parser(value_parser!(f64))
      .default_value("0.45"),
    Arg::new("green")
      .long("green")
      .help(
        "Green zone for dynamic komi for UCT. Should be fractional number \
         between 0 and 1. Should also be more than red zone",
      )
      .num_args(1)
      .value_parser(value_parser!(f64))
      .default_value("0.5"),
    Arg::new("komi-type")
      .long("komi-type")
      .help("Type of komi evaluation for UTC during the game")
      .num_args(1)
      .value_parser(value_parser!(UctKomiType))
      .ignore_case(true)
      .default_value("Dynamic"),
    Arg::new("komi-min-iterations")
      .long("komi-min-iterations")
      .help("Dynamic komi for UCT will be updated after this number of iterations")
      .num_args(1)
      .value_parser(value_parser!(usize))
      .default_value("3000"),
    Arg::new("fpu")
      .long("fpu")
      .help("First-play urgency")
      .num_args(1)
      .value_parser(value_parser!(f64))
      .default_value("1.1"),
    Arg::new("no-ladders-solver")
      .long("no-ladders-solver")
      .help("Disable ladders solver")
      .action(ArgAction::SetFalse),
    Arg::new("ladders-score-limit")
      .long("ladders-score-limit")
      .help("Score that a ladder should have to be accepted")
      .num_args(1)
      .value_parser(value_parser!(u32))
      .default_value("0"),
    Arg::new("ladders-depth-limit")
      .long("ladders-depth-limit")
      .help("Depth that a ladder should have to be accepted")
      .num_args(1)
      .value_parser(value_parser!(u32))
      .default_value("0"),
    Arg::new("ladders-time-limit")
      .long("ladders-time-limit")
      .help("Time limit for ladders solving")
      .num_args(1)
      .value_parser(value_parser!(humantime::Duration))
      .default_value("1s"),
  ]
}

pub fn parse_config(matches: &ArgMatches) -> Config {
  let threads_count = matches.get_one("threads-count").copied();
  let uct_config = UctConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get),
    radius: matches.get_one("radius").copied().unwrap(),
    ucb_type: matches.get_one("ucb-type").copied().unwrap(),
    draw_weight: matches.get_one("draw-weight").copied().unwrap(),
    uctk: matches.get_one("uctk").copied().unwrap(),
    when_create_children: matches.get_one("when-create-children").copied().unwrap(),
    depth: matches.get_one("uct-depth").copied().unwrap(),
    komi_type: matches.get_one("komi-type").copied().unwrap(),
    red: matches.get_one("red").copied().unwrap(),
    green: matches.get_one("green").copied().unwrap(),
    komi_min_iterations: matches.get_one("komi-min-iterations").copied().unwrap(),
    fpu: matches.get_one("fpu").copied().unwrap(),
  };
  let minimax_config = MinimaxConfig {
    threads_count: threads_count.unwrap_or_else(num_cpus::get_physical),
    minimax_type: matches.get_one("minimax-type").copied().unwrap(),
    hash_table_size: matches.get_one("hash-table-size").copied().unwrap(),
    rebuild_trajectories: matches.get_flag("rebuild-trajectories"),
  };
  Config {
    uct: uct_config,
    minimax: minimax_config,
    solver: matches.get_one("solver").copied().unwrap(),
    ladders: matches.get_flag("no-ladders-solver"),
    ladders_score_limit: matches.get_one("ladders-score-limit").copied().unwrap(),
    ladders_depth_limit: matches.get_one("ladders-depth-limit").copied().unwrap(),
    ladders_time_limit: matches
      .get_one::<humantime::Duration>("ladders-time-limit")
      .copied()
      .unwrap()
      .into(),
  }
}
