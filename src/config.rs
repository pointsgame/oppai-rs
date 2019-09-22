use crate::minimax::{MinimaxConfig, MinimaxMovesSorting, MinimaxType};
use clap::{App, Arg, ArgGroup};
use num_cpus;
use oppai_uct::uct::{UcbType, UctConfig, UctKomiType};
use std::fmt;
use std::str;

const CONFIG_STR: &str = "config";

const UCB_TYPE_VARIANTS: [&'static str; 3] = ["Winrate", "Ucb1", "Ucb1Tuned"];

const UCT_KOMI_TYPE_VARIANTS: [&'static str; 3] = ["None", "Static", "Dynamic"];

struct UcbTypeArg(UcbType);

impl fmt::Display for UcbTypeArg {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self.0 {
      UcbType::Winrate => write!(f, "Winrate"),
      UcbType::Ucb1 => write!(f, "Ucb1"),
      UcbType::Ucb1Tuned => write!(f, "Ucb1Tuned"),
    }
  }
}

impl str::FromStr for UcbTypeArg {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if s.eq_ignore_ascii_case("Winrate") {
      Ok(UcbTypeArg(UcbType::Winrate))
    } else if s.eq_ignore_ascii_case("Ucb1") {
      Ok(UcbTypeArg(UcbType::Ucb1))
    } else if s.eq_ignore_ascii_case("Ucb1Tuned") {
      Ok(UcbTypeArg(UcbType::Ucb1Tuned))
    } else {
      Err(format!("valid values: {}", UCB_TYPE_VARIANTS.join(", ")))
    }
  }
}

struct UctKomiTypeArg(UctKomiType);

impl fmt::Display for UctKomiTypeArg {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self.0 {
      UctKomiType::None => write!(f, "None"),
      UctKomiType::Static => write!(f, "Static"),
      UctKomiType::Dynamic => write!(f, "Dynamic"),
    }
  }
}

impl str::FromStr for UctKomiTypeArg {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if s.eq_ignore_ascii_case("None") {
      Ok(UctKomiTypeArg(UctKomiType::None))
    } else if s.eq_ignore_ascii_case("Static") {
      Ok(UctKomiTypeArg(UctKomiType::Static))
    } else if s.eq_ignore_ascii_case("Dynamic") {
      Ok(UctKomiTypeArg(UctKomiType::Dynamic))
    } else {
      Err(format!("valid values: {}", UCT_KOMI_TYPE_VARIANTS.join(", ")))
    }
  }
}

arg_enum! {
  #[derive(Clone, Copy, PartialEq, Debug)]
  pub enum Solver {
    Uct,
    Minimax,
    Heuristic
  }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Config {
  pub uct: UctConfig,
  pub minimax: MinimaxConfig,
  bot: BotConfig,
}

#[derive(Clone, PartialEq, Debug)]
struct BotConfig {
  time_gap: u32,
  solver: Solver,
}

const DEFAULT_UCT_CONFIG: UctConfig = UctConfig {
  threads_count: 4,
  radius: 3,
  ucb_type: UcbType::Ucb1Tuned,
  draw_weight: 0.4,
  uctk: 1.0,
  when_create_children: 2,
  depth: 8,
  komi_type: UctKomiType::Dynamic,
  red: 0.45,
  green: 0.5,
  komi_min_iterations: 3_000,
};

const DEFAULT_MINIMAX_CONFIG: MinimaxConfig = MinimaxConfig {
  threads_count: 4,
  minimax_type: MinimaxType::NegaScout,
  minimax_moves_sorting: MinimaxMovesSorting::TrajectoriesCount,
  hash_table_size: 10_000,
  rebuild_trajectories: false,
};

const DEFAULT_BOT_CONFIG: BotConfig = BotConfig {
  time_gap: 100,
  solver: Solver::Uct,
};

const DEFAULT_CONFIG: Config = Config {
  uct: DEFAULT_UCT_CONFIG,
  minimax: DEFAULT_MINIMAX_CONFIG,
  bot: DEFAULT_BOT_CONFIG,
};

static mut CONFIG: Config = DEFAULT_CONFIG;

#[inline]
pub fn config() -> &'static Config {
  unsafe { &CONFIG }
}

#[inline]
fn config_mut() -> &'static mut Config {
  unsafe { &mut CONFIG }
}

pub fn cli_parse() {
  let num_cpus_string = num_cpus::get().to_string();
  let matches = App::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .group(
      ArgGroup::with_name("Minimax")
        .args(&["minimax-type", "moves-order", "rebuild-trajectories"])
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
        .possible_values(&Solver::variants())
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
        .possible_values(&MinimaxType::variants())
        .case_insensitive(true)
        .default_value("NegaScout"),
    )
    .arg(
      Arg::with_name("moves-order")
        .long("moves-order")
        .help("Moves sorting method for Minimax")
        .takes_value(true)
        .possible_values(&MinimaxMovesSorting::variants())
        .case_insensitive(true)
        .default_value("TrajectoriesCount"),
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
        .possible_values(&UCB_TYPE_VARIANTS)
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
        .possible_values(&UCT_KOMI_TYPE_VARIANTS)
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
    ucb_type: value_t!(matches.value_of("ucb-type"), UcbTypeArg)
      .unwrap_or_else(|e| e.exit())
      .0,
    draw_weight: value_t!(matches.value_of("draw-weight"), f64).unwrap_or_else(|e| e.exit()),
    uctk: value_t!(matches.value_of("uctk"), f64).unwrap_or_else(|e| e.exit()),
    when_create_children: value_t!(matches.value_of("when-create-children"), usize).unwrap_or_else(|e| e.exit()),
    depth: value_t!(matches.value_of("depth"), u32).unwrap_or_else(|e| e.exit()),
    komi_type: value_t!(matches.value_of("komi-type"), UctKomiTypeArg)
      .unwrap_or_else(|e| e.exit())
      .0,
    red: value_t!(matches.value_of("red"), f64).unwrap_or_else(|e| e.exit()),
    green: value_t!(matches.value_of("green"), f64).unwrap_or_else(|e| e.exit()),
    komi_min_iterations: value_t!(matches.value_of("komi-min-iterations"), usize).unwrap_or_else(|e| e.exit()),
  };
  let minimax_config = MinimaxConfig {
    threads_count,
    minimax_type: value_t!(matches.value_of("minimax-type"), MinimaxType).unwrap_or_else(|e| e.exit()),
    minimax_moves_sorting: value_t!(matches.value_of("moves-order"), MinimaxMovesSorting).unwrap_or_else(|e| e.exit()),
    hash_table_size: value_t!(matches.value_of("hash-table-size"), usize).unwrap_or_else(|e| e.exit()),
    rebuild_trajectories: matches.is_present("rebuild-trajectories"),
  };
  let bot_config = BotConfig {
    time_gap: value_t!(matches.value_of("time-gap"), u32).unwrap_or_else(|e| e.exit()),
    solver: value_t!(matches.value_of("solver"), Solver).unwrap_or_else(|e| e.exit()),
  };
  let config = Config {
    uct: uct_config,
    minimax: minimax_config,
    bot: bot_config,
  };
  unsafe {
    CONFIG = config;
  }
}

#[inline]
pub fn hash_table_size() -> usize {
  config().minimax.hash_table_size
}

#[inline]
pub fn time_gap() -> u32 {
  config().bot.time_gap
}

#[inline]
pub fn solver() -> Solver {
  config().bot.solver
}
