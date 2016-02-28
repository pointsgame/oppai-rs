use std::fmt::{Display, Formatter};
use std::fmt::Result as FmtResult;
use std::io::{Write, Read};
use std::str::FromStr;
use num_cpus;
use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
use toml;

const CONFIG_STR: &'static str = "config";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UcbType {
  Winrate,
  Ucb1,
  Ucb1Tuned
}

const UCB_TYPE_WINRATE_STR: &'static str = "Winrate";

const UCB_TYPE_UCB1_STR: &'static str = "Ucb1";

const UCB_TYPE_UCB1_TUNED_STR: &'static str = "Ucb1Tuned";

impl UcbType {
  pub fn as_str(&self) -> &'static str {
    match *self {
      UcbType::Winrate => UCB_TYPE_WINRATE_STR,
      UcbType::Ucb1 => UCB_TYPE_UCB1_STR,
      UcbType::Ucb1Tuned => UCB_TYPE_UCB1_TUNED_STR
    }
  }
}

impl FromStr for UcbType {
  type Err = &'static str;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      UCB_TYPE_WINRATE_STR => Ok(UcbType::Winrate),
      UCB_TYPE_UCB1_STR => Ok(UcbType::Ucb1),
      UCB_TYPE_UCB1_TUNED_STR => Ok(UcbType::Ucb1Tuned),
      _ => Err("Invalid string!")
    }
  }
}

impl Display for UcbType {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.as_str())
  }
}

impl Encodable for UcbType {
  fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
    s.emit_str(self.as_str())
  }
}

impl Decodable for UcbType {
  fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
    d.read_str().and_then(|s| UcbType::from_str(s.as_str()).map_err(|s| d.error(s)))
  }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UctKomiType {
  None,
  Static,
  Dynamic
}

const UCT_KOMI_TYPE_NONE_STR: &'static str = "None";

const UCT_KOMI_TYPE_STATIC_STR: &'static str = "Static";

const UCT_KOMI_TYPE_DYNAMIC_STR: &'static str = "Dynamic";

impl UctKomiType {
  pub fn as_str(&self) -> &'static str {
    match *self {
      UctKomiType::None => UCT_KOMI_TYPE_NONE_STR,
      UctKomiType::Static => UCT_KOMI_TYPE_STATIC_STR,
      UctKomiType::Dynamic => UCT_KOMI_TYPE_DYNAMIC_STR
    }
  }
}

impl FromStr for UctKomiType {
  type Err = &'static str;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      UCT_KOMI_TYPE_NONE_STR => Ok(UctKomiType::None),
      UCT_KOMI_TYPE_STATIC_STR => Ok(UctKomiType::Static),
      UCT_KOMI_TYPE_DYNAMIC_STR => Ok(UctKomiType::Dynamic),
      _ => Err("Invalid string!")
    }
  }
}

impl Display for UctKomiType {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.as_str())
  }
}

impl Encodable for UctKomiType {
  fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
    s.emit_str(self.as_str())
  }
}

impl Decodable for UctKomiType {
  fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
    d.read_str().and_then(|s| UctKomiType::from_str(s.as_str()).map_err(|s| d.error(s)))
  }
}

const MINIMAX_MOVES_SORTING_NONE: &'static str = "None";

const MINIMAX_MOVES_SORTING_RANDOM: &'static str = "Random";

const MINIMAX_MOVES_SORTING_TRAJECTORIES_COUNT: &'static str = "TrajectoriesCount";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MinimaxMovesSorting {
  None,
  Random,
  TrajectoriesCount
  // Heuristic
}

impl MinimaxMovesSorting {
  pub fn as_str(&self) -> &'static str {
    match *self {
      MinimaxMovesSorting::None => MINIMAX_MOVES_SORTING_NONE,
      MinimaxMovesSorting::Random => MINIMAX_MOVES_SORTING_RANDOM,
      MinimaxMovesSorting::TrajectoriesCount => MINIMAX_MOVES_SORTING_TRAJECTORIES_COUNT
    }
  }
}

impl FromStr for MinimaxMovesSorting {
  type Err = &'static str;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      MINIMAX_MOVES_SORTING_NONE => Ok(MinimaxMovesSorting::None),
      MINIMAX_MOVES_SORTING_RANDOM => Ok(MinimaxMovesSorting::Random),
      MINIMAX_MOVES_SORTING_TRAJECTORIES_COUNT => Ok(MinimaxMovesSorting::TrajectoriesCount),
      _ => Err("Invalid string!")
    }
  }
}

impl Display for MinimaxMovesSorting {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.as_str())
  }
}

impl Encodable for MinimaxMovesSorting {
  fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
    s.emit_str(self.as_str())
  }
}

impl Decodable for MinimaxMovesSorting {
  fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
    d.read_str().and_then(|s| MinimaxMovesSorting::from_str(s.as_str()).map_err(|s| d.error(s)))
  }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Solver {
  Uct,
  NegaScout,
  Heuristic
}

const SOLVER_UCT_STR: &'static str = "UCT";

const SOLVER_NEGA_SCOUT_STR: &'static str = "NegaScout";

const SOLVER_HEURISTIC_STR: &'static str = "Heuristic";

impl Solver {
  pub fn as_str(&self) -> &'static str {
    match *self {
      Solver::Uct => SOLVER_UCT_STR,
      Solver::NegaScout => SOLVER_NEGA_SCOUT_STR,
      Solver::Heuristic => SOLVER_HEURISTIC_STR
    }
  }
}

impl FromStr for Solver {
  type Err = &'static str;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      SOLVER_UCT_STR => Ok(Solver::Uct),
      SOLVER_NEGA_SCOUT_STR => Ok(Solver::NegaScout),
      SOLVER_HEURISTIC_STR => Ok(Solver::Heuristic),
      _ => Err("Invalid string!")
    }
  }
}

impl Display for Solver {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.as_str())
  }
}

impl Encodable for Solver {
  fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
    s.emit_str(self.as_str())
  }
}

impl Decodable for Solver {
  fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
    d.read_str().and_then(|s| Solver::from_str(s.as_str()).map_err(|s| d.error(s)))
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
struct Config {
  uct: UctConfig,
  minimax: MinimaxConfig,
  bot: BotConfig
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
struct UctConfig {
  radius: u32,
  ucb_type: UcbType,
  final_ucb_type: UcbType,
  draw_weight: f64,
  uctk: f64,
  when_create_children: usize,
  depth: u32,
  komi_type: UctKomiType,
  red: f64,
  green: f64,
  komi_min_iterations: usize
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
struct MinimaxConfig {
  minimax_moves_sorting: MinimaxMovesSorting
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
struct BotConfig {
  threads_count: Option<usize>,
  time_gap: u32,
  solver: Solver
}

const DEFAULT_UCT_CONFIG: UctConfig = UctConfig {
  radius: 3,
  ucb_type: UcbType::Ucb1Tuned,
  final_ucb_type: UcbType::Winrate,
  draw_weight: 0.4,
  uctk: 1.0,
  when_create_children: 2,
  depth: 8,
  komi_type: UctKomiType::Dynamic,
  red: 0.45,
  green: 0.5,
  komi_min_iterations: 3000
};

const DEFAULT_MINIMAX_CONFIG: MinimaxConfig = MinimaxConfig {
  minimax_moves_sorting: MinimaxMovesSorting::Random
};

const DEFAULT_BOT_CONFIG: BotConfig = BotConfig {
  threads_count: None,
  time_gap: 100,
  solver: Solver::Uct
};

const DEFAULT_CONFIG: Config = Config {
  uct: DEFAULT_UCT_CONFIG,
  minimax: DEFAULT_MINIMAX_CONFIG,
  bot: DEFAULT_BOT_CONFIG
};

static mut NUM_CPUS: usize = 4;

static mut CONFIG: Config = DEFAULT_CONFIG;

#[inline]
fn config() -> &'static Config {
  unsafe { &CONFIG }
}

pub fn init() {
  let num_cpus = num_cpus::get();
  unsafe {
    NUM_CPUS = num_cpus;
  }
  info!(target: CONFIG_STR, "Default threads count is {}.", num_cpus);
}

pub fn read<T: Read>(input: &mut T) {
  let mut string = String::new();
  input.read_to_string(&mut string).ok();
  if let Some(config) = toml::decode_str::<Config>(string.as_str()) {
    unsafe {
      CONFIG = config
    }
    debug!(target: CONFIG_STR, "Config has been loaded.");
  } else {
    error!(target: CONFIG_STR, "Bad config file!");
  }
}

pub fn write<T: Write>(output: &mut T) {
  write!(output, "{}", toml::encode(config())).ok();
  info!(target: CONFIG_STR, "Config has been written.");
}

#[inline]
pub fn uct_radius() -> u32 {
  config().uct.radius
}

#[inline]
pub fn ucb_type() -> UcbType {
  config().uct.ucb_type
}

#[inline]
pub fn final_ucb_type() -> UcbType {
  config().uct.final_ucb_type
}

#[inline]
pub fn uct_draw_weight() -> f64 {
  config().uct.draw_weight
}

#[inline]
pub fn uctk() -> f64 {
  config().uct.uctk
}

#[inline]
pub fn uct_when_create_children() -> usize {
  config().uct.when_create_children
}

#[inline]
pub fn uct_depth() -> u32 {
  config().uct.depth
}

#[inline]
pub fn threads_count() -> usize {
  config().bot.threads_count.unwrap_or(unsafe { NUM_CPUS })
}

#[inline]
pub fn uct_komi_type() -> UctKomiType {
  config().uct.komi_type
}

#[inline]
pub fn uct_red() -> f64 {
  config().uct.red
}

#[inline]
pub fn uct_green() -> f64 {
  config().uct.green
}

#[inline]
pub fn uct_komi_min_iterations() -> usize {
  config().uct.komi_min_iterations
}

#[inline]
pub fn minimax_moves_sorting() -> MinimaxMovesSorting {
  config().minimax.minimax_moves_sorting
}

#[inline]
pub fn time_gap() -> u32 {
  config().bot.time_gap
}

#[inline]
pub fn solver() -> Solver {
  config().bot.solver
}
