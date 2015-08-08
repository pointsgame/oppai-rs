use std::sync::Arc;
use rand::{XorShiftRng, SeedableRng};
use types::{Coord, Time};
use player::Player;
use config;
use zobrist::Zobrist;
use field;
use field::Field;
use uct::UctRoot;
use heuristic;

const BOT_STR: &'static str = "bot";

const MIN_COMPLEXITY: u8 = 0;

const MAX_COMPLEXITY: u8 = 100;

const MIN_UCT_ITERATIONS: usize = 0;

const MAX_UCT_ITERATIONS: usize = 500000;

pub struct Bot {
  rng: XorShiftRng,
  zobrist: Arc<Zobrist>,
  field: Field,
  uct: UctRoot
}

impl Bot {
  pub fn new(width: Coord, height: Coord, seed: u64) -> Bot {
    info!(target: BOT_STR, "Initialization with width {0}, height {1}, seed {2}.", width, height, seed);
    let length = field::length(width, height);
    let seed_array = [3, seed as u32, 7, (seed >> 32) as u32];
    let mut rng = XorShiftRng::from_seed(seed_array);
    let zobrist = Arc::new(Zobrist::new(length * 2, &mut rng));
    let field_zobrist = zobrist.clone();
    Bot {
      rng: rng,
      zobrist: zobrist,
      field: Field::new(width, height, field_zobrist),
      uct: UctRoot::new(length)
    }
  }

  pub fn best_move(&mut self, player: Player) -> Option<(Coord, Coord)> {
    self.best_move_with_complexity(player, (MAX_COMPLEXITY - MIN_COMPLEXITY) / 2 + MIN_COMPLEXITY)
  }

  pub fn best_move_with_time(&mut self, player: Player, time: Time) -> Option<(Coord, Coord)> {
    let mut result = self.uct.best_move_with_time(&self.field, player, &mut self.rng, time - config::time_gap());
    if result.is_none() {
      result = heuristic::heuristic(&self.field, player);
    }
    result.map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
  }

  pub fn best_move_with_complexity(&mut self, player: Player, complexity: u8) -> Option<(Coord, Coord)> {
    let iterations_count = (complexity - MIN_COMPLEXITY) as usize * (MAX_UCT_ITERATIONS - MIN_UCT_ITERATIONS) / (MAX_COMPLEXITY - MIN_COMPLEXITY) as usize + MIN_UCT_ITERATIONS;
    let mut result = self.uct.best_move_with_iterations_count(&self.field, player, &mut self.rng, iterations_count);
    if result.is_none() {
      result = heuristic::heuristic(&self.field, player);
    }
    result.map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
  }

  pub fn put_point(&mut self, x: Coord, y: Coord, player: Player) -> bool {
    let pos = self.field.to_pos(x, y);
    self.field.put_point(pos, player)
  }

  pub fn undo(&mut self) -> bool {
    self.field.undo()
  }
}
