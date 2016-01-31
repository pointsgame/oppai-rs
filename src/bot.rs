use std::cmp;
use std::sync::Arc;
use rand::{XorShiftRng, SeedableRng};
use player::Player;
use config;
use config::Solver;
use zobrist::Zobrist;
use field;
use field::Field;
use uct::UctRoot;
use heuristic;
use minimax;
use patterns::Patterns;

const BOT_STR: &'static str = "bot";

const MIN_COMPLEXITY: u32 = 0;

const MAX_COMPLEXITY: u32 = 100;

const MIN_UCT_ITERATIONS: usize = 0;

const MAX_UCT_ITERATIONS: usize = 500000;

const MIN_MINIMAX_DEPTH: u32 = 0;

const MAX_MINIMAX_DEPTH: u32 = 8;

pub struct Bot {
  rng: XorShiftRng,
  patterns: Arc<Patterns>,
  zobrist: Arc<Zobrist>,
  field: Field,
  uct: UctRoot
}

impl Bot {
  pub fn new(width: u32, height: u32, seed: u64, patterns: Arc<Patterns>) -> Bot {
    info!(target: BOT_STR, "Initialization with width {0}, height {1}, seed {2}.", width, height, seed);
    let length = field::length(width, height);
    let seed_array = [3, seed as u32, 7, (seed >> 32) as u32];
    let mut rng = XorShiftRng::from_seed(seed_array);
    let zobrist = Arc::new(Zobrist::new(length * 2, &mut rng));
    let field_zobrist = zobrist.clone();
    Bot {
      rng: rng,
      patterns: patterns,
      zobrist: zobrist,
      field: Field::new(width, height, field_zobrist),
      uct: UctRoot::new(length)
    }
  }

  pub fn initial_move(&self) -> Option<(u32, u32)> {
    match self.field.moves_count() {
      0 => Some((self.field.width() / 2, self.field.height() / 2)),
      1 => {
        let width = self.field.width();
        let height = self.field.height();
        let pos = self.field.points_seq()[0];
        let x = self.field.to_x(pos);
        let y = self.field.to_y(pos);
        if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
          Some((width / 2, height / 2))
        } else if cmp::min(x, width - x - 1) < cmp::min(y, height - y - 1) {
          if x < width - x - 1 {
            Some((x + 1, y))
          } else {
            Some((x - 1, y))
          }
        } else if cmp::min(x, width - x - 1) > cmp::min(y, height - y - 1) {
          if y < height - y - 1 {
            Some((x, y + 1))
          } else {
            Some((x, y - 1))
          }
        } else {
          let dx = x as i32 - (width / 2) as i32;
          let dy = y as i32 - (height / 2) as i32;
          if dx.abs() > dy.abs() {
            if dx < 0 {
              Some((x + 1, y))
            } else {
              Some((x - 1, y))
            }
          } else {
            if dy < 0 {
              Some((x, y + 1))
            } else {
              Some((x, y - 1))
            }
          }
        }
      },
      _ => None
    }
  }

  fn is_field_occupied(field: &Field) -> bool {
    for pos in field.min_pos() .. field.max_pos() + 1 {
      if field.cell(pos).is_putting_allowed() {
        return false;
      }
    }
    true
  }

  pub fn best_move(&mut self, player: Player) -> Option<(u32, u32)> {
    self.best_move_with_complexity(player, (MAX_COMPLEXITY - MIN_COMPLEXITY) / 2 + MIN_COMPLEXITY)
  }

  pub fn best_move_with_time(&mut self, player: Player, time: u32) -> Option<(u32, u32)> {
    if self.field.width() < 3 || self.field.height() < 3 || Bot::is_field_occupied(&self.field) {
      return None;
    }
    if let Some(m) = self.initial_move() {
      return Some(m);
    }
    if let Some(pos) = self.patterns.find_rand(&self.field, player, false, &mut self.rng) {
      return Some((self.field.to_x(pos), self.field.to_y(pos)));
    }
    match config::solver() {
      Solver::Uct => {
        self.uct.best_move_with_time(&self.field, player, &mut self.rng, time - config::time_gap())
          .or_else(|| heuristic::heuristic(&self.field, player))
          .map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      },
      Solver::Minimax => {
        minimax::minimax_with_time(&mut self.field, player, &mut self.rng, time)
          .or_else(|| heuristic::heuristic(&self.field, player))
          .map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      },
      Solver::Heuristic => {
        heuristic::heuristic(&self.field, player).map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      }
    }
  }

  pub fn best_move_with_complexity(&mut self, player: Player, complexity: u32) -> Option<(u32, u32)> {
    if self.field.width() < 3 || self.field.height() < 3 || Bot::is_field_occupied(&self.field) {
      return None;
    }
    if let Some(m) = self.initial_move() {
      return Some(m);
    }
    if let Some(pos) = self.patterns.find_rand(&self.field, player, false, &mut self.rng) {
      return Some((self.field.to_x(pos), self.field.to_y(pos)));
    }
    match config::solver() {
      Solver::Uct => {
        let iterations_count = (complexity - MIN_COMPLEXITY) as usize * (MAX_UCT_ITERATIONS - MIN_UCT_ITERATIONS) / (MAX_COMPLEXITY - MIN_COMPLEXITY) as usize + MIN_UCT_ITERATIONS;
        self.uct.best_move_with_iterations_count(&self.field, player, &mut self.rng, iterations_count)
          .or_else(|| heuristic::heuristic(&self.field, player))
          .map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      },
      Solver::Minimax => {
        let depth = (complexity - MIN_COMPLEXITY) * (MAX_MINIMAX_DEPTH - MIN_MINIMAX_DEPTH) / (MAX_COMPLEXITY - MIN_COMPLEXITY) + MIN_MINIMAX_DEPTH;
        minimax::minimax(&mut self.field, player, &mut self.rng, depth)
          .or_else(|| heuristic::heuristic(&self.field, player))
          .map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      },
      Solver::Heuristic => {
        heuristic::heuristic(&self.field, player).map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
      }
    }
  }

  pub fn put_point(&mut self, x: u32, y: u32, player: Player) -> bool {
    let pos = self.field.to_pos(x, y);
    self.field.put_point(pos, player)
  }

  pub fn undo(&mut self) -> bool {
    self.field.undo()
  }
}
