use crate::config::{Config, Solver};
use crate::heuristic;
use oppai_field::field::{self, Field, NonZeroPos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use oppai_minimax::minimax::Minimax;
use oppai_patterns::patterns::Patterns;
use oppai_uct::uct::UctRoot;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::{cmp, sync::Arc, time::Duration};

const MIN_COMPLEXITY: u32 = 0;

const MAX_COMPLEXITY: u32 = 100;

const MIN_UCT_ITERATIONS: usize = 0;

const MAX_UCT_ITERATIONS: usize = 500_000;

const MIN_MINIMAX_DEPTH: u32 = 0;

const MAX_MINIMAX_DEPTH: u32 = 12;

fn is_field_occupied(field: &Field) -> bool {
  for pos in field.min_pos()..=field.max_pos() {
    if field.cell(pos).is_putting_allowed() {
      return false;
    }
  }
  true
}

pub struct Bot<R> {
  pub rng: R,
  pub patterns: Arc<Patterns>,
  pub field: Field,
  pub uct: UctRoot,
  pub minimax: Minimax,
  pub config: Config,
}

impl<S, R> Bot<R>
where
  R: Rng + SeedableRng<Seed = S> + Send,
  Standard: Distribution<S>,
{
  pub fn new(width: u32, height: u32, mut rng: R, patterns: Arc<Patterns>, config: Config) -> Self {
    info!("Initialization with width {0}, height {1}.", width, height);
    let length = field::length(width, height);
    let zobrist = Arc::new(Zobrist::new(length * 2, &mut rng));
    let field_zobrist = Arc::clone(&zobrist);
    Bot {
      rng,
      patterns,
      field: Field::new(width, height, field_zobrist),
      uct: UctRoot::new(config.uct.clone(), length),
      minimax: Minimax::new(config.minimax.clone()),
      config,
    }
  }

  pub fn initial_move(&self) -> Option<NonZeroPos> {
    let result = match self.field.moves_count() {
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
          } else if dy < 0 {
            Some((x, y + 1))
          } else {
            Some((x, y - 1))
          }
        }
      }
      _ => None,
    };
    result.and_then(|(x, y)| NonZeroPos::new(self.field.to_pos(x, y)))
  }

  pub fn best_move(&mut self, player: Player) -> Option<NonZeroPos> {
    self.best_move_with_complexity(player, (MAX_COMPLEXITY - MIN_COMPLEXITY) / 2 + MIN_COMPLEXITY)
  }

  pub fn best_move_with_time(&mut self, player: Player, time: u32) -> Option<NonZeroPos> {
    if self.field.width() < 3 || self.field.height() < 3 || is_field_occupied(&self.field) {
      return None;
    }
    if let Some(pos) = self.initial_move() {
      return Some(pos);
    }
    if let Some(&pos) = self.patterns.find(&self.field, player, false).choose(&mut self.rng) {
      return NonZeroPos::new(pos);
    }
    match self.config.solver {
      Solver::Uct => self
        .uct
        .best_move_with_time(
          &self.field,
          player,
          &mut self.rng,
          Duration::from_millis(u64::from(time - self.config.time_gap)),
        )
        .or_else(|| heuristic::heuristic(&self.field, player)),
      Solver::Minimax => self
        .minimax
        .minimax_with_time(
          &mut self.field,
          player,
          Duration::from_millis(u64::from(time - self.config.time_gap)),
        )
        .or_else(|| heuristic::heuristic(&self.field, player)),
      Solver::Heuristic => heuristic::heuristic(&self.field, player),
    }
  }

  pub fn best_move_with_full_time(
    &mut self,
    player: Player,
    remaining_time: u32,
    time_per_move: u32,
  ) -> Option<NonZeroPos> {
    self.best_move_with_time(player, time_per_move + remaining_time / 25)
  }

  pub fn best_move_with_complexity(&mut self, player: Player, complexity: u32) -> Option<NonZeroPos> {
    if self.field.width() < 3 || self.field.height() < 3 || is_field_occupied(&self.field) {
      return None;
    }
    if let Some(pos) = self.initial_move() {
      return Some(pos);
    }
    if let Some(&pos) = self.patterns.find(&self.field, player, false).choose(&mut self.rng) {
      return NonZeroPos::new(pos);
    }
    match self.config.solver {
      Solver::Uct => {
        let iterations_count = (complexity - MIN_COMPLEXITY) as usize * (MAX_UCT_ITERATIONS - MIN_UCT_ITERATIONS)
          / (MAX_COMPLEXITY - MIN_COMPLEXITY) as usize
          + MIN_UCT_ITERATIONS;
        self
          .uct
          .best_move_with_iterations_count(&self.field, player, &mut self.rng, iterations_count)
          .or_else(|| heuristic::heuristic(&self.field, player))
      }
      Solver::Minimax => {
        let depth = (complexity - MIN_COMPLEXITY) * (MAX_MINIMAX_DEPTH - MIN_MINIMAX_DEPTH)
          / (MAX_COMPLEXITY - MIN_COMPLEXITY)
          + MIN_MINIMAX_DEPTH;
        self
          .minimax
          .minimax(&mut self.field, player, depth)
          .or_else(|| heuristic::heuristic(&self.field, player))
      }
      Solver::Heuristic => heuristic::heuristic(&self.field, player),
    }
  }
}