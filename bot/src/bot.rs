use crate::config::{Config, Solver};
use crate::heuristic;
use oppai_field::field::{self, Field, NonZeroPos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use oppai_ladders::ladders::ladders;
use oppai_minimax::minimax::Minimax;
use oppai_patterns::patterns::Patterns;
use oppai_uct::uct::UctRoot;
#[cfg(feature = "zero")]
use oppai_zero::zero::Zero;
#[cfg(feature = "zero")]
use oppai_zero_torch::model::PyModel;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
#[cfg(feature = "zero")]
use std::path::PathBuf;
use std::{
  cmp,
  sync::atomic::{AtomicBool, Ordering},
  sync::Arc,
  time::{Duration, Instant},
};

fn is_field_occupied(field: &Field) -> bool {
  for pos in field.min_pos()..=field.max_pos() {
    if field.cell(pos).is_putting_allowed() {
      return false;
    }
  }
  true
}

fn with_timeout<T: Send, F: FnOnce() -> T + Send>(f: F, should_stop: &AtomicBool, time: Duration) -> T {
  let (s, r) = crossbeam::channel::bounded(1);
  crossbeam::scope(|scope| {
    scope.spawn(move |_| {
      let result = f();
      s.send(result).unwrap();
    });
    if let Ok(result) = r.recv_timeout(time) {
      return result;
    }
    should_stop.store(true, Ordering::Relaxed);
    r.recv().unwrap()
  })
  .unwrap()
}

pub struct Bot<R> {
  pub rng: R,
  pub patterns: Arc<Patterns>,
  pub field: Field,
  pub uct: UctRoot,
  pub minimax: Minimax,
  #[cfg(feature = "zero")]
  pub zero: Zero<f64, PyModel<f64>>,
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
      #[cfg(feature = "zero")]
      zero: Zero::new({
        // TODO: move to config
        let path = PathBuf::from("model.pt");
        let exists = path.exists();
        if !exists {
          log::warn!("No model at {}", path.display());
        }
        let model = PyModel::<f64>::new(path, width, height, 4).unwrap();
        if exists {
          model.load().unwrap();
        }
        model
      }),
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

  pub fn best_move(
    &mut self,
    player: Player,
    uct_iterations: usize,
    minimax_depth: u32,
    should_stop: &AtomicBool,
  ) -> Option<NonZeroPos> {
    let now = Instant::now();
    if self.field.width() < 3 || self.field.height() < 3 || is_field_occupied(&self.field) {
      return None;
    }
    if let Some(pos) = self.initial_move() {
      return Some(pos);
    }
    if let Some(&pos) = self.patterns.find(&self.field, player, false).choose(&mut self.rng) {
      info!(
        "Cumulative time for patterns evaluation (move is found): {:?}.",
        now.elapsed()
      );
      return NonZeroPos::new(pos);
    }
    let elapsed = now.elapsed();
    info!("Cumulative time for patterns evaluation: {:?}.", elapsed);
    if self.config.ladders {
      let ladders_time_limit = self.config.ladders_time_limit;
      if let (Some(pos), score, depth) = with_timeout(
        || ladders(&mut self.field, player, should_stop),
        should_stop,
        ladders_time_limit,
      ) {
        info!(
          "Cumulative time for ladders evaluation (move is found with score {} and depth {}): {:?}.",
          score,
          depth,
          now.elapsed()
        );
        if (score - self.field.score(player)) as u32 > self.config.ladders_score_limit
          && depth > self.config.ladders_depth_limit
        {
          return Some(pos);
        }
      };
    }
    let elapsed = now.elapsed();
    info!("Cumulative time for ladders evaluation: {:?}.", elapsed);
    let result = match self.config.solver {
      Solver::Uct => self
        .uct
        .best_move(&self.field, player, &mut self.rng, should_stop, uct_iterations)
        .or_else(|| heuristic::heuristic(&self.field, player)),
      Solver::Minimax => self
        .minimax
        .minimax(&mut self.field, player, minimax_depth, should_stop)
        .or_else(|| heuristic::heuristic(&self.field, player)),
      #[cfg(feature = "zero")]
      Solver::Zero => self
        .zero
        .best_move(&self.field, player, &mut self.rng, should_stop, uct_iterations)
        .unwrap()
        .or_else(|| heuristic::heuristic(&self.field, player)),
      Solver::Heuristic => heuristic::heuristic(&self.field, player),
    };
    info!("Cumulative time for best move evaluation: {:?}", now.elapsed());
    result
  }

  pub fn best_move_with_time(
    &mut self,
    player: Player,
    time: Duration,
    should_stop: &AtomicBool,
  ) -> Option<NonZeroPos> {
    let now = Instant::now();
    let time = time - self.config.time_gap;
    if self.field.width() < 3 || self.field.height() < 3 || is_field_occupied(&self.field) {
      return None;
    }
    if let Some(pos) = self.initial_move() {
      return Some(pos);
    }
    if let Some(&pos) = self.patterns.find(&self.field, player, false).choose(&mut self.rng) {
      info!(
        "Cumulative time for patterns evaluation (move is found): {:?}.",
        now.elapsed()
      );
      return NonZeroPos::new(pos);
    }
    let elapsed = now.elapsed();
    info!("Cumulative time for patterns evaluation: {:?}.", elapsed);
    let time_left = if let Some(time_left) = time.checked_sub(elapsed) {
      time_left
    } else {
      return heuristic::heuristic(&self.field, player);
    };
    if self.config.ladders {
      let ladders_time_limit = self.config.ladders_time_limit;
      if let (Some(pos), score, depth) = with_timeout(
        || ladders(&mut self.field, player, should_stop),
        should_stop,
        ladders_time_limit.min(time_left),
      ) {
        info!(
          "Cumulative time for ladders evaluation (move is found with score {} and depth {}): {:?}.",
          score,
          depth,
          now.elapsed()
        );
        if (score - self.field.score(player)) as u32 > self.config.ladders_score_limit
          && depth > self.config.ladders_depth_limit
        {
          return Some(pos);
        }
      };
    }
    let elapsed = now.elapsed();
    info!("Cumulative time for ladders evaluation: {:?}.", elapsed);
    let time_left = if let Some(time_left) = time.checked_sub(elapsed) {
      time_left
    } else {
      return heuristic::heuristic(&self.field, player);
    };
    let result = match self.config.solver {
      Solver::Uct => with_timeout(
        || {
          self
            .uct
            .best_move(&self.field, player, &mut self.rng, should_stop, usize::max_value())
            .or_else(|| heuristic::heuristic(&self.field, player))
        },
        should_stop,
        time_left,
      ),
      Solver::Minimax => with_timeout(
        || {
          self
            .minimax
            .minimax_with_time(&mut self.field, player, should_stop)
            .or_else(|| heuristic::heuristic(&self.field, player))
        },
        should_stop,
        time_left,
      ),
      #[cfg(feature = "zero")]
      Solver::Zero => with_timeout(
        || {
          self
            .zero
            .best_move(&self.field, player, &mut self.rng, should_stop, usize::max_value())
            .unwrap()
            .or_else(|| heuristic::heuristic(&self.field, player))
        },
        should_stop,
        time_left,
      ),
      Solver::Heuristic => heuristic::heuristic(&self.field, player),
    };
    info!("Cumulative time for best move evaluation: {:?}", now.elapsed());
    result
  }

  pub fn clear(&mut self) {
    self.field.clear();
    self.uct.clear();
    self.minimax.clear();
  }
}
