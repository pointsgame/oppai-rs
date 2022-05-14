use crate::hash_table::{HashData, HashTable, HashType};
use crate::trajectories_pruning::TrajectoriesPruning;
use crossbeam::{self, queue::SegQueue};
use oppai_common::common;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use std::{
  iter,
  sync::atomic::{AtomicBool, AtomicIsize, Ordering},
};
use strum::{EnumString, EnumVariantNames};

#[derive(Clone, Copy, PartialEq, Debug, EnumString, EnumVariantNames)]
pub enum MinimaxType {
  NegaScout,
  Mtdf,
}

#[derive(Clone, PartialEq, Debug)]
pub struct MinimaxConfig {
  pub threads_count: usize,
  pub minimax_type: MinimaxType,
  pub hash_table_size: usize,
  pub rebuild_trajectories: bool,
}

impl Default for MinimaxConfig {
  fn default() -> Self {
    Self {
      threads_count: num_cpus::get_physical(),
      minimax_type: MinimaxType::NegaScout,
      hash_table_size: 10000,
      rebuild_trajectories: false,
    }
  }
}

pub struct Minimax {
  config: MinimaxConfig,
  hash_table: HashTable,
}

impl Minimax {
  pub fn new(config: MinimaxConfig) -> Minimax {
    let hash_table = HashTable::new(config.hash_table_size);
    Minimax { config, hash_table }
  }

  #[inline]
  fn put_new_hash_value(hash_table: &HashTable, hash: u64, pos: Pos, depth: u32, cur_estimation: i32, beta: i32) {
    let new_hash_type = if cur_estimation < beta {
      HashType::Exact
    } else {
      HashType::Beta
    };
    let new_hash_value = HashData::new(depth, new_hash_type, pos, cur_estimation);
    hash_table.put(hash, new_hash_value);
  }

  fn alpha_beta(
    field: &mut Field,
    depth: u32,
    last_pos: Option<NonZeroPos>,
    player: Player,
    trajectories_pruning: &TrajectoriesPruning,
    alpha: i32,
    beta: i32,
    empty_board: &mut Vec<u32>,
    hash_table: &HashTable,
    should_stop: &AtomicBool,
  ) -> i32 {
    if should_stop.load(Ordering::Relaxed) {
      return alpha;
    }
    let enemy = player.next();
    if let Some(last_pos) = last_pos {
      if common::is_last_move_stupid(field, last_pos.get(), enemy) {
        return i32::max_value();
      }
    }
    if depth == 0 {
      return field.score(player);
    }
    let moves = trajectories_pruning.moves();
    if moves.is_empty() {
      return field.score(player);
    }
    let mut cur_alpha = alpha;
    let hash_value = hash_table.get(field.colored_hash(player));
    let hash_type = hash_value.hash_type();
    let hash_pos_option = match hash_type {
      HashType::Exact | HashType::Beta => {
        if hash_value.depth() == depth {
          let hash_estimation = hash_value.estimation();
          if hash_estimation > alpha {
            cur_alpha = hash_estimation;
            if cur_alpha >= beta {
              return cur_alpha;
            }
          }
        }
        Some(hash_value.pos())
      }
      HashType::Alpha => {
        if hash_value.depth() == depth && hash_value.estimation() <= alpha {
          return alpha;
        }
        None
      }
      HashType::Empty => None,
    };
    if last_pos.is_some() && beta - alpha > 1 {
      let enemy_trajectories_pruning = trajectories_pruning.dec_and_swap(depth - 1, empty_board);
      let cur_estimation = -Minimax::alpha_beta(
        field,
        depth - 1,
        None,
        enemy,
        &enemy_trajectories_pruning,
        -beta,
        -beta + 1,
        empty_board,
        hash_table,
        should_stop,
      );
      if cur_estimation >= beta {
        return cur_estimation;
      }
    }
    // Try the best move from the hash table.
    if let Some(hash_pos) = hash_pos_option {
      field.put_point(hash_pos, player);
      if common::is_penult_move_stupid(field) {
        field.undo();
        return i32::max_value();
      }
      let next_trajectories_pruning =
        trajectories_pruning.next(field, enemy, depth - 1, empty_board, hash_pos, should_stop);
      let cur_estimation = -Minimax::alpha_beta(
        field,
        depth - 1,
        NonZeroPos::new(hash_pos),
        enemy,
        &next_trajectories_pruning,
        -beta,
        -cur_alpha,
        empty_board,
        hash_table,
        should_stop,
      );
      field.undo();
      // We should check it before putting the best move to the hash table because
      // it's possible that current estimation is higher than real in case of time
      // out.
      if should_stop.load(Ordering::Relaxed) {
        return cur_alpha;
      }
      if cur_estimation > cur_alpha {
        Minimax::put_new_hash_value(
          hash_table,
          field.colored_hash(player),
          hash_pos,
          depth,
          cur_estimation,
          beta,
        );
        cur_alpha = cur_estimation;
        if cur_alpha >= beta {
          return cur_alpha;
        }
      }
    }
    // For all moves instead the one from the hash table.
    for &pos in moves.iter().filter(|&&pos| Some(pos) != hash_pos_option) {
      field.put_point(pos, player);
      if common::is_penult_move_stupid(field) {
        field.undo();
        return i32::max_value();
      }
      let next_trajectories_pruning = trajectories_pruning.next(field, enemy, depth - 1, empty_board, pos, should_stop);
      let mut cur_estimation = -Minimax::alpha_beta(
        field,
        depth - 1,
        NonZeroPos::new(pos),
        enemy,
        &next_trajectories_pruning,
        -cur_alpha - 1,
        -cur_alpha,
        empty_board,
        hash_table,
        should_stop,
      );
      if cur_estimation > cur_alpha && cur_estimation < beta {
        cur_estimation = -Minimax::alpha_beta(
          field,
          depth - 1,
          NonZeroPos::new(pos),
          enemy,
          &next_trajectories_pruning,
          -beta,
          -cur_estimation,
          empty_board,
          hash_table,
          should_stop,
        );
      }
      field.undo();
      // We should check it before putting the best move to the hash table because
      // it's possible that current estimation is higher than real in case of time
      // out.
      if should_stop.load(Ordering::Relaxed) {
        return cur_alpha;
      }
      if cur_estimation > cur_alpha {
        Minimax::put_new_hash_value(hash_table, field.colored_hash(player), pos, depth, cur_estimation, beta);
        cur_alpha = cur_estimation;
        if cur_alpha >= beta {
          break;
        }
      }
    }
    if cur_alpha == alpha {
      let new_hash_value = HashData::new(depth, HashType::Alpha, 0, alpha);
      hash_table.put(field.colored_hash(player), new_hash_value);
    }
    cur_alpha
  }

  pub fn alpha_beta_parallel(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    alpha: i32,
    beta: i32,
    trajectories_pruning: &mut TrajectoriesPruning,
    best_move: &mut Option<NonZeroPos>,
    should_stop: &AtomicBool,
  ) -> i32 {
    info!(
      "Starting parallel alpha beta with depth {}, player {} and beta {}.",
      depth, player, beta
    );
    if depth == 0 || should_stop.load(Ordering::Relaxed) {
      return field.score(player);
    }
    let moves = trajectories_pruning.moves();
    debug!(
      "Moves in consideration: {:?}.",
      moves
        .iter()
        .map(|&pos| (field.to_x(pos), field.to_y(pos)))
        .collect::<Vec<(u32, u32)>>()
    );
    if moves.is_empty() || should_stop.load(Ordering::Relaxed) {
      return field.score(player);
    }
    let queue = SegQueue::new();
    if let Some(best_pos) = *best_move {
      queue.push(best_pos.get());
      for &pos in moves.iter().filter(|&&pos| pos != best_pos.get()) {
        queue.push(pos);
      }
    } else {
      for &pos in moves.iter() {
        queue.push(pos);
      }
    }
    let atomic_alpha = AtomicIsize::new(alpha as isize);
    let best_moves = SegQueue::new();
    let skipped_moves = SegQueue::new();
    let first_move_considered = AtomicBool::new(best_move.is_none());
    crossbeam::scope(|scope| {
      for _ in 0..self.config.threads_count {
        scope.spawn(|_| {
          let mut local_field = field.clone();
          let mut local_empty_board = iter::repeat(0u32).take(field.length()).collect::<Vec<_>>();
          let mut local_best_move = 0;
          let mut local_alpha = alpha;
          let enemy = player.next();
          while let Some(pos) = queue.pop() {
            if should_stop.load(Ordering::Relaxed) {
              break;
            }
            local_field.put_point(pos, player);
            let next_trajectories_pruning = trajectories_pruning.next(
              &mut local_field,
              enemy,
              depth - 1,
              &mut local_empty_board,
              pos,
              should_stop,
            );
            if should_stop.load(Ordering::Relaxed) {
              break;
            }
            let cur_alpha = atomic_alpha.load(Ordering::Relaxed) as i32;
            if cur_alpha >= beta {
              skipped_moves.push(pos);
              break;
            }
            let mut cur_estimation = -Minimax::alpha_beta(
              &mut local_field,
              depth - 1,
              NonZeroPos::new(pos),
              enemy,
              &next_trajectories_pruning,
              -cur_alpha - 1,
              -cur_alpha,
              &mut local_empty_board,
              &self.hash_table,
              should_stop,
            );
            if should_stop.load(Ordering::Relaxed) {
              break;
            }
            if cur_estimation > cur_alpha && cur_estimation < beta {
              cur_estimation = -Minimax::alpha_beta(
                &mut local_field,
                depth - 1,
                NonZeroPos::new(pos),
                enemy,
                &next_trajectories_pruning,
                -beta,
                -cur_estimation,
                &mut local_empty_board,
                &self.hash_table,
                should_stop,
              );
            }
            // We should check it before the best move assignment because it's possible
            // that current estimation is higher than real in case of time out.
            if should_stop.load(Ordering::Relaxed) {
              break;
            }
            debug!(
              "Estimation for move ({}, {}) is {}, alpha is {}, beta is {}.",
              field.to_x(pos),
              field.to_y(pos),
              cur_estimation,
              cur_alpha,
              beta
            );
            local_field.undo();
            if cur_estimation > cur_alpha {
              local_alpha = cur_estimation;
              local_best_move = pos;
            }
            loop {
              let last_alpha = atomic_alpha.load(Ordering::SeqCst);
              if cur_estimation <= last_alpha as i32
                || atomic_alpha
                  .compare_exchange_weak(last_alpha, cur_estimation as isize, Ordering::SeqCst, Ordering::SeqCst)
                  .is_ok()
              {
                break;
              }
            }
            if *best_move == NonZeroPos::new(pos) {
              first_move_considered.store(true, Ordering::SeqCst);
            }
          }
          if local_best_move != 0 {
            best_moves.push((local_best_move, local_alpha));
          }
        });
      }
    })
    .expect("Minimax alpha_beta_parallel panic");
    let mut result = 0;
    let best_alpha = atomic_alpha.load(Ordering::SeqCst) as i32;
    if best_alpha > alpha {
      let moves = trajectories_pruning.moves_mut();
      moves.clear();
      while let Some((pos, pos_alpha)) = best_moves.pop() {
        if pos_alpha == best_alpha || pos_alpha >= beta {
          moves.push(pos);
        }
        if pos_alpha == best_alpha && result == 0 {
          result = pos;
        }
      }
      while let Some(pos) = skipped_moves.pop() {
        moves.push(pos);
      }
      while let Some(pos) = queue.pop() {
        moves.push(pos);
      }
    }
    if !first_move_considered.load(Ordering::SeqCst) {
      info!("First move was not considered.");
    } else if result == 0 {
      info!("Best move is not found.");
      *best_move = None;
    } else {
      info!("Best move is ({}, {}).", field.to_x(result), field.to_y(result));
      *best_move = NonZeroPos::new(result);
    }
    info!("Estimation is {}.", best_alpha);
    best_alpha
  }

  fn mtdf(
    &self,
    field: &mut Field,
    player: Player,
    trajectories_pruning: &mut TrajectoriesPruning,
    depth: u32,
    best_move: &mut Option<NonZeroPos>,
    should_stop: &AtomicBool,
  ) -> i32 {
    let mut alpha = trajectories_pruning.alpha().unwrap_or_else(|| field.score(player));
    let mut beta = trajectories_pruning.beta().unwrap_or_else(|| field.score(player));
    while alpha != beta {
      if let [single_move] = *trajectories_pruning.moves().as_slice() {
        *best_move = NonZeroPos::new(single_move);
        return alpha;
      }
      if should_stop.load(Ordering::Relaxed) {
        return alpha;
      }
      let mut cur_best_move = *best_move;
      let center = if (alpha + beta) % 2 == -1 {
        (alpha + beta) / 2 - 1
      } else {
        (alpha + beta) / 2
      };
      let cur_estimation = self.alpha_beta_parallel(
        field,
        player,
        depth,
        center,
        center + 1,
        trajectories_pruning,
        &mut cur_best_move,
        should_stop,
      );
      if cur_estimation > center {
        alpha = cur_estimation;
      } else {
        beta = cur_estimation;
      }
      *best_move = cur_best_move.or(*best_move);
    }
    alpha
  }

  fn nega_scout(
    &self,
    field: &mut Field,
    player: Player,
    trajectories_pruning: &mut TrajectoriesPruning,
    depth: u32,
    best_move: &mut Option<NonZeroPos>,
    should_stop: &AtomicBool,
  ) -> i32 {
    let alpha = trajectories_pruning.alpha().unwrap_or_else(|| field.score(player));
    let beta = trajectories_pruning.beta().unwrap_or_else(|| field.score(player));
    self.alpha_beta_parallel(
      field,
      player,
      depth,
      alpha,
      beta,
      trajectories_pruning,
      best_move,
      should_stop,
    )
  }

  pub fn minimax(&self, field: &mut Field, player: Player, depth: u32, should_stop: &AtomicBool) -> Option<NonZeroPos> {
    info!("Starting minimax with depth {} and player {}.", depth, player);
    if depth == 0 {
      return None;
    }
    let mut empty_board = iter::repeat(0u32).take(field.length()).collect::<Vec<_>>();
    let mut trajectories_pruning = TrajectoriesPruning::new(
      self.config.rebuild_trajectories,
      field,
      player,
      depth,
      &mut empty_board,
      should_stop,
    );
    let mut best_move = None;
    info!("Calculating of our estimation. Player is {}", player);
    let minimax_function = match self.config.minimax_type {
      MinimaxType::NegaScout => Minimax::nega_scout,
      MinimaxType::Mtdf => Minimax::mtdf,
    };
    let estimation = minimax_function(
      self,
      field,
      player,
      &mut trajectories_pruning,
      depth,
      &mut best_move,
      should_stop,
    );
    let enemy = player.next();
    let mut enemy_best_move = best_move;
    let mut enemy_trajectories_pruning = trajectories_pruning.dec_and_swap(depth - 1, &mut empty_board);
    info!(
      "Calculating of enemy estimation with upper bound {}. Player is {}",
      -estimation + 1,
      enemy
    );
    // Check if we could lose something if we don't make the current best move.
    // If we couldn't that means that the current best move is just a random move.
    if -self.alpha_beta_parallel(
      field,
      enemy,
      depth - 1,
      -estimation,
      -estimation + 1,
      &mut enemy_trajectories_pruning,
      &mut enemy_best_move,
      should_stop,
    ) < estimation
    {
      info!(
        "Estimation is greater than enemy estimation. So the best move is {:?}, estimation is {}.",
        best_move.map(|pos| (field.to_x(pos.get()), field.to_y(pos.get()))),
        estimation
      );
      best_move
    } else {
      info!(
        "Estimation is less than or equal enemy estimation. So all moves have the same estimation {}.",
        estimation
      );
      None
    }
  }

  pub fn minimax_with_time(&self, field: &mut Field, player: Player, should_stop: &AtomicBool) -> Option<NonZeroPos> {
    let enemy = player.next();
    let mut depth = 1;
    let mut best_move = None;
    let mut cur_best_move = None;
    let mut enemy_best_move = None;
    let mut empty_board = iter::repeat(0u32).take(field.length()).collect::<Vec<_>>();
    let mut trajectories_pruning = TrajectoriesPruning::new(
      self.config.rebuild_trajectories,
      field,
      player,
      depth,
      &mut empty_board,
      should_stop,
    );
    let minimax_function = match self.config.minimax_type {
      MinimaxType::NegaScout => Minimax::nega_scout,
      MinimaxType::Mtdf => Minimax::mtdf,
    };
    while !should_stop.load(Ordering::Relaxed) {
      let estimation = minimax_function(
        self,
        field,
        player,
        &mut trajectories_pruning,
        depth,
        &mut cur_best_move,
        should_stop,
      );
      if should_stop.load(Ordering::Relaxed) {
        // If we found the best move on the previous iteration then the current best
        // move can't be worse than that move. Otherwise it's possible that the
        // current best move is just a
        // random move.
        if best_move.is_some() {
          best_move = cur_best_move;
        }
        break;
      }
      // Check if we could lose something if we don't make the current best move.
      // If we couldn't that means that the current best move is just a random move.
      // If we found the best move on previous iteration then likely the current best
      // move is also the best one.
      best_move = if best_move.is_some()
        || cur_best_move.is_none()
        || -self.alpha_beta_parallel(
          field,
          enemy,
          depth - 1,
          -estimation,
          -estimation + 1,
          &mut trajectories_pruning.dec_and_swap(depth - 1, &mut empty_board),
          &mut enemy_best_move,
          should_stop,
        ) < estimation
      {
        info!(
          "Found best move {:?} with estimation {} at depth {}",
          cur_best_move.map(|pos| (field.to_x(pos.get()), field.to_y(pos.get()))),
          estimation,
          depth
        );
        cur_best_move
      } else {
        None
      };
      if should_stop.load(Ordering::Relaxed) {
        break;
      }
      depth += 1;
      trajectories_pruning = trajectories_pruning.inc(field, player, depth, &mut empty_board, should_stop);
    }
    best_move
  }
}
