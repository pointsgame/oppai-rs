use std::{iter, thread};
use std::sync::atomic::{AtomicIsize, AtomicUsize, AtomicBool, Ordering};
use rand::{Rng, XorShiftRng};
use comm;
use config;
use player::Player;
use field::{Pos, Field};
use trajectories_pruning::TrajectoriesPruning;
use common;

const MINIMAX_STR: &'static str = "minimax";

fn alpha_beta<T: Rng>(field: &mut Field, depth: u32, last_pos: Pos, player: Player, trajectories_pruning: &TrajectoriesPruning, mut alpha: i32, beta: i32, empty_board: &mut Vec<u32>, rng: &mut T, should_stop: &AtomicBool) -> i32 {
  let enemy = player.next();
  if common::is_last_move_stupid(field, last_pos, enemy) {
    return i32::max_value();
  }
  if depth == 0 {
    return field.score(player);
  }
  let moves = trajectories_pruning.moves();
  if moves.is_empty() {
    return field.score(player);
  }
  for &pos in moves {
    if should_stop.load(Ordering::Relaxed) {
      break;
    }
    field.put_point(pos, player);
    if common::is_penult_move_stuped(field) {
      field.undo();
      return i32::max_value();
    }
    let next_trajectories_pruning = TrajectoriesPruning::from_last(field, enemy, depth - 1, empty_board, rng, trajectories_pruning, pos, should_stop);
    let mut cur_estimation = -alpha_beta(field, depth - 1, pos, enemy, &next_trajectories_pruning, -alpha - 1, -alpha, empty_board, rng, should_stop);
    if cur_estimation > alpha && cur_estimation < beta {
      cur_estimation = -alpha_beta(field, depth - 1, pos, enemy, &next_trajectories_pruning, -beta, -cur_estimation, empty_board, rng, should_stop);
    }
    field.undo();
    if cur_estimation > alpha {
      alpha = cur_estimation;
      if alpha >= beta {
        break;
      }
    }
  }
  alpha
}

fn alpha_beta_parallel<T: Rng>(field: &mut Field, player: Player, depth: u32, alpha: i32, beta: i32, trajectories_pruning: &TrajectoriesPruning, rng: &mut T, best_move: &mut Option<Pos>, should_stop: &AtomicBool) -> i32 {
  info!(target: MINIMAX_STR, "Starting parellel alpha beta with depth {}, player {} and beta {}.", depth, player, beta);
  if depth == 0 || should_stop.load(Ordering::Relaxed) {
    *best_move = None;
    return field.score(player);
  }
  let moves = trajectories_pruning.moves();
  debug!(target: MINIMAX_STR, "Moves in consideration: {:?}.", moves.iter().map(|&pos| (field.to_x(pos), field.to_y(pos))).collect::<Vec<(u32, u32)>>());
  if moves.is_empty() || should_stop.load(Ordering::Relaxed) {
    *best_move = None;
    return field.score(player);
  }
  let (producer, consumer) = comm::spmc::unbounded::new();
  if let Some(best_pos) = *best_move {
    producer.send(best_pos).ok();
    for &pos in moves.iter().filter(|&&pos| pos != best_pos) {
      producer.send(pos).ok();
    }
  } else {
    for &pos in moves.iter() {
      producer.send(pos).ok();
    }
  }
  let threads_count = config::threads_count();
  let atomic_alpha = AtomicIsize::new(alpha as isize);
  let atomic_best_move = AtomicUsize::new(0);
  let mut guards = Vec::with_capacity(threads_count);
  for _ in 0 .. threads_count {
    let xor_shift_rng = rng.gen::<XorShiftRng>();
    guards.push(thread::scoped(|| {
      let local_consumer = consumer.clone();
      let mut local_field = field.clone();
      let mut local_rng = xor_shift_rng;
      let mut local_empty_board = iter::repeat(0u32).take(field.length()).collect();
      let enemy = player.next();
      while let Some(pos) = local_consumer.recv_async().ok() {
        if should_stop.load(Ordering::Relaxed) {
          debug!(target: MINIMAX_STR, "Time-out!");
          break;
        }
        local_field.put_point(pos, player);
        let next_trajectories_pruning = TrajectoriesPruning::from_last(&mut local_field, enemy, depth - 1, &mut local_empty_board, &mut local_rng, &trajectories_pruning, pos, should_stop);
        if should_stop.load(Ordering::Relaxed) {
          debug!(target: MINIMAX_STR, "Time-out!");
          break;
        }
        let cur_alpha = atomic_alpha.load(Ordering::SeqCst) as i32;
        if cur_alpha >= beta {
          break;
        }
        let mut cur_estimation = -alpha_beta(&mut local_field, depth - 1, pos, enemy, &next_trajectories_pruning, -cur_alpha - 1, -cur_alpha, &mut local_empty_board, &mut local_rng, should_stop);
        if cur_estimation > cur_alpha {
          if !should_stop.load(Ordering::Relaxed) {
            cur_estimation = -alpha_beta(&mut local_field, depth - 1, pos, enemy, &next_trajectories_pruning, -beta, -cur_estimation, &mut local_empty_board, &mut local_rng, should_stop);
          } else {
            debug!(target: MINIMAX_STR, "Time-out! Next estimation ma be approximated.");
          }
        }
        local_field.undo();
        loop {
          let last_pos = atomic_best_move.load(Ordering::SeqCst);
          let last_alpha = atomic_alpha.load(Ordering::SeqCst);
          if cur_estimation > last_alpha as i32 {
            if atomic_alpha.compare_and_swap(last_alpha, cur_estimation as isize, Ordering::SeqCst) == last_alpha && atomic_best_move.compare_and_swap(last_pos, pos, Ordering::SeqCst) == last_pos {
              debug!(target: MINIMAX_STR, "{} for move ({}, {}) is {}.", if cur_estimation < beta { "Estimation" } else { "Lower bound of estimation" }, field.to_x(pos), field.to_y(pos), cur_estimation);
              break;
            }
          } else {
            debug!(target: MINIMAX_STR, "{} for move ({}, {}) is {}.", if cur_estimation > cur_alpha { if cur_estimation < beta { "Estimation" } else { "Lower bound of estimation" } } else { "Upper bound of estimation" }, field.to_x(pos), field.to_y(pos), cur_estimation);
            break;
          }
        }
      }
    }));
  }
  drop(guards);
  let result = atomic_best_move.load(Ordering::SeqCst);
  if result != 0 {
    info!(target: MINIMAX_STR, "Best move is ({}, {}).", field.to_x(result), field.to_y(result));
    *best_move = Some(result);
  } else {
    info!(target: MINIMAX_STR, "Best move is not found.");
    *best_move = None;
  }
  let cur_alpha = atomic_alpha.load(Ordering::SeqCst);
  info!(target: MINIMAX_STR, "Estimation is {}.", cur_alpha);
  cur_alpha as i32
}

pub fn minimax<T: Rng>(field: &mut Field, player: Player, rng: &mut T, depth: u32) -> Option<Pos> {
  info!(target: MINIMAX_STR, "Starting minimax with depth {} and player {}.", depth, player);
  if depth == 0 {
    return None;
  }
  let should_stop = AtomicBool::new(false);
  let mut empty_board = iter::repeat(0u32).take(field.length()).collect();
  let trajectories_pruning = TrajectoriesPruning::new(field, player, depth, &mut empty_board, rng, &should_stop);
  let mut best_move = None;
  info!(target: MINIMAX_STR, "Calculating of our estimation. Player is {}", player);
  let estimation = alpha_beta_parallel(field, player, depth, i32::min_value() + 1, i32::max_value(), &trajectories_pruning, rng, &mut best_move, &should_stop);
  let enemy = player.next();
  let mut enemy_best_move = best_move;
  let enemy_trajectories_pruning = TrajectoriesPruning::dec_exists(&field, enemy, depth - 1, &mut empty_board, rng, &trajectories_pruning, &should_stop);
  info!(target: MINIMAX_STR, "Calculating of enemy estimation with upper bound {}. Player is {}", -estimation + 1, enemy);
  if -alpha_beta_parallel(field, enemy, depth - 1, -estimation, -estimation + 1, &enemy_trajectories_pruning, rng, &mut enemy_best_move, &should_stop) < estimation {
    info!(target: MINIMAX_STR,  "Estimation is greater than enemy estimation. So the best move is {:?}, estimation is {}.", best_move.map(|pos| (field.to_x(pos), field.to_y(pos))), estimation);
    best_move
  } else {
    info!(target: MINIMAX_STR,  "Estimation is less than or equal enemy estimation. So all moves have the same estimation {}.", estimation);
    None
  }
}

pub fn minimax_with_time<T: Rng>(field: &mut Field, player: Player, rng: &mut T, time: u32) -> Option<Pos> {
  let should_stop = AtomicBool::new(false);
  let guard = thread::scoped(|| {
    thread::sleep_ms(time);
    should_stop.store(true, Ordering::Relaxed);
  });
  let enemy = player.next();
  let mut depth = 1;
  let mut best_move = None;
  let mut cur_best_move = None;
  let mut enemy_best_move = None;
  let mut empty_board = iter::repeat(0u32).take(field.length()).collect();
  let mut trajectories_pruning = TrajectoriesPruning::new(field, player, depth, &mut empty_board, rng, &should_stop);
  while !should_stop.load(Ordering::Relaxed) {
    let estimation = alpha_beta_parallel(field, player, depth, i32::min_value() + 1, i32::max_value(), &trajectories_pruning, rng, &mut cur_best_move, &should_stop);
    if should_stop.load(Ordering::Relaxed) { //TODO: use calculated move.
      break;
    }
    let enemy_trajectories_pruning = TrajectoriesPruning::dec_exists(&field, enemy, depth - 1, &mut empty_board, rng, &trajectories_pruning, &should_stop);
    if should_stop.load(Ordering::Relaxed) {
      break;
    }
    best_move = if -alpha_beta_parallel(field, enemy, depth - 1, -estimation, -estimation + 1, &enemy_trajectories_pruning, rng, &mut enemy_best_move, &should_stop) < estimation || should_stop.load(Ordering::Relaxed) {
      cur_best_move
    } else {
      None
    };
    if should_stop.load(Ordering::Relaxed) {
      break;
    }
    depth += 1;
    trajectories_pruning = TrajectoriesPruning::inc_exists(field, player, depth, &mut empty_board, rng, &trajectories_pruning, &should_stop);
  }
  drop(guard);
  best_move
}
