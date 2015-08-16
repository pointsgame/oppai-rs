use std::{iter, thread};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use comm;
use config;
use player::Player;
use field::{Pos, Field};
use trajectories_pruning::TrajectoriesPruning;
use common;

fn alpha_beta(field: &mut Field, depth: u32, last_pos: Pos, player: Player, trajectories_pruning: &TrajectoriesPruning, mut alpha: i32, beta: i32, empty_board: &mut Vec<u32>) -> i32 {
  let enemy = player.next();
  if common::is_last_move_stupid(field, last_pos, enemy) {
    return i32::max_value();
  }
  if depth == 0 {
    return field.score(player);
  }
  let moves = trajectories_pruning.calculate_moves(empty_board);
  if moves.is_empty() {
    return field.score(player);
  }
  for pos in moves {
    field.put_point(pos, player);
    let next_trajectories_pruning = TrajectoriesPruning::new_from_last(field, enemy, depth - 1, empty_board, trajectories_pruning, pos);
    let mut cur_estimation = -alpha_beta(field, depth - 1, pos, enemy, &next_trajectories_pruning, -alpha - 1, -alpha, empty_board);
    if cur_estimation > alpha && cur_estimation < beta {
      cur_estimation = -alpha_beta(field, depth - 1, pos, enemy, &next_trajectories_pruning, -beta, -cur_estimation, empty_board);
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

pub fn minimax(field: &mut Field, player: Player, depth: u32) -> Option<Pos> {
  if depth == 0 {
    return None;
  }
  let mut empty_board = iter::repeat(0u32).take(field.length()).collect();
  let trajectories_pruning = TrajectoriesPruning::new(field, player, depth, &mut empty_board);
  let moves = trajectories_pruning.calculate_moves(&mut empty_board);
  let (producer, consumer) = comm::spmc::unbounded::new();
  for pos in moves {
    producer.send(pos).ok();
  }
  let threads_count = config::threads_count();
  let alpha = AtomicIsize::new((i32::min_value() + 1) as isize);
  let best_move = AtomicUsize::new(0);
  let mut guards = Vec::with_capacity(threads_count);
  for _ in 0 .. threads_count {
    guards.push(thread::scoped(|| {
      let local_consumer = consumer.clone();
      let mut local_field = field.clone();
      let mut local_empty_board = iter::repeat(0u32).take(field.length()).collect();
      let enemy = player.next();
      while let Some(pos) = local_consumer.recv_async().ok() {
        local_field.put_point(pos, player);
        let next_trajectories_pruning = TrajectoriesPruning::new_from_last(&mut local_field, enemy, depth - 1, &mut local_empty_board, &trajectories_pruning, pos);
        let cur_alpha = alpha.load(Ordering::SeqCst) as i32;
        let mut cur_estimation = -alpha_beta(&mut local_field, depth - 1, pos, enemy, &next_trajectories_pruning, -cur_alpha - 1, -cur_alpha, &mut local_empty_board);
        if cur_estimation > cur_alpha {
          cur_estimation = -alpha_beta(&mut local_field, depth - 1, pos, enemy, &next_trajectories_pruning, i32::min_value() + 1, -cur_estimation, &mut local_empty_board);
        }
        local_field.undo();
        loop {
          let last_pos = best_move.load(Ordering::SeqCst);
          let last_alpha = alpha.load(Ordering::SeqCst);
          if cur_estimation > last_alpha as i32 {
            if alpha.compare_and_swap(last_alpha, cur_estimation as isize, Ordering::SeqCst) == last_alpha && best_move.compare_and_swap(last_pos, pos, Ordering::SeqCst) == last_pos {
              break;
            }
          } else {
            break;
          }
        }
      }
    }));
  }
  drop(guards);
  let result = best_move.load(Ordering::SeqCst);
  if result != 0 {
    Some(result)
  } else {
    None
  }
}
