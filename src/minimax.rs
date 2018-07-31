use common;
use config::{self, MinimaxType};
use crossbeam::{self, queue::MsQueue};
use field::{Field, Pos};
use hash_table::{HashData, HashTable, HashType};
use player::Player;
use rand::{Rng, SeedableRng, XorShiftRng};
use std::{
  iter,
  sync::atomic::{AtomicBool, AtomicIsize, Ordering},
  thread,
  time::Duration,
};
use trajectories_pruning::TrajectoriesPruning;

const MINIMAX_STR: &str = "minimax";

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

fn alpha_beta<T: Rng>(
  field: &mut Field,
  depth: u32,
  last_pos: Pos,
  player: Player,
  trajectories_pruning: &TrajectoriesPruning,
  alpha: i32,
  beta: i32,
  empty_board: &mut Vec<u32>,
  hash_table: &HashTable,
  rng: &mut T,
  should_stop: &AtomicBool,
) -> i32 {
  if should_stop.load(Ordering::Relaxed) {
    return alpha;
  }
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
  // Try the best move from the hash table.
  if let Some(hash_pos) = hash_pos_option {
    field.put_point(hash_pos, player);
    if common::is_penult_move_stuped(field) {
      field.undo();
      return i32::max_value();
    }
    let next_trajectories_pruning = TrajectoriesPruning::from_last(
      field,
      enemy,
      depth - 1,
      empty_board,
      rng,
      trajectories_pruning,
      hash_pos,
      should_stop,
    );
    let cur_estimation = -alpha_beta(
      field,
      depth - 1,
      hash_pos,
      enemy,
      &next_trajectories_pruning,
      -beta,
      -cur_alpha,
      empty_board,
      hash_table,
      rng,
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
      put_new_hash_value(
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
    if common::is_penult_move_stuped(field) {
      field.undo();
      return i32::max_value();
    }
    let next_trajectories_pruning = TrajectoriesPruning::from_last(
      field,
      enemy,
      depth - 1,
      empty_board,
      rng,
      trajectories_pruning,
      pos,
      should_stop,
    );
    let mut cur_estimation = -alpha_beta(
      // TODO: check if cur_alpha is -Inf
      field,
      depth - 1,
      pos,
      enemy,
      &next_trajectories_pruning,
      -cur_alpha - 1,
      -cur_alpha,
      empty_board,
      hash_table,
      rng,
      should_stop,
    );
    if cur_estimation > cur_alpha && cur_estimation < beta {
      cur_estimation = -alpha_beta(
        field,
        depth - 1,
        pos,
        enemy,
        &next_trajectories_pruning,
        -beta,
        -cur_estimation,
        empty_board,
        hash_table,
        rng,
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
      put_new_hash_value(hash_table, field.colored_hash(player), pos, depth, cur_estimation, beta);
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

pub fn alpha_beta_parallel<T: Rng>(
  field: &mut Field,
  player: Player,
  depth: u32,
  alpha: i32,
  beta: i32,
  trajectories_pruning: &TrajectoriesPruning,
  hash_table: &HashTable,
  rng: &mut T,
  best_move: &mut Option<Pos>,
  should_stop: &AtomicBool,
) -> i32 {
  info!(
    target: MINIMAX_STR,
    "Starting parellel alpha beta with depth {}, player {} and beta {}.",
    depth,
    player,
    beta
  );
  if depth == 0 || should_stop.load(Ordering::Relaxed) {
    *best_move = None;
    return field.score(player);
  }
  let moves = trajectories_pruning.moves();
  debug!(
    target: MINIMAX_STR,
    "Moves in consideration: {:?}.",
    moves
      .iter()
      .map(|&pos| (field.to_x(pos), field.to_y(pos)))
      .collect::<Vec<(u32, u32)>>()
  );
  if moves.is_empty() || should_stop.load(Ordering::Relaxed) {
    *best_move = None;
    return field.score(player);
  }
  let queue = MsQueue::new();
  if let Some(best_pos) = *best_move {
    queue.push(best_pos);
    for &pos in moves.iter().filter(|&&pos| pos != best_pos) {
      queue.push(pos);
    }
  } else {
    for &pos in moves.iter() {
      queue.push(pos);
    }
  }
  let threads_count = config::threads_count();
  let atomic_alpha = AtomicIsize::new(alpha as isize);
  let best_moves = MsQueue::new();
  crossbeam::scope(|scope| {
    for _ in 0 .. threads_count {
      let xor_shift_rng = XorShiftRng::from_seed(rng.gen());
      scope.spawn(|| {
        let mut local_field = field.clone();
        let mut local_rng = xor_shift_rng;
        let mut local_empty_board = iter::repeat(0u32).take(field.length()).collect();
        let mut local_best_move = 0;
        let mut local_alpha = alpha;
        let enemy = player.next();
        while let Some(pos) = queue.try_pop() {
          if should_stop.load(Ordering::Relaxed) {
            break;
          }
          local_field.put_point(pos, player);
          let next_trajectories_pruning = TrajectoriesPruning::from_last(
            &mut local_field,
            enemy,
            depth - 1,
            &mut local_empty_board,
            &mut local_rng,
            trajectories_pruning,
            pos,
            should_stop,
          );
          if should_stop.load(Ordering::Relaxed) {
            break;
          }
          let cur_alpha = atomic_alpha.load(Ordering::Relaxed) as i32;
          if cur_alpha >= beta {
            break;
          }
          let mut cur_estimation = -alpha_beta(
            &mut local_field,
            depth - 1,
            pos,
            enemy,
            &next_trajectories_pruning,
            -cur_alpha - 1,
            -cur_alpha,
            &mut local_empty_board,
            hash_table,
            &mut local_rng,
            should_stop,
          );
          if should_stop.load(Ordering::Relaxed) {
            break;
          }
          if cur_estimation > cur_alpha && cur_estimation < beta {
            cur_estimation = -alpha_beta(
              &mut local_field,
              depth - 1,
              pos,
              enemy,
              &next_trajectories_pruning,
              -beta,
              -cur_estimation,
              &mut local_empty_board,
              hash_table,
              &mut local_rng,
              should_stop,
            );
          }
          // We should check it before the best move assignment because it's possible
          // that current estimation is higher than real in case of time out.
          if should_stop.load(Ordering::Relaxed) {
            break;
          }
          debug!(
            target: MINIMAX_STR,
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
            let last_alpha = atomic_alpha.load(Ordering::Relaxed);
            if cur_estimation <= last_alpha as i32
              || atomic_alpha.compare_and_swap(last_alpha, cur_estimation as isize, Ordering::Relaxed) == last_alpha
            {
              break;
            }
          }
        }
        if local_best_move != 0 && local_alpha == atomic_alpha.load(Ordering::Relaxed) as i32 {
          best_moves.push((local_best_move, local_alpha));
        }
      });
    }
  });
  let mut result = 0;
  let best_alpha = atomic_alpha.load(Ordering::SeqCst) as i32;
  while let Some((pos, pos_alpha)) = best_moves.try_pop() {
    if pos_alpha == best_alpha {
      result = pos;
      break;
    }
  }
  if result == 0 {
    info!(target: MINIMAX_STR, "Best move is not found.");
    *best_move = None;
  } else {
    info!(
      target: MINIMAX_STR,
      "Best move is ({}, {}).",
      field.to_x(result),
      field.to_y(result)
    );
    *best_move = Some(result);
  }
  info!(target: MINIMAX_STR, "Estimation is {}.", best_alpha);
  best_alpha
}

fn mtdf<T: Rng>(
  field: &mut Field,
  player: Player,
  trajectories_pruning: &TrajectoriesPruning,
  hash_table: &HashTable,
  rng: &mut T,
  depth: u32,
  best_move: &mut Option<Pos>,
  should_stop: &AtomicBool,
) -> i32 {
  let mut alpha = 0;
  let mut beta = 0;
  for &pos in field.points_seq() {
    if field.cell(pos).get_player() == player {
      alpha -= 1;
    } else {
      beta += 1;
    }
  }
  while alpha != beta {
    if should_stop.load(Ordering::Relaxed) {
      *best_move = None;
      return alpha;
    }
    let mut cur_best_move = *best_move;
    let center = if (alpha + beta) % 2 == -1 {
      (alpha + beta) / 2 - 1
    } else {
      (alpha + beta) / 2
    };
    let cur_estimation = alpha_beta_parallel(
      field,
      player,
      depth,
      center,
      center + 1,
      trajectories_pruning,
      hash_table,
      rng,
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

fn nega_scout<T: Rng>(
  field: &mut Field,
  player: Player,
  trajectories_pruning: &TrajectoriesPruning,
  hash_table: &HashTable,
  rng: &mut T,
  depth: u32,
  best_move: &mut Option<Pos>,
  should_stop: &AtomicBool,
) -> i32 {
  alpha_beta_parallel(
    field,
    player,
    depth,
    i32::min_value() + 1,
    i32::max_value(),
    trajectories_pruning,
    hash_table,
    rng,
    best_move,
    should_stop,
  )
}

pub fn minimax<T: Rng>(
  field: &mut Field,
  player: Player,
  hash_table: &HashTable,
  rng: &mut T,
  depth: u32,
) -> Option<Pos> {
  info!(
    target: MINIMAX_STR,
    "Starting minimax with depth {} and player {}.",
    depth,
    player
  );
  if depth == 0 {
    return None;
  }
  let should_stop = AtomicBool::new(false);
  let mut empty_board = iter::repeat(0u32).take(field.length()).collect();
  let trajectories_pruning = TrajectoriesPruning::new(field, player, depth, &mut empty_board, rng, &should_stop);
  let mut best_move = None;
  info!(
    target: MINIMAX_STR,
    "Calculating of our estimation. Player is {}",
    player
  );
  let minimax_function = match config::minimax_type() {
    MinimaxType::NegaScout => nega_scout,
    MinimaxType::MTDF => mtdf,
  };
  let estimation = minimax_function(
    field,
    player,
    &trajectories_pruning,
    hash_table,
    rng,
    depth,
    &mut best_move,
    &should_stop,
  );
  let enemy = player.next();
  let mut enemy_best_move = best_move;
  let enemy_trajectories_pruning = TrajectoriesPruning::dec_and_swap_exists(
    field,
    depth - 1,
    &mut empty_board,
    rng,
    &trajectories_pruning,
    &should_stop,
  );
  info!(
    target: MINIMAX_STR,
    "Calculating of enemy estimation with upper bound {}. Player is {}",
    -estimation + 1,
    enemy
  );
  // Check if we could lose something if we don't make the current best move.
  // If we couldn't that means that the current best move is just a random move.
  if -alpha_beta_parallel(
    field,
    enemy,
    depth - 1,
    -estimation,
    -estimation + 1,
    &enemy_trajectories_pruning,
    hash_table,
    rng,
    &mut enemy_best_move,
    &should_stop,
  ) < estimation
  {
    info!(
      target: MINIMAX_STR,
      "Estimation is greater than enemy estimation. So the best move is {:?}, estimation is {}.",
      best_move.map(|pos| (field.to_x(pos), field.to_y(pos))),
      estimation
    );
    best_move
  } else {
    info!(
      target: MINIMAX_STR,
      "Estimation is less than or equal enemy estimation. So all moves have the same estimation {}.",
      estimation
    );
    None
  }
}

pub fn minimax_with_time<T: Rng>(
  field: &mut Field,
  player: Player,
  hash_table: &HashTable,
  rng: &mut T,
  time: u32,
) -> Option<Pos> {
  let should_stop = AtomicBool::new(false);
  crossbeam::scope(|scope| {
    scope.spawn(|| {
      thread::sleep(Duration::from_millis(u64::from(time)));
      debug!(target: MINIMAX_STR, "Time-out!");
      should_stop.store(true, Ordering::Relaxed);
    });
    let enemy = player.next();
    let mut depth = 1;
    let mut best_move = None;
    let mut cur_best_move = None;
    let mut enemy_best_move = None;
    let mut empty_board = iter::repeat(0u32).take(field.length()).collect();
    let mut trajectories_pruning = TrajectoriesPruning::new(field, player, depth, &mut empty_board, rng, &should_stop);
    let minimax_function = match config::minimax_type() {
      MinimaxType::NegaScout => nega_scout,
      MinimaxType::MTDF => mtdf,
    };
    while !should_stop.load(Ordering::Relaxed) {
      let estimation = minimax_function(
        field,
        player,
        &trajectories_pruning,
        hash_table,
        rng,
        depth,
        &mut cur_best_move,
        &should_stop,
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
      let enemy_trajectories_pruning = TrajectoriesPruning::dec_and_swap_exists(
        field,
        depth - 1,
        &mut empty_board,
        rng,
        &trajectories_pruning,
        &should_stop,
      );
      if should_stop.load(Ordering::Relaxed) {
        // See previous comment.
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
        || -alpha_beta_parallel(
          field,
          enemy,
          depth - 1,
          -estimation,
          -estimation + 1,
          &enemy_trajectories_pruning,
          hash_table,
          rng,
          &mut enemy_best_move,
          &should_stop,
        ) < estimation
      {
        cur_best_move
      } else {
        None
      };
      if should_stop.load(Ordering::Relaxed) {
        break;
      }
      depth += 1;
      trajectories_pruning = TrajectoriesPruning::inc_exists(
        field,
        player,
        depth,
        &mut empty_board,
        rng,
        &trajectories_pruning,
        &should_stop,
      );
    }
    best_move
  })
}
