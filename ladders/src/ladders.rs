use oppai_common::common::is_last_move_stupid;
use oppai_common::trajectory::{Trajectory, build_trajectories, build_trajectories_from};
use oppai_field::field::wave_diag;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::iter;

fn mark_group(field: &mut Field, start_pos: Pos, player: Player, empty_board: &mut [u32], marks: &mut Vec<Pos>) {
  wave_diag(&mut field.q, field.stride, start_pos, |pos| {
    if empty_board[pos] != 0 {
      return false;
    }
    let cell = field.points[pos];
    if cell.is_players_point(player) {
      empty_board[pos] = 1;
      marks.push(pos);
      true
    } else {
      false
    }
  });
}

fn collect_near_moves(field: &Field, player: Player, empty_board: &mut [u32]) -> Vec<Pos> {
  let mut moves = Vec::new();
  for &pos in &field.moves {
    if field.points[pos].is_players_point(player) {
      for &near_pos in field.directions_diag(pos).iter() {
        if empty_board[near_pos] == 0 && field.is_putting_allowed(near_pos) {
          moves.push(near_pos);
          empty_board[near_pos] = 1;
        }
      }
    }
  }
  for &pos in &moves {
    empty_board[pos] = 0;
  }
  moves
}

fn is_trajectoty_alive(field: &mut Field, trajectory: &Trajectory<2>, player: Player, empty_board: &mut [u32]) -> bool {
  let mut put = 0;
  for &pos in &trajectory.points {
    if field.put_point(pos, player) {
      put += 1;
    }
  }

  if put == 0 {
    return false;
  }

  let result = field.get_delta_score(player) > 0 || {
    let moves = collect_near_moves(field, player, empty_board);
    moves.into_iter().any(|pos| {
      field.put_point(pos, player);
      if field.get_delta_score(player) <= 0
        || trajectory
          .points
          .iter()
          .any(|&trajectory_pos| !field.cell(trajectory_pos).is_bound())
      {
        field.undo();
        return false;
      }
      for _ in 0..trajectory.points.len() + 1 {
        field.undo();
      }
      field.put_point(pos, player);
      let result = is_trajectoty_viable(field, trajectory, player, empty_board);
      field.undo();
      for &pos in &trajectory.points {
        field.put_point(pos, player);
      }
      result
    })
  };

  for _ in 0..put {
    field.undo();
  }

  result
}

fn is_trajectoty_viable(
  field: &mut Field,
  trajectory: &Trajectory<2>,
  player: Player,
  empty_board: &mut [u32],
) -> bool {
  if trajectory.points.len() == 1 {
    return true;
  }

  let enemy = player.next();
  let moves = collect_near_moves(field, enemy, empty_board);
  moves.into_iter().all(|enemy_pos| {
    if trajectory.points.contains(&enemy_pos) {
      return true;
    }

    field.put_point(enemy_pos, enemy);

    if field.get_delta_score(enemy) <= 0 {
      field.undo();
      return true;
    }

    let result = is_trajectoty_alive(field, trajectory, player, empty_board);

    field.undo();

    result
  })
}

fn ladders_rec<SS: Fn() -> bool>(
  field: &mut Field,
  player: Player,
  trajectory: &Trajectory<2>,
  mut alpha: i32,
  beta: i32,
  empty_board: &mut Vec<u32>,
  should_stop: &SS,
  depth: u32,
  marks: &mut Vec<Pos>,
) -> (Option<NonZeroPos>, i32, u32, bool) {
  match *trajectory.points.as_slice() {
    [pos] => {
      field.put_point(pos, player);
      let cur_score = field.score(player);
      field.undo();
      (NonZeroPos::new(pos), cur_score, depth, true)
    }
    [pos1, pos2] => {
      let mut best_move = None;
      let mut capture_depth = 0;
      let mut viable = false;

      for &(our_pos, enemy_pos) in &[(pos1, pos2), (pos2, pos1)] {
        if trajectory.score <= alpha || alpha >= beta || should_stop() {
          break;
        }

        if field.cell(our_pos).is_players_empty_base(player.next()) {
          continue;
        }

        if !field.put_point(our_pos, player) {
          panic!(
            "Failed to put a point to ({}, {}) on the field:\n{}",
            field.to_x(our_pos),
            field.to_y(our_pos),
            field,
          );
        }

        if is_last_move_stupid(field, our_pos, player) {
          field.undo();
          continue;
        }

        if field.get_delta_score(player) > 0 {
          let cur_score = field.score(player);
          field.undo();
          if cur_score > alpha {
            alpha = cur_score;
            best_move = NonZeroPos::new(our_pos);
            capture_depth = depth;
          }
          continue;
        }

        if !field.put_point(enemy_pos, player.next()) {
          panic!(
            "Failed to put a point to ({}, {}) on the field:\n{}",
            field.to_x(enemy_pos),
            field.to_y(enemy_pos),
            field,
          );
        }

        let trajectories: SmallVec<[_; 2]> =
          build_trajectories_from(field, our_pos, player, 2, empty_board, should_stop);

        if should_stop() {
          field.undo();
          field.undo();
          break;
        }

        let marks_len = marks.len();
        mark_group(field, our_pos, player, empty_board, marks);

        for trajectory in trajectories {
          if alpha >= beta || should_stop() {
            break;
          }

          if trajectory.score <= alpha {
            continue;
          }

          let (_, cur_score, cur_capture_depth, cur_viable) = ladders_rec(
            field,
            player,
            &trajectory,
            alpha,
            beta.min(trajectory.score),
            empty_board,
            should_stop,
            depth + 1,
            marks,
          );
          let cur_score = cur_score.min(trajectory.score);

          if cur_score > alpha && cur_viable {
            viable = trajectory.points.len() > 1;
            alpha = cur_score;
            best_move = NonZeroPos::new(our_pos);
            capture_depth = cur_capture_depth;
          }
        }

        for &pos in &marks[marks_len..] {
          empty_board[pos] = 0;
        }
        marks.truncate(marks_len);

        field.undo();
        field.undo();
      }

      (
        best_move,
        alpha,
        capture_depth,
        best_move.is_some() && (viable || is_trajectoty_viable(field, trajectory, player, empty_board)),
      )
    }
    _ => unreachable!("Trajectory with {} points", trajectory.points.len()),
  }
}

pub fn ladders<SS: Fn() -> bool>(
  field: &mut Field,
  player: Player,
  should_stop: &SS,
) -> (Option<NonZeroPos>, i32, u32) {
  let mut empty_board = iter::repeat_n(0u32, field.length()).collect::<Vec<_>>();

  let mut trajectories: SmallVec<[_; 8]> = build_trajectories(field, player, 2, &mut empty_board, should_stop);
  trajectories.sort_unstable_by_key(|trajectory| -trajectory.score);

  info!("Solving ladders for {} trajectories.", trajectories.len());

  let mut alpha = field.score(player);
  let mut capture_depth = 0;
  let mut best_move = None;

  for trajectory in trajectories {
    if should_stop() {
      break;
    }

    if trajectory.score <= alpha {
      continue;
    }

    let mut marks = Vec::with_capacity(field.length());
    if let [pos1, _] = *trajectory.points.as_slice()
      && let Some(&pos) = field
        .directions_diag(pos1)
        .iter()
        .find(|&&pos| field.cell(pos).is_players_point(player))
    {
      // mark one of near groups to not search trajectories from it
      mark_group(field, pos, player, &mut empty_board, &mut marks);
    };

    let (cur_pos, cur_score, cur_capture_depth, cur_viable) = ladders_rec(
      field,
      player,
      &trajectory,
      alpha,
      trajectory.score,
      &mut empty_board,
      should_stop,
      0,
      &mut marks,
    );
    let cur_score = cur_score.min(trajectory.score);
    if cur_score > alpha && cur_viable {
      alpha = cur_score;
      capture_depth = cur_capture_depth;
      best_move = cur_pos;
    }

    for pos in marks {
      empty_board[pos] = 0;
    }
  }

  (best_move, alpha, capture_depth)
}

pub fn ladder_moves<SS: Fn() -> bool>(field: &mut Field, player: Player, should_stop: &SS) -> HashSet<Pos> {
  let mut empty_board = iter::repeat_n(0u32, field.length()).collect::<Vec<_>>();

  let trajectories: SmallVec<[Trajectory<2>; 8]> = build_trajectories(field, player, 2, &mut empty_board, should_stop);

  let alpha = field.score(player);

  let mut moves = HashSet::new();

  for trajectory in trajectories {
    if should_stop() {
      break;
    }

    match *trajectory.points.as_slice() {
      [pos] => {
        moves.insert(pos);
        continue;
      }
      [pos1, pos2] => {
        let mut marks = Vec::with_capacity(field.length());
        if let Some(&pos) = field
          .directions_diag(pos1)
          .iter()
          .find(|&&pos| field.cell(pos).is_players_point(player))
        {
          // mark one of near groups to not search trajectories from it
          mark_group(field, pos, player, &mut empty_board, &mut marks);
        };

        for &(our_pos, enemy_pos) in &[(pos1, pos2), (pos2, pos1)] {
          if should_stop() {
            break;
          }

          if field.cell(our_pos).is_players_empty_base(player.next()) {
            continue;
          }

          if !field.put_point(our_pos, player) {
            panic!(
              "Failed to put a point to ({}, {}) on the field:\n{}",
              field.to_x(our_pos),
              field.to_y(our_pos),
              field,
            );
          }

          if is_last_move_stupid(field, our_pos, player) {
            field.undo();
            continue;
          }

          if field.get_delta_score(player) > 0 {
            field.undo();
            moves.insert(our_pos);
            continue;
          }

          if !field.put_point(enemy_pos, player.next()) {
            panic!(
              "Failed to put a point to ({}, {}) on the field:\n{}",
              field.to_x(enemy_pos),
              field.to_y(enemy_pos),
              field,
            );
          }

          let trajectories: SmallVec<[_; 2]> =
            build_trajectories_from(field, our_pos, player, 2, &mut empty_board, should_stop);

          if should_stop() {
            field.undo();
            field.undo();
            break;
          }

          let marks_len = marks.len();
          mark_group(field, our_pos, player, &mut empty_board, &mut marks);

          for next_trajectory in trajectories {
            if should_stop() {
              break;
            }

            let (_, cur_score, _, cur_viable) = ladders_rec(
              field,
              player,
              &next_trajectory,
              alpha,
              trajectory.score.min(next_trajectory.score),
              &mut empty_board,
              should_stop,
              0,
              &mut marks,
            );
            let cur_score = cur_score.min(next_trajectory.score);

            if cur_score > alpha
              && cur_viable
              && (next_trajectory.points.len() > 1
                || is_trajectoty_viable(field, &trajectory, player, &mut empty_board))
            {
              moves.insert(our_pos);
              break;
            }
          }

          for &pos in &marks[marks_len..] {
            empty_board[pos] = 0;
          }
          marks.truncate(marks_len);

          field.undo();
          field.undo();
        }

        for pos in marks {
          empty_board[pos] = 0;
        }
      }
      _ => unreachable!("Trajectory with {} points", trajectory.points.len()),
    }
  }

  moves
}
