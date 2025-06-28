use oppai_common::common::is_last_move_stupid;
use oppai_common::trajectory::{Trajectory, build_trajectories, build_trajectories_from};
use oppai_field::field::wave_diag;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use std::iter;

fn mark_group(field: &mut Field, start_pos: Pos, player: Player, empty_board: &mut [u32]) -> Vec<Pos> {
  let mut marks = Vec::new();
  wave_diag(&mut field.q, field.width, start_pos, |pos| {
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
  marks
}

fn collect_near_moves(field: &Field, player: Player, empty_board: &mut [u32]) -> Vec<Pos> {
  let mut moves = Vec::new();
  for &pos in field
    .moves
    .iter()
    .filter(|&&pos| field.cell(pos).is_players_point(player))
  {
    for &near_pos in field.directions(pos).iter() {
      if empty_board[near_pos] == 0 && field.is_putting_allowed(near_pos) {
        moves.push(near_pos);
        empty_board[near_pos] = 1;
      }
    }
  }
  for &pos in &moves {
    empty_board[pos] = 0;
  }
  moves
}

fn is_trajectoty_alive(field: &mut Field, trajectory: &Trajectory<2>, player: Player, empty_board: &mut [u32]) -> bool {
  if trajectory.points.iter().any(|&pos| !field.is_putting_allowed(pos)) {
    return false;
  }

  for &pos in &trajectory.points {
    field.put_point(pos, player);
  }

  let result = field.get_delta_score(player) > 0 || {
    let enemy = player.next();
    let moves = collect_near_moves(field, enemy, empty_board);
    moves.into_iter().any(|pos| {
      field.put_point(pos, player);
      let result = field.get_delta_score(player) > 0
        && trajectory
          .points
          .iter()
          .all(|&trajectory_pos| field.cell(trajectory_pos).is_bound());
      field.undo();
      result && {
        let enemies_around = field
          .directions(pos)
          .iter()
          .filter(|&&pos| field.cell(pos).is_players_point(enemy))
          .count();
        enemies_around < 3
          || enemies_around == 3
            && field
              .directions(pos)
              .iter()
              .all(|&near_pos| !field.cell(near_pos).is_putting_allowed())
      }
    })
  };

  for _ in 0..trajectory.points.len() {
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
  let moves = collect_near_moves(field, player, empty_board);
  let enemy = player.next();
  moves.into_iter().all(|enemy_pos| {
    field.put_point(enemy_pos, enemy);

    if field.get_delta_score(enemy) == 0 {
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
) -> (Option<NonZeroPos>, i32, u32) {
  match *trajectory.points.as_slice() {
    [pos] => {
      field.put_point(pos, player);
      let cur_score = field.score(player);
      field.undo();
      (NonZeroPos::new(pos), cur_score, depth)
    }
    [pos1, pos2] => {
      let mut best_move = None;
      let mut capture_depth = 0;

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
            field.to_x(our_pos),
            field.to_y(our_pos),
            field,
          );
        }

        let trajectories = build_trajectories_from(field, our_pos, player, 2, empty_board, should_stop);

        if should_stop() {
          field.undo();
          field.undo();
          break;
        }

        let marks = mark_group(field, our_pos, player, empty_board);

        for trajectory in trajectories {
          if alpha >= beta || should_stop() {
            break;
          }

          if trajectory.score <= alpha {
            continue;
          }

          let (_, cur_score, cur_capture_depth) = ladders_rec(
            field,
            player,
            &trajectory,
            alpha,
            beta.min(trajectory.score),
            empty_board,
            should_stop,
            depth + 1,
          );
          let cur_score = cur_score.min(trajectory.score);

          if cur_score > alpha && is_trajectoty_viable(field, &trajectory, player, empty_board) {
            alpha = cur_score;
            best_move = NonZeroPos::new(our_pos);
            capture_depth = cur_capture_depth;
          }
        }

        for pos in marks {
          empty_board[pos] = 0;
        }

        field.undo();
        field.undo();
      }

      (best_move, alpha, capture_depth)
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

  let mut trajectories = build_trajectories(field, player, 2, &mut empty_board, should_stop);
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

    let marks = if let [pos1, _] = *trajectory.points.as_slice() {
      if let Some(&pos) = field
        .directions_diag(pos1)
        .iter()
        .find(|&&pos| field.cell(pos).is_players_point(player))
      {
        // mark one of near groups to not search trajectories from it
        mark_group(field, pos, player, &mut empty_board)
      } else {
        Vec::new()
      }
    } else {
      Vec::new()
    };

    let (cur_pos, cur_score, cur_capture_depth) = ladders_rec(
      field,
      player,
      &trajectory,
      alpha,
      trajectory.score,
      &mut empty_board,
      should_stop,
      0,
    );
    let cur_score = cur_score.min(trajectory.score);
    if cur_score > alpha && is_trajectoty_viable(field, &trajectory, player, &mut empty_board) {
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
