use oppai_common::common::is_last_move_stupid;
use oppai_common::trajectory::{build_trajectories, build_trajectories_from, Trajectory};
use oppai_field::field::wave_diag;
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use std::iter;
use std::sync::atomic::{AtomicBool, Ordering};

fn mark_group(field: &Field, start_pos: Pos, player: Player, empty_board: &mut Vec<u32>) -> Vec<Pos> {
  let mut marks = Vec::new();
  wave_diag(field.width(), start_pos, |pos| {
    if empty_board[pos] != 0 {
      return false;
    }
    let cell = field.cell(pos);
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

fn ladders_rec(
  field: &mut Field,
  player: Player,
  trajectory: &Trajectory,
  empty_board: &mut Vec<u32>,
  should_stop: &AtomicBool,
  depth: usize,
) -> (Pos, i32) {
  match *trajectory.points().as_slice() {
    [pos] => {
      field.put_point(pos, player);
      let cur_score = field.score(player);
      field.undo();
      (pos, cur_score)
    }
    [pos1, pos2] => {
      let mut max_score = field.score(player);
      let mut best_move = 0;

      for &(our_pos, enemy_pos) in &[(pos1, pos2), (pos2, pos1)] {
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
          if cur_score > max_score {
            max_score = cur_score;
            best_move = our_pos;
          }

          field.undo();
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

        if should_stop.load(Ordering::Relaxed) {
          break;
        }

        let marks = mark_group(field, our_pos, player, empty_board);

        for trajectory in trajectories {
          let (_, cur_score) = ladders_rec(field, player, &trajectory, empty_board, should_stop, depth);
          if cur_score > max_score {
            max_score = cur_score;
            best_move = our_pos;
          }
        }

        for pos in marks {
          empty_board[pos] = 0;
        }

        field.undo();
        field.undo();
      }

      (best_move, max_score)
    }
    _ => unreachable!("Trajectory with {} points", trajectory.len()),
  }
}

pub fn ladders(field: &mut Field, player: Player, should_stop: &AtomicBool) -> (Pos, i32) {
  let mut empty_board = iter::repeat(0u32).take(field.length()).collect();

  let trajectories = build_trajectories(field, player, 2, &mut empty_board, &should_stop);

  let mut max_score = field.score(player);
  let mut best_move = 0;

  for trajectory in trajectories {
    if should_stop.load(Ordering::Relaxed) {
      break;
    }

    let marks = if let [pos1, _] = *trajectory.points().as_slice() {
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

    let (cur_pos, cur_score) = ladders_rec(field, player, &trajectory, &mut empty_board, should_stop, 0);
    let cur_score = cur_score.min(trajectory.score());
    if cur_score > max_score {
      max_score = cur_score;
      best_move = cur_pos;
    }

    for pos in marks {
      empty_board[pos] = 0;
    }
  }

  (best_move, max_score)
}
