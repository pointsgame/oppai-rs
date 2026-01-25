use oppai_common::common::is_last_move_stupid;
use oppai_common::trajectory::{Trajectory, build_trajectories, build_trajectories_from};
use oppai_field::field::wave_diag;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;

fn mark_group(field: &mut Field, start_pos: Pos, player: Player, marks: &mut Vec<Pos>) {
  wave_diag(&mut field.q, field.stride, start_pos, |pos| {
    if field.points[pos].is_tagged_2() {
      return false;
    }
    let cell = field.points[pos];
    if cell.is_players_point(player) {
      field.points[pos].set_tag_2();
      marks.push(pos);
      true
    } else {
      false
    }
  });
}

fn collect_near_moves(field: &mut Field, player: Player) -> Vec<Pos> {
  let mut moves = Vec::new();
  for &pos in &field.moves {
    if field.points[pos].is_players_point(player) {
      for &near_pos in field.directions(pos).iter() {
        if !field.points[near_pos].is_tagged_2() && field.is_putting_allowed(near_pos) {
          moves.push(near_pos);
          field.points[near_pos].set_tag_2();
        }
      }
    }
  }
  for &pos in &moves {
    field.points[pos].clear_tag_2();
  }
  moves
}

fn is_trajectoty_alive(field: &mut Field, trajectory: &Trajectory<2>, player: Player) -> bool {
  if trajectory.points.iter().any(|&pos| !field.is_putting_allowed(pos)) {
    return false;
  }

  for &pos in &trajectory.points {
    field.put_point(pos, player);
  }

  let result = field.get_delta_score(player) > 0 || {
    let enemy = player.next();
    let moves = collect_near_moves(field, enemy);
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

fn is_trajectoty_viable(field: &mut Field, trajectory: &Trajectory<2>, player: Player) -> bool {
  let moves = collect_near_moves(field, player);
  let enemy = player.next();
  moves.into_iter().all(|enemy_pos| {
    field.put_point(enemy_pos, enemy);

    if field.get_delta_score(enemy) == 0 {
      field.undo();
      return true;
    }

    let result = is_trajectoty_alive(field, trajectory, player);

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
  should_stop: &SS,
  depth: u32,
  marks: &mut Vec<Pos>,
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

        let trajectories = build_trajectories_from(field, our_pos, player, 2, should_stop);

        if should_stop() {
          field.undo();
          field.undo();
          break;
        }

        let marks_len = marks.len();
        mark_group(field, our_pos, player, marks);

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
            should_stop,
            depth + 1,
            marks,
          );
          let cur_score = cur_score.min(trajectory.score);

          if cur_score > alpha && is_trajectoty_viable(field, &trajectory, player) {
            alpha = cur_score;
            best_move = NonZeroPos::new(our_pos);
            capture_depth = cur_capture_depth;
          }
        }

        for &pos in &marks[marks_len..] {
          field.points[pos].clear_tag_2();
        }
        marks.truncate(marks_len);

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
  let mut trajectories = build_trajectories(field, player, 2, should_stop);
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
      mark_group(field, pos, player, &mut marks);
    };

    let (cur_pos, cur_score, cur_capture_depth) = ladders_rec(
      field,
      player,
      &trajectory,
      alpha,
      trajectory.score,
      should_stop,
      0,
      &mut marks,
    );
    let cur_score = cur_score.min(trajectory.score);
    if cur_score > alpha && is_trajectoty_viable(field, &trajectory, player) {
      alpha = cur_score;
      capture_depth = cur_capture_depth;
      best_move = cur_pos;
    }

    for pos in marks {
      field.points[pos].clear_tag_2();
    }
  }

  (best_move, alpha, capture_depth)
}
