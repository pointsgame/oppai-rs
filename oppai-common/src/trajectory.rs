use oppai_field::field::{euclidean, wave_diag, Field, Pos};
use oppai_field::player::Player;
use std::{
  ops::Index,
  sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone)]
pub struct Trajectory {
  points: Vec<Pos>,
  hash: u64,
  score: i32,
}

impl Trajectory {
  pub fn new(points: Vec<Pos>, hash: u64, score: i32) -> Trajectory {
    Trajectory { points, hash, score }
  }

  pub fn points(&self) -> &Vec<Pos> {
    &self.points
  }

  pub fn hash(&self) -> u64 {
    self.hash
  }

  pub fn score(&self) -> i32 {
    self.score
  }

  pub fn len(&self) -> usize {
    self.points.len()
  }

  pub fn is_empty(&self) -> bool {
    self.points.is_empty()
  }
}

fn add_trajectory(field: &Field, trajectories: &mut Vec<Trajectory>, points: &[Pos], player: Player) {
  for &pos in points {
    if !field.cell(pos).is_bound() || field.number_near_groups(pos, player) < 2 {
      return;
    }
  }
  let zobrist = field.zobrist();
  let mut hash = 0u64;
  for &pos in points {
    hash ^= zobrist.get_hash(pos);
  }
  for trajectory in trajectories.iter() {
    if trajectory.hash() == hash {
      return;
    }
  }
  let trajectory = Trajectory::new(points.to_vec(), hash, field.score(player));
  trajectories.push(trajectory);
}

fn next_moves(
  field: &Field,
  start_pos: Pos,
  player: Player,
  empty_board: &mut Vec<u32>,
  marks: &mut Vec<Pos>,
) -> Vec<Pos> {
  let mut moves = Vec::new();
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
      if cell.is_putting_allowed() && !cell.is_players_empty_base(player) {
        empty_board[pos] = 1;
        moves.push(pos);
      }
      false
    }
  });
  for &pos in &moves {
    empty_board[pos] = 0;
  }
  moves
}

fn build_trajectories_rec(
  field: &mut Field,
  trajectories: &mut Vec<Trajectory>,
  player: Player,
  cur_depth: u32,
  depth: u32,
  empty_board: &mut Vec<u32>,
  last_pos: Pos,
  moves: Vec<Pos>,
  ensure_pos: Pos,
  should_stop: &AtomicBool,
) {
  for pos in moves {
    if should_stop.load(Ordering::Relaxed) {
      break;
    }
    if field.number_near_points(pos, player) >= 3 {
      continue;
    }
    let cell = field.cell(pos);
    if cell.is_players_empty_base(player.next()) {
      field.put_point(pos, player);
      if field.get_delta_score(player) > 0 && (ensure_pos == 0 || field.cell(ensure_pos).is_bound()) {
        add_trajectory(
          field,
          trajectories,
          field
            .points_seq()
            .index(field.moves_count() - cur_depth as usize..field.moves_count()),
          player,
        );
      }
      field.undo();
    } else {
      field.put_point(pos, player);
      if field.get_delta_score(player) > 0 && (ensure_pos == 0 || field.cell(ensure_pos).is_bound()) {
        add_trajectory(
          field,
          trajectories,
          field
            .points_seq()
            .index(field.moves_count() - cur_depth as usize..field.moves_count()),
          player,
        );
      } else if depth > 0 {
        let mut marks = Vec::new();
        let mut next_moves = next_moves(field, pos, player, empty_board, &mut marks);
        if last_pos != 0 {
          next_moves.retain(|&next_pos| euclidean(field.width(), last_pos, next_pos) > 2);
        }
        build_trajectories_rec(
          field,
          trajectories,
          player,
          cur_depth + 1,
          depth - 1,
          empty_board,
          pos,
          next_moves,
          ensure_pos,
          should_stop,
        );
        for mark_pos in marks {
          empty_board[mark_pos] = 0;
        }
      }
      field.undo();
    }
  }
}

pub fn build_trajectories(
  field: &mut Field,
  player: Player,
  depth: u32,
  empty_board: &mut Vec<u32>,
  should_stop: &AtomicBool,
) -> Vec<Trajectory> {
  let mut trajectories = Vec::new();

  if depth == 0 {
    return trajectories;
  }

  let mut marks = Vec::new();
  for pos in field.points_seq().clone() {
    if field.cell(pos).get_player() != player {
      continue;
    }

    if should_stop.load(Ordering::Relaxed) {
      break;
    }

    let moves = next_moves(field, pos, player, empty_board, &mut marks);

    build_trajectories_rec(
      field,
      &mut trajectories,
      player,
      1,
      depth - 1,
      empty_board,
      0,
      moves,
      0,
      should_stop,
    );
  }

  for pos in marks {
    empty_board[pos] = 0;
  }

  trajectories
}

pub fn build_trajectories_from(
  field: &mut Field,
  pos: Pos,
  player: Player,
  depth: u32,
  empty_board: &mut Vec<u32>,
  should_stop: &AtomicBool,
) -> Vec<Trajectory> {
  let mut trajectories = Vec::new();

  if depth == 0 {
    return trajectories;
  }

  let mut marks = Vec::new();
  let moves = next_moves(field, pos, player, empty_board, &mut marks);

  build_trajectories_rec(
    field,
    &mut trajectories,
    player,
    1,
    depth - 1,
    empty_board,
    0,
    moves,
    pos,
    should_stop,
  );

  for pos in marks {
    empty_board[pos] = 0;
  }

  trajectories
}
