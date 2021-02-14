use oppai_common::trajectory::{build_trajectories, Trajectory};
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use std::{
  collections::HashSet,
  sync::atomic::{AtomicBool, Ordering},
};

pub struct TrajectoriesPruning {
  rebuild_trajectories: bool,
  cur_trajectories: Vec<Trajectory>,
  enemy_trajectories: Vec<Trajectory>,
  moves: Vec<Pos>,
}

impl TrajectoriesPruning {
  fn project(trajectories: &[Trajectory], empty_board: &mut Vec<u32>) {
    for &pos in trajectories.iter().flat_map(|trajectory| trajectory.points().iter()) {
      empty_board[pos] += 1;
    }
  }

  fn project_length(trajectories: &[Trajectory], empty_board: &mut Vec<u32>) {
    for trajectory in trajectories {
      let len = trajectory.len() as u32;
      for &pos in trajectory.points() {
        if empty_board[pos] == 0 || empty_board[pos] > len {
          empty_board[pos] = len;
        }
      }
    }
  }

  fn deproject(trajectories: &[Trajectory], empty_board: &mut Vec<u32>) {
    for &pos in trajectories.iter().flat_map(|trajectory| trajectory.points().iter()) {
      empty_board[pos] = 0;
    }
  }

  fn exclude_unnecessary_trajectories(trajectories: &mut Vec<Trajectory>, empty_board: &mut Vec<u32>) -> bool {
    let mut need_exclude = false;
    trajectories.retain(|trajectory| {
      let single_count = trajectory.points().iter().filter(|&&pos| empty_board[pos] == 1).count();
      if single_count > 1 {
        for &pos in trajectory.points() {
          empty_board[pos] -= 1;
        }
        need_exclude = true;
        false
      } else {
        true
      }
    });
    need_exclude
  }

  fn calculate_moves(
    trajectories1: &mut Vec<Trajectory>,
    trajectories2: &mut Vec<Trajectory>,
    empty_board: &mut Vec<u32>,
  ) -> Vec<Pos> {
    TrajectoriesPruning::project(trajectories1, empty_board);
    TrajectoriesPruning::project(trajectories2, empty_board);
    while TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories1, empty_board)
      || TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories2, empty_board)
    {}
    let mut result_set = HashSet::new();
    for &pos in trajectories1
      .iter()
      .chain(trajectories2.iter())
      .flat_map(|trajectory| trajectory.points().iter())
    {
      result_set.insert(pos);
    }
    let mut result = result_set.into_iter().collect::<Vec<Pos>>();
    result.sort_unstable_by(|&pos1, &pos2| empty_board[pos2].cmp(&empty_board[pos1]));
    TrajectoriesPruning::deproject(trajectories1, empty_board);
    TrajectoriesPruning::deproject(trajectories2, empty_board);
    TrajectoriesPruning::project_length(trajectories1, empty_board);
    TrajectoriesPruning::project_length(trajectories2, empty_board);
    result.sort_by(|&pos1, &pos2| empty_board[pos1].cmp(&empty_board[pos2]));
    TrajectoriesPruning::deproject(trajectories1, empty_board);
    TrajectoriesPruning::deproject(trajectories2, empty_board);
    result
  }

  #[inline]
  pub fn empty(rebuild_trajectories: bool) -> TrajectoriesPruning {
    TrajectoriesPruning {
      rebuild_trajectories,
      cur_trajectories: Vec::with_capacity(0),
      enemy_trajectories: Vec::with_capacity(0),
      moves: Vec::with_capacity(0),
    }
  }

  pub fn new(
    rebuild_trajectories: bool,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut Vec<u32>,
    should_stop: &AtomicBool,
  ) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    let mut cur_trajectories = build_trajectories(field, player, (depth + 1) / 2, empty_board, should_stop);
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    let mut enemy_trajectories = build_trajectories(field, player.next(), depth / 2, empty_board, should_stop);
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    let moves = TrajectoriesPruning::calculate_moves(&mut cur_trajectories, &mut enemy_trajectories, empty_board);
    TrajectoriesPruning {
      rebuild_trajectories,
      cur_trajectories,
      enemy_trajectories,
      moves,
    }
  }

  fn last_pos_trajectory(field: &Field, player: Player, depth: u32, last_pos: Pos) -> Option<Trajectory> {
    let mut points = Vec::with_capacity(4);
    let mut hash = 0;
    for &pos in &field.directions(last_pos) {
      if field.cell(pos).is_putting_allowed() {
        let mut neighbors_count = 0;
        for &neighbor in &field.directions(pos) {
          if field.cell(neighbor).is_players_point(player) {
            neighbors_count += 1;
          }
        }
        if neighbors_count < 3 {
          points.push(pos);
          hash ^= field.zobrist().get_hash(pos);
        }
      } else if !field.cell(pos).is_players_point(player) {
        return None;
      }
    }
    if points.len() as u32 <= (depth + 1) / 2 {
      Some(Trajectory::new(points, hash, field.score(player) + 1))
    } else {
      None
    }
  }

  pub fn next(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut Vec<u32>,
    last_pos: Pos,
    should_stop: &AtomicBool,
  ) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let mut cur_trajectories = if self.rebuild_trajectories {
      build_trajectories(field, player, (depth + 1) / 2, empty_board, should_stop)
    } else {
      self
        .enemy_trajectories
        .iter()
        .filter(|trajectory| {
          trajectory
            .points()
            .iter()
            .all(|&pos| field.cell(pos).is_putting_allowed())
        })
        .cloned()
        .chain(TrajectoriesPruning::last_pos_trajectory(field, player, depth, last_pos).into_iter())
        .collect()
    };
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let enemy_depth = depth / 2;
    let mut enemy_trajectories = if enemy_depth > 0 {
      self
        .cur_trajectories
        .iter()
        .filter_map(|trajectory| {
          let len = trajectory.len() as u32;
          let contains_pos = trajectory.points().contains(&last_pos);
          if (len <= enemy_depth || len == enemy_depth + 1 && contains_pos)
            && trajectory
              .points()
              .iter()
              .all(|&pos| field.cell(pos).is_putting_allowed() || pos == last_pos)
          {
            let new_trajectory = if contains_pos {
              if len == 1 {
                return None;
              }
              Trajectory::new(
                trajectory
                  .points()
                  .iter()
                  .cloned()
                  .filter(|&pos| pos != last_pos)
                  .collect(),
                trajectory.hash() ^ field.zobrist().get_hash(last_pos),
                trajectory.score(),
              )
            } else {
              trajectory.clone()
            };
            Some(new_trajectory)
          } else {
            None
          }
        })
        .collect()
    } else {
      Vec::new()
    };
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let moves = TrajectoriesPruning::calculate_moves(&mut cur_trajectories, &mut enemy_trajectories, empty_board);
    TrajectoriesPruning {
      rebuild_trajectories: self.rebuild_trajectories,
      cur_trajectories,
      enemy_trajectories,
      moves,
    }
  }

  pub fn dec_and_swap(&self, depth: u32, empty_board: &mut Vec<u32>) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let mut cur_trajectories = self.enemy_trajectories.clone();
    let enemy_depth = depth / 2;
    let mut enemy_trajectories = if enemy_depth > 0 {
      self
        .cur_trajectories
        .iter()
        .filter(|trajectory| trajectory.len() as u32 <= enemy_depth)
        .cloned()
        .collect()
    } else {
      Vec::new()
    };
    let moves = TrajectoriesPruning::calculate_moves(&mut cur_trajectories, &mut enemy_trajectories, empty_board);
    TrajectoriesPruning {
      rebuild_trajectories: self.rebuild_trajectories,
      cur_trajectories,
      enemy_trajectories,
      moves,
    }
  }

  pub fn inc(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut Vec<u32>,
    should_stop: &AtomicBool,
  ) -> TrajectoriesPruning {
    let (mut cur_trajectories, mut enemy_trajectories) = if depth % 2 == 0 {
      let enemy_trajectories = build_trajectories(field, player.next(), depth / 2, empty_board, should_stop);
      if should_stop.load(Ordering::Relaxed) {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      (self.cur_trajectories.clone(), enemy_trajectories)
    } else {
      let cur_trajectories = build_trajectories(field, player, (depth + 1) / 2, empty_board, should_stop);
      if should_stop.load(Ordering::Relaxed) {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      (cur_trajectories, self.enemy_trajectories.clone())
    };
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let moves = TrajectoriesPruning::calculate_moves(&mut cur_trajectories, &mut enemy_trajectories, empty_board);
    TrajectoriesPruning {
      rebuild_trajectories: self.rebuild_trajectories,
      cur_trajectories,
      enemy_trajectories,
      moves,
    }
  }

  fn trajectories_score(trajectories: &[Trajectory]) -> Option<i32> {
    trajectories.iter().map(Trajectory::score).max()
  }

  pub fn alpha(&self) -> Option<i32> {
    TrajectoriesPruning::trajectories_score(&self.enemy_trajectories).map(|score| -score)
  }

  pub fn beta(&self) -> Option<i32> {
    TrajectoriesPruning::trajectories_score(&self.cur_trajectories)
  }

  pub fn moves(&self) -> &Vec<Pos> {
    &self.moves
  }

  pub fn moves_mut(&mut self) -> &mut Vec<Pos> {
    &mut self.moves
  }
}
