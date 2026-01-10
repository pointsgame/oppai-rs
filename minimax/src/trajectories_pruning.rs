use oppai_common::trajectory::{Trajectory, build_trajectories};
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use smallvec::SmallVec;

pub struct TrajectoriesPruning {
  pub rebuild_trajectories: bool,
  pub cur_trajectories: Vec<Trajectory<8>>,
  pub enemy_trajectories: Vec<Trajectory<8>>,
  pub moves: Vec<Pos>,
}

impl TrajectoriesPruning {
  fn project(trajectories: &[Trajectory<8>], empty_board: &mut [u32]) {
    for &pos in trajectories.iter().flat_map(|trajectory| trajectory.points.iter()) {
      empty_board[pos] += 1;
    }
  }

  fn project_length(trajectories: &[Trajectory<8>], empty_board: &mut [u32]) {
    for trajectory in trajectories {
      let len = trajectory.points.len() as u32;
      for &pos in &trajectory.points {
        let cur_len = empty_board[pos] >> 16;
        if cur_len == 0 || cur_len > len {
          empty_board[pos] = (empty_board[pos] & 0xFFFF) | (len << 16);
        }
      }
    }
  }

  fn deproject(trajectories: &[Trajectory<8>], empty_board: &mut [u32]) {
    for &pos in trajectories.iter().flat_map(|trajectory| trajectory.points.iter()) {
      empty_board[pos] = 0;
    }
  }

  fn exclude_unnecessary_trajectories(trajectories: &mut Vec<Trajectory<8>>, empty_board: &mut [u32]) -> bool {
    let mut need_exclude = false;
    trajectories.retain(|trajectory| {
      let single_count = trajectory.points.iter().filter(|&&pos| empty_board[pos] == 1).count();
      if single_count > 1 {
        for &pos in &trajectory.points {
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
    trajectories1: &mut Vec<Trajectory<8>>,
    trajectories2: &mut Vec<Trajectory<8>>,
    empty_board: &mut [u32],
  ) -> Vec<Pos> {
    TrajectoriesPruning::project(trajectories1, empty_board);
    TrajectoriesPruning::project(trajectories2, empty_board);

    while TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories1, empty_board)
      || TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories2, empty_board)
    {}

    const SEEN_FLAG: u32 = 0x8000_0000;

    let mut result = Vec::new();
    for &pos in trajectories1
      .iter()
      .chain(trajectories2.iter())
      .flat_map(|trajectory| trajectory.points.iter())
    {
      if empty_board[pos] & SEEN_FLAG == 0 {
        empty_board[pos] |= SEEN_FLAG;
        result.push(pos);
      }
    }
    for &pos in &result {
      empty_board[pos] &= !SEEN_FLAG;
    }

    TrajectoriesPruning::project_length(trajectories1, empty_board);
    TrajectoriesPruning::project_length(trajectories2, empty_board);

    result.sort_unstable_by_key(|&pos| (empty_board[pos] >> 16, -((empty_board[pos] & 0xFFFF) as i32)));

    TrajectoriesPruning::deproject(trajectories1, empty_board);
    TrajectoriesPruning::deproject(trajectories2, empty_board);

    result
  }

  #[inline]
  pub fn empty(rebuild_trajectories: bool) -> TrajectoriesPruning {
    TrajectoriesPruning {
      rebuild_trajectories,
      cur_trajectories: Vec::new(),
      enemy_trajectories: Vec::new(),
      moves: Vec::new(),
    }
  }

  pub fn new<SS: Fn() -> bool>(
    rebuild_trajectories: bool,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut [u32],
    should_stop: &SS,
  ) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    let mut cur_trajectories = build_trajectories(field, player, depth.div_ceil(2), empty_board, should_stop);
    if should_stop() {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    let mut enemy_trajectories = build_trajectories(field, player.next(), depth / 2, empty_board, should_stop);
    if should_stop() {
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

  fn last_pos_trajectory(field: &Field, player: Player, depth: u32, last_pos: Pos) -> Option<Trajectory<8>> {
    let mut points = SmallVec::new();
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
          hash ^= field.zobrist().hashes[pos];
        }
      } else if !field.cell(pos).is_players_point(player) {
        return None;
      }
    }
    if points.len() as u32 <= depth.div_ceil(2) {
      Some(Trajectory::new(points, hash, field.score(player) + 1))
    } else {
      None
    }
  }

  pub fn next<SS: Fn() -> bool>(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut [u32],
    last_pos: Pos,
    should_stop: &SS,
  ) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let mut cur_trajectories = if self.rebuild_trajectories {
      build_trajectories(field, player, depth.div_ceil(2), empty_board, should_stop)
    } else {
      self
        .enemy_trajectories
        .iter()
        .filter(|trajectory| {
          trajectory
            .points
            .iter()
            .all(|&pos| field.cell(pos).is_putting_allowed())
        })
        .cloned()
        .chain(TrajectoriesPruning::last_pos_trajectory(field, player, depth, last_pos))
        .collect()
    };
    if should_stop() {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let enemy_depth = depth / 2;
    let mut enemy_trajectories = if enemy_depth > 0 {
      self
        .cur_trajectories
        .iter()
        .filter_map(|trajectory| {
          let len = trajectory.points.len() as u32;
          let contains_pos = trajectory.points.contains(&last_pos);
          if (len <= enemy_depth || len == enemy_depth + 1 && contains_pos)
            && trajectory
              .points
              .iter()
              .all(|&pos| field.cell(pos).is_putting_allowed() || pos == last_pos)
          {
            let new_trajectory = if contains_pos {
              if len == 1 {
                return None;
              }
              Trajectory::new(
                trajectory
                  .points
                  .iter()
                  .cloned()
                  .filter(|&pos| pos != last_pos)
                  .collect(),
                trajectory.hash ^ field.zobrist().hashes[last_pos],
                trajectory.score,
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
    if should_stop() {
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

  pub fn dec_and_swap(&self, depth: u32, empty_board: &mut [u32]) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let mut cur_trajectories = self.enemy_trajectories.clone();
    let enemy_depth = depth / 2;
    let mut enemy_trajectories = if enemy_depth > 0 {
      self
        .cur_trajectories
        .iter()
        .filter(|trajectory| trajectory.points.len() as u32 <= enemy_depth)
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

  pub fn inc<SS: Fn() -> bool>(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut [u32],
    should_stop: &SS,
  ) -> TrajectoriesPruning {
    let (mut cur_trajectories, mut enemy_trajectories) = if depth.is_multiple_of(2) {
      let enemy_trajectories = build_trajectories(field, player.next(), depth / 2, empty_board, should_stop);
      if should_stop() {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      (self.cur_trajectories.clone(), enemy_trajectories)
    } else {
      let cur_trajectories = build_trajectories(field, player, depth.div_ceil(2), empty_board, should_stop);
      if should_stop() {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      (cur_trajectories, self.enemy_trajectories.clone())
    };
    if should_stop() {
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

  fn trajectories_score(trajectories: &[Trajectory<8>]) -> Option<i32> {
    trajectories.iter().map(|trajectory| trajectory.score).max()
  }

  pub fn alpha(&self) -> Option<i32> {
    TrajectoriesPruning::trajectories_score(&self.enemy_trajectories).map(|score| -score)
  }

  pub fn beta(&self) -> Option<i32> {
    TrajectoriesPruning::trajectories_score(&self.cur_trajectories)
  }
}
