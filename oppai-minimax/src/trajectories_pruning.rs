use oppai_field::field::{euclidean, wave_diag, Field, Pos};
use oppai_field::player::Player;
use std::{
  collections::HashSet,
  ops::Index,
  sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone)]
struct Trajectory {
  points: Vec<Pos>,
  hash: u64,
}

impl Trajectory {
  pub fn new(points: Vec<Pos>, hash: u64) -> Trajectory {
    Trajectory { points, hash }
  }

  pub fn points(&self) -> &Vec<Pos> {
    &self.points
  }

  pub fn hash(&self) -> u64 {
    self.hash
  }

  pub fn len(&self) -> usize {
    self.points.len()
  }
}

pub struct TrajectoriesPruning {
  rebuild_trajectories: bool,
  cur_trajectories: Vec<Trajectory>,
  enemy_trajectories: Vec<Trajectory>,
  moves: Vec<Pos>,
}

impl TrajectoriesPruning {
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
    let trajectory = Trajectory::new(points.to_vec(), hash);
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
      empty_board[pos] = 1;
      let cell = field.cell(pos);
      if cell.is_players_point(player) {
        marks.push(pos);
        true
      } else {
        if cell.is_putting_allowed() && !cell.is_players_empty_base(player) {
          moves.push(pos);
        } else {
          marks.push(pos);
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
        if field.get_delta_score(player) > 0 {
          TrajectoriesPruning::add_trajectory(
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
        if field.get_delta_score(player) > 0 {
          TrajectoriesPruning::add_trajectory(
            field,
            trajectories,
            field
              .points_seq()
              .index(field.moves_count() - cur_depth as usize..field.moves_count()),
            player,
          );
        } else if depth > 0 {
          let mut marks = Vec::new();
          let mut next_moves = TrajectoriesPruning::next_moves(field, pos, player, empty_board, &mut marks);
          if last_pos != 0 {
            next_moves.retain(|&next_pos| euclidean(field.width(), last_pos, next_pos) > 2);
          }
          TrajectoriesPruning::build_trajectories_rec(
            field,
            trajectories,
            player,
            cur_depth + 1,
            depth - 1,
            empty_board,
            pos,
            next_moves,
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

  fn build_trajectories(
    field: &mut Field,
    trajectories: &mut Vec<Trajectory>,
    player: Player,
    depth: u32,
    empty_board: &mut Vec<u32>,
    should_stop: &AtomicBool,
  ) {
    if depth == 0 {
      return;
    }

    let mut marks = Vec::new();
    for pos in field.points_seq().clone() {
      if field.cell(pos).get_player() != player {
        continue;
      }

      if should_stop.load(Ordering::Relaxed) {
        break;
      }

      let moves = TrajectoriesPruning::next_moves(field, pos, player, empty_board, &mut marks);

      TrajectoriesPruning::build_trajectories_rec(
        field,
        trajectories,
        player,
        1,
        depth - 1,
        empty_board,
        0,
        moves,
        should_stop,
      );
    }

    for pos in marks {
      empty_board[pos] = 0;
    }
  }

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
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    TrajectoriesPruning::build_trajectories(
      field,
      &mut cur_trajectories,
      player,
      (depth + 1) / 2,
      empty_board,
      should_stop,
    );
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(rebuild_trajectories);
    }
    TrajectoriesPruning::build_trajectories(
      field,
      &mut enemy_trajectories,
      player.next(),
      depth / 2,
      empty_board,
      should_stop,
    );
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
    for &pos in &[
      field.n(last_pos),
      field.s(last_pos),
      field.w(last_pos),
      field.e(last_pos),
    ] {
      if field.cell(pos).is_putting_allowed() {
        let mut neighbors_count = 0;
        for &neighbor in &[field.n(pos), field.s(pos), field.w(pos), field.e(pos)] {
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
      Some(Trajectory::new(points, hash))
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
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    if self.rebuild_trajectories {
      TrajectoriesPruning::build_trajectories(
        field,
        &mut cur_trajectories,
        player,
        (depth + 1) / 2,
        empty_board,
        should_stop,
      );
    } else {
      for trajectory in &self.enemy_trajectories {
        if trajectory
          .points()
          .iter()
          .all(|&pos| field.cell(pos).is_putting_allowed())
        {
          cur_trajectories.push((*trajectory).clone());
        }
      }
      if let Some(new_cur_trajectory) = TrajectoriesPruning::last_pos_trajectory(field, player, depth, last_pos) {
        cur_trajectories.push(new_cur_trajectory);
      }
    }
    if should_stop.load(Ordering::Relaxed) {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let enemy_depth = depth / 2;
    if enemy_depth > 0 {
      for trajectory in &self.cur_trajectories {
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
              continue;
            }
            Trajectory::new(
              trajectory
                .points
                .iter()
                .cloned()
                .filter(|&pos| pos != last_pos)
                .collect(),
              trajectory.hash() ^ field.zobrist().get_hash(last_pos),
            )
          } else {
            Trajectory::new(trajectory.points.clone(), trajectory.hash())
          };
          enemy_trajectories.push(new_trajectory);
        }
      }
    }
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

  pub fn dec_and_swap(&self, depth: u32, empty_board: &mut Vec<u32>, should_stop: &AtomicBool) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty(self.rebuild_trajectories);
    }
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    for trajectory in &self.enemy_trajectories {
      cur_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
    }
    let enemy_depth = depth / 2;
    if enemy_depth > 0 {
      for trajectory in self
        .cur_trajectories
        .iter()
        .filter(|trajectory| trajectory.len() as u32 <= enemy_depth)
      {
        enemy_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    }
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

  pub fn inc(
    &self,
    field: &mut Field,
    player: Player,
    depth: u32,
    empty_board: &mut Vec<u32>,
    should_stop: &AtomicBool,
  ) -> TrajectoriesPruning {
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    if depth % 2 == 0 {
      TrajectoriesPruning::build_trajectories(
        field,
        &mut enemy_trajectories,
        player.next(),
        depth / 2,
        empty_board,
        should_stop,
      );
      if should_stop.load(Ordering::Relaxed) {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      for trajectory in &self.cur_trajectories {
        cur_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    } else {
      TrajectoriesPruning::build_trajectories(
        field,
        &mut cur_trajectories,
        player,
        (depth + 1) / 2,
        empty_board,
        should_stop,
      );
      if should_stop.load(Ordering::Relaxed) {
        return TrajectoriesPruning::empty(self.rebuild_trajectories);
      }
      for trajectory in &self.enemy_trajectories {
        enemy_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    }
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

  pub fn moves(&self) -> &Vec<Pos> {
    &self.moves
  }
}
