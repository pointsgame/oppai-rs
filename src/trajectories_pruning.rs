use std::ops::Index;
use zobrist::Zobrist;
use player::Player;
use field::{Pos, Field};

struct Trajectory {
  points: Vec<Pos>,
  hash: u64,
  excluded: bool
}

impl Trajectory {
  pub fn new(points: Vec<Pos>, hash: u64) -> Trajectory {
    Trajectory {
      points: points,
      hash: hash,
      excluded: false
    }
  }

  pub fn points(&self) -> &Vec<Pos> {
    &self.points
  }

  pub fn hash(&self) -> u64 {
    self.hash
  }

  pub fn excluded(&self) -> bool {
    self.excluded
  }

  pub fn len(&self) -> usize {
    self.points.len()
  }

  pub fn exclude(&mut self) {
    self.excluded = true;
  }
}

pub struct TrajectoriesPruning {
  red_trajectories: Vec<Trajectory>,
  black_trajectories: Vec<Trajectory>
}

impl TrajectoriesPruning {
  fn add_trajectory(field: &Field, trajectories: &mut Vec<Trajectory>, points: &[Pos], player: Player) {
    for &pos in points {
      if !field.is_bound(pos) || field.number_near_groups(pos, player) < 2 {
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
    let trajectory = Trajectory::new(points.iter().map(|&pos| pos).collect(), hash);
    trajectories.push(trajectory);
  }

  fn build_trajectories_rec(field: &mut Field, trajectories: &mut Vec<Trajectory>, player: Player, cur_depth: u32, depth: u32) {
    for pos in field.min_pos() .. field.max_pos() + 1 {
      if field.is_putting_allowed(pos) && field.has_near_points(pos, player) && !field.is_players_empty_base(pos, player) { //TODO: optimize with dsu.
        if field.is_players_empty_base(pos, player.next()) {
          field.put_point(pos, player);
          if field.get_delta_score(player) > 0 {
            TrajectoriesPruning::add_trajectory(field, trajectories, field.points_seq().index(field.moves_count() - cur_depth as usize .. field.moves_count()), player);
          }
          field.undo();
        } else {
          field.put_point(pos, player);
          if field.get_delta_score(player) > 0 {
            TrajectoriesPruning::add_trajectory(field, trajectories, field.points_seq().index(field.moves_count() - cur_depth as usize .. field.moves_count()), player);
          } else if depth > 0 {
            TrajectoriesPruning::build_trajectories_rec(field, trajectories, player, cur_depth + 1, depth - 1);
          }
          field.undo();
        }
      }
    }
  }

  fn build_trajectories(field: &mut Field, trajectories: &mut Vec<Trajectory>, player: Player, depth: u32) {
    if depth > 0 {
      TrajectoriesPruning::build_trajectories_rec(field, trajectories, player, 1, depth - 1);
    }
  }

  fn intersection_hash(trajectory1: &Trajectory, trajectory2: &Trajectory, zobrist: &Zobrist, empty_board: &mut Vec<u32>) -> u64 {
    let mut result = trajectory1.hash() ^ trajectory2.hash();
    for &pos in trajectory1.points() {
      empty_board[pos] = 1;
    }
    for &pos in trajectory2.points() {
      if empty_board[pos] != 0 {
        result ^= zobrist.get_hash(pos);
      }
    }
    for &pos in trajectory1.points() {
      empty_board[pos] = 0;
    }
    result
  }

  fn exclude_composite_trajectories(trajectories: &mut Vec<Trajectory>, zobrist: &Zobrist, empty_board: &mut Vec<u32>) {
    let len = trajectories.len();
    for k in 0 .. len {
      for i in 0 .. len - 1 {
        if trajectories[k].len() > trajectories[i].len() {
          for j in i + 1 .. len {
            if trajectories[k].len() > trajectories[j].len() && trajectories[k].hash() == TrajectoriesPruning::intersection_hash(&trajectories[i], &trajectories[j], zobrist, empty_board) {
              trajectories[k].exclude();
            }
          }
        }
      }
    }
  }

  fn project(trajectories: &Vec<Trajectory>, empty_board: &mut Vec<u32>) {
    for &pos in trajectories.iter().filter(|trajectory| !trajectory.excluded()).flat_map(|trajectory| trajectory.points().iter()) {
      empty_board[pos] += 1;
    }
  }

  fn deproject(trajectories: &Vec<Trajectory>, empty_board: &mut Vec<u32>) {
    for &pos in trajectories.iter().filter(|trajectory| !trajectory.excluded()).flat_map(|trajectory| trajectory.points().iter()) {
      empty_board[pos] -= 1;
    }
  }

  fn exclude_unnecessary_trajectories(trajectories: &mut Vec<Trajectory>, empty_board: &mut Vec<u32>) -> bool {
    let mut need_exclude = false;
    for trajectory in trajectories.iter_mut().filter(|trajectory| !trajectory.excluded()) {
      let single_count = trajectory.points().iter().filter(|&&pos| empty_board[pos] == 1).count();
      if single_count > 1 {
        for &pos in trajectory.points() {
          empty_board[pos] -= 1;
        }
        trajectory.exclude();
        need_exclude = true;
      }
    }
    need_exclude
  }

  fn exclude_trajectories(trajectories1: &mut Vec<Trajectory>, trajectories2: &mut Vec<Trajectory>, zobrist: &Zobrist, empty_board: &mut Vec<u32>) {
    TrajectoriesPruning::exclude_composite_trajectories(trajectories1, zobrist, empty_board);
    TrajectoriesPruning::exclude_composite_trajectories(trajectories2, zobrist, empty_board);
    TrajectoriesPruning::project(trajectories1, empty_board);
    TrajectoriesPruning::project(trajectories2, empty_board);
    while TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories1, empty_board) || TrajectoriesPruning::exclude_unnecessary_trajectories(trajectories2, empty_board) { }
    TrajectoriesPruning::deproject(trajectories1, empty_board);
    TrajectoriesPruning::deproject(trajectories2, empty_board);
  }

  pub fn calculate_moves(&self, empty_board: &mut Vec<u32>) -> Vec<Pos> {
    let mut result = Vec::new();
    for &pos in self.red_trajectories.iter().chain(self.black_trajectories.iter()).filter(|trajectory| !trajectory.excluded()).flat_map(|trajectory| trajectory.points().iter()) {
      if empty_board[pos] == 0 {
        empty_board[pos] = 1;
        result.push(pos);
      }
    }
    for &pos in result.iter() {
      empty_board[pos] = 0;
    }
    result
  }

  #[inline]
  fn build_result(cur_trajectories: Vec<Trajectory>, enemy_trajectories: Vec<Trajectory>, player: Player) -> TrajectoriesPruning {
    match player {
      Player::Red => TrajectoriesPruning {
        red_trajectories: cur_trajectories,
        black_trajectories: enemy_trajectories
      },
      Player::Black => TrajectoriesPruning {
        red_trajectories: enemy_trajectories,
        black_trajectories: cur_trajectories
      }
    }
  }

  #[inline]
  pub fn empty() -> TrajectoriesPruning {
    TrajectoriesPruning {
      red_trajectories: Vec::with_capacity(0),
      black_trajectories: Vec::with_capacity(0)
    }
  }

  pub fn new(field: &mut Field, player: Player, depth: u32, empty_board: &mut Vec<u32>) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty();
    }
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    TrajectoriesPruning::build_trajectories(field, &mut cur_trajectories, player, (depth + 1) / 2);
    TrajectoriesPruning::build_trajectories(field, &mut enemy_trajectories, player.next(), depth / 2);
    TrajectoriesPruning::exclude_trajectories(&mut cur_trajectories, &mut enemy_trajectories, field.zobrist(), empty_board);
    TrajectoriesPruning::build_result(cur_trajectories, enemy_trajectories, player)
  }

  pub fn from_last(field: &mut Field, player: Player, depth: u32, empty_board: &mut Vec<u32>, last: &TrajectoriesPruning, last_pos: Pos) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty();
    }
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    let last_enemy_trajectories = match player {
      Player::Red => &last.black_trajectories,
      Player::Black => &last.red_trajectories
    };
    TrajectoriesPruning::build_trajectories(field, &mut cur_trajectories, player, (depth + 1) / 2);
    let enemy_depth = depth / 2;
    if enemy_depth > 0 {
      for trajectory in last_enemy_trajectories {
        let len = trajectory.len() as u32;
        let contains_pos = trajectory.points().contains(&last_pos);
        if (len <= enemy_depth || len == enemy_depth + 1 && contains_pos) && trajectory.points().iter().all(|&pos| field.is_putting_allowed(pos) || pos == last_pos) {
          let new_trajectory = if contains_pos {
            if len == 1 {
              continue;
            }
            Trajectory::new(trajectory.points.iter().map(|&pos| pos).filter(|&pos| pos != last_pos).collect(), trajectory.hash() ^ field.zobrist().get_hash(last_pos))
          } else {
            Trajectory::new(trajectory.points.clone(), trajectory.hash())
          };
          enemy_trajectories.push(new_trajectory);
        }
      }
    }
    TrajectoriesPruning::exclude_trajectories(&mut cur_trajectories, &mut enemy_trajectories, field.zobrist(), empty_board);
    TrajectoriesPruning::build_result(cur_trajectories, enemy_trajectories, player)
  }

  pub fn dec_exists(field: &Field, player: Player, depth: u32, empty_board: &mut Vec<u32>, exists: &TrajectoriesPruning) -> TrajectoriesPruning {
    if depth == 0 {
      return TrajectoriesPruning::empty();
    }
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    let (exists_cur_trajectories, exists_enemy_trajectories) = match player {
      Player::Red => (&exists.red_trajectories, &exists.black_trajectories),
      Player::Black => (&exists.black_trajectories, &exists.red_trajectories)
    };
    for trajectory in exists_cur_trajectories {
      cur_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
    }
    let enemy_depth = depth / 2;
    if enemy_depth > 0 {
      for trajectory in exists_enemy_trajectories.iter().filter(|trajectory| { trajectory.len() as u32 <= enemy_depth }) {
        enemy_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    }
    TrajectoriesPruning::exclude_trajectories(&mut cur_trajectories, &mut enemy_trajectories, field.zobrist(), empty_board);
    TrajectoriesPruning::build_result(cur_trajectories, enemy_trajectories, player)
  }

  pub fn inc_exists(field: &mut Field, player: Player, depth: u32, empty_board: &mut Vec<u32>, exists: &TrajectoriesPruning) -> TrajectoriesPruning {
    let mut cur_trajectories = Vec::new();
    let mut enemy_trajectories = Vec::new();
    let (exists_cur_trajectories, exists_enemy_trajectories) = match player {
      Player::Red => (&exists.red_trajectories, &exists.black_trajectories),
      Player::Black => (&exists.black_trajectories, &exists.red_trajectories)
    };
    if depth % 2 == 0 {
      TrajectoriesPruning::build_trajectories(field, &mut enemy_trajectories, player.next(), depth / 2);
      for trajectory in exists_cur_trajectories {
        cur_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    } else {
      TrajectoriesPruning::build_trajectories(field, &mut cur_trajectories, player, (depth + 1) / 2);
      for trajectory in exists_enemy_trajectories {
        enemy_trajectories.push(Trajectory::new(trajectory.points.clone(), trajectory.hash()));
      }
    }
    TrajectoriesPruning::exclude_trajectories(&mut cur_trajectories, &mut enemy_trajectories, field.zobrist(), empty_board);
    TrajectoriesPruning::build_result(cur_trajectories, enemy_trajectories, player)
  }
}
