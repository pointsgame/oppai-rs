use oppai_field::field::{Field, Pos, euclidean, wave_diag};
use oppai_field::player::Player;
use smallvec::{Array, SmallVec};
use std::ops::Index;

pub trait VecLike<T> {
  fn new() -> Self;
  fn push(&mut self, item: T);
  fn iter(&self) -> std::slice::Iter<'_, T>;
}

impl<T> VecLike<T> for Vec<T> {
  fn new() -> Self {
    Vec::new()
  }

  fn push(&mut self, item: T) {
    self.push(item);
  }

  fn iter(&self) -> std::slice::Iter<'_, T> {
    self.as_slice().iter()
  }
}

impl<T, A> VecLike<T> for SmallVec<A>
where
  A: Array<Item = T>,
{
  fn new() -> Self {
    SmallVec::new()
  }

  fn push(&mut self, item: T) {
    SmallVec::push(self, item)
  }

  fn iter(&self) -> std::slice::Iter<'_, T> {
    self.as_slice().iter()
  }
}

#[derive(Debug, Clone)]
pub struct Trajectory<const N: usize>
where
  [Pos; N]: Array<Item = Pos>,
{
  pub points: SmallVec<[Pos; N]>,
  pub hash: u64,
  pub score: i32,
}

impl<const N: usize> Trajectory<N>
where
  [Pos; N]: Array<Item = Pos>,
{
  pub fn new(points: SmallVec<[Pos; N]>, hash: u64, score: i32) -> Trajectory<N> {
    Trajectory { points, hash, score }
  }
}

fn add_trajectory<const N: usize, C: VecLike<Trajectory<N>>>(
  field: &Field,
  trajectories: &mut C,
  points: &[Pos],
  player: Player,
) where
  [Pos; N]: Array<Item = Pos>,
{
  let zobrist = field.zobrist();
  let mut hash = 0u64;
  for &pos in points {
    if !field.cell(pos).is_bound() || field.number_near_groups(pos, player) < 2 {
      return;
    }
    hash ^= zobrist.hashes[pos];
  }
  for trajectory in trajectories.iter() {
    if trajectory.hash == hash {
      return;
    }
  }
  let trajectory = Trajectory::new(SmallVec::from_slice(points), hash, field.score(player));
  trajectories.push(trajectory);
}

fn next_moves(field: &mut Field, start_pos: Pos, player: Player, marks: &mut SmallVec<[Pos; 1]>) -> SmallVec<[Pos; 7]> {
  let mut moves = SmallVec::new();
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
      if cell.is_putting_allowed() && !cell.is_players_empty_base(player) {
        field.points[pos].set_tag_2();
        moves.push(pos);
      }
      false
    }
  });
  for &pos in &moves {
    field.points[pos].clear_tag_2();
  }
  moves
}

fn build_trajectories_rec<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  trajectories: &mut C,
  player: Player,
  cur_depth: u32,
  depth: u32,
  last_pos: Pos,
  moves: SmallVec<[Pos; 7]>,
  ensure_pos: Pos,
  should_stop: &SS,
) where
  [Pos; N]: Array<Item = Pos>,
{
  for pos in moves {
    if should_stop() {
      break;
    }
    if field.number_near_points(pos, player) >= 3 {
      continue;
    }
    let cell = field.cell(pos);
    field.put_point(pos, player);
    if cell.is_players_empty_base(player.next()) {
      if field.get_delta_score(player) > 0 && (ensure_pos == 0 || field.cell(ensure_pos).is_bound()) {
        add_trajectory(
          field,
          trajectories,
          field
            .moves
            .index(field.moves_count() - cur_depth as usize..field.moves_count()),
          player,
        );
      }
    } else if field.get_delta_score(player) > 0 && (ensure_pos == 0 || field.cell(ensure_pos).is_bound()) {
      add_trajectory(
        field,
        trajectories,
        field
          .moves
          .index(field.moves_count() - cur_depth as usize..field.moves_count()),
        player,
      );
    } else if depth > 0 {
      let mut marks = SmallVec::new();
      let mut next_moves = next_moves(field, pos, player, &mut marks);
      if last_pos != 0 {
        next_moves.retain(|&mut next_pos| euclidean(field.stride, last_pos, next_pos) > 2);
      }
      build_trajectories_rec(
        field,
        trajectories,
        player,
        cur_depth + 1,
        depth - 1,
        pos,
        next_moves,
        ensure_pos,
        should_stop,
      );
      for mark_pos in marks {
        field.points[mark_pos].clear_tag_2();
      }
    }
    field.undo();
  }
}

pub fn build_trajectories<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  player: Player,
  depth: u32,
  should_stop: &SS,
) -> C
where
  [Pos; N]: Array<Item = Pos>,
{
  let mut trajectories = C::new();

  if depth == 0 {
    return trajectories;
  }

  let mut marks = SmallVec::new();
  for pos in field.moves.clone() {
    if field.cell(pos).get_player() != player {
      continue;
    }

    if should_stop() {
      break;
    }

    let moves = next_moves(field, pos, player, &mut marks);

    build_trajectories_rec(field, &mut trajectories, player, 1, depth - 1, 0, moves, 0, should_stop);
  }

  for pos in marks {
    field.points[pos].clear_tag_2();
  }

  trajectories
}

pub fn build_trajectories_from<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  pos: Pos,
  player: Player,
  depth: u32,
  should_stop: &SS,
) -> C
where
  [Pos; N]: Array<Item = Pos>,
{
  let mut trajectories = C::new();

  if depth == 0 {
    return trajectories;
  }

  let mut marks = SmallVec::new();
  let moves = next_moves(field, pos, player, &mut marks);

  build_trajectories_rec(
    field,
    &mut trajectories,
    player,
    1,
    depth - 1,
    0,
    moves,
    pos,
    should_stop,
  );

  for pos in marks {
    field.points[pos].clear_tag_2();
  }

  trajectories
}
