use oppai_field::field::{Field, Hash, ORTHOGONAL, Pos, ring_groups, wave_diag};
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
  pub hash: Hash,
  pub score: i32,
}

impl<const N: usize> Trajectory<N>
where
  [Pos; N]: Array<Item = Pos>,
{
  pub fn new(points: SmallVec<[Pos; N]>, hash: Hash, score: i32) -> Trajectory<N> {
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
  let mut hash = 0;
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

/// Whether the runs of `player`'s dots around `center` can change other than
/// by `player` putting a dot next to it. A capture brings the player's
/// captured dots inside it back to life, so they can join a run later; the
/// opponent's captured count is how many of them there are.
fn runs_settled(field: &Field, center: Pos, player: Player) -> bool {
  field.captured_count(player.next()) == 0 || field.captured_ring(center, player) == 0
}

/// [`Field::number_near_groups`] of `center` with a dot of `player`'s at
/// `extra` as well, or 2 if that may still change by a capture (see
/// [`runs_settled`]).
fn number_near_groups_with(field: &Field, center: Pos, extra: Pos, player: Player) -> u32 {
  let mut ring = field.live_ring(center, player);
  if let Some(direction) = (0..8).find(|&direction| field.ring_pos(center, direction) == extra) {
    ring |= 1 << direction;
  }
  ring_runs_settled(field, center, ring, player)
}

/// [`ring_groups`] of `ring`, the dots around `center`, or 2 if that may
/// still change by a capture (see [`runs_settled`]), so that a dot is never
/// ruled out by a count that may still change.
fn ring_runs_settled(field: &Field, center: Pos, ring: u8, player: Player) -> u32 {
  let groups = ring_groups(ring);
  if groups >= 2 || runs_settled(field, center, player) {
    groups
  } else {
    2
  }
}

fn next_moves(
  field: &mut Field,
  start_pos: Pos,
  player: Player,
  empty_board: &mut [u32],
  marks: &mut SmallVec<[Pos; 1]>,
) -> SmallVec<[Pos; 7]> {
  let mut moves = SmallVec::new();
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

/// The ring directions of the neighbours in the order of `directions_diag`.
const DIAG_DIRECTIONS: [u32; 8] = [0, 4, 6, 2, 7, 1, 5, 3];

/// The moves `next_moves` finds from the just put `pos` when it's the dot
/// before the last one and joins fewer than two runs of the player's dots,
/// leaving out the ones that can't be the last dot. Such a dot needs the last
/// one next to it as its second run, so only the neighbours giving it a second
/// run are kept, in the order `next_moves` finds them. Its only run belongs to
/// a group whose walk already marked it, so the walk from it ends at its
/// neighbours unless an unmarked dot of the player is next to it, in which
/// case `None` is returned.
fn last_moves_of_weak(field: &Field, pos: Pos, player: Player, empty_board: &[u32]) -> Option<SmallVec<[Pos; 7]>> {
  let ring = field.live_ring(pos, player);
  if ring_runs_settled(field, pos, ring, player) >= 2 {
    return None;
  }
  let neighbors = field.directions_diag(pos);
  if neighbors
    .iter()
    .any(|&neighbor| field.cell(neighbor).is_players_point(player) && empty_board[neighbor] == 0)
  {
    return None;
  }
  let mut moves = SmallVec::new();
  for (neighbor, direction) in neighbors.into_iter().zip(DIAG_DIRECTIONS) {
    let cell = field.cell(neighbor);
    if empty_board[neighbor] == 0
      && cell.is_putting_allowed()
      && !cell.is_players_empty_base(player)
      && ring_groups(ring | 1 << direction) >= 2
    {
      moves.push(neighbor);
    }
  }
  Some(moves)
}

/// Leaves the moves outside the 3x3 neighbourhood of `last_pos` if it's set.
fn retain_far(field: &Field, last_pos: Pos, mut moves: SmallVec<[Pos; 7]>) -> SmallVec<[Pos; 7]> {
  if last_pos != 0 {
    // the same as euclidean(field.stride, last_pos, next_pos) > 2 which
    // means being outside the 3x3 neighborhood, but without divisions;
    // unambiguous since dx is at most stride - 2 by absolute value
    let stride = field.stride as isize;
    moves.retain(|&mut next_pos| {
      let delta = (next_pos as isize - last_pos as isize).abs();
      delta > 1 && (delta - stride).abs() > 1
    });
  }
  moves
}

fn build_trajectories_rec<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  trajectories: &mut C,
  player: Player,
  cur_depth: u32,
  depth: u32,
  empty_board: &mut [u32],
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
    // Every check below reads the dots around `pos` from this one ring.
    let ring = field.live_ring(pos, player);
    if (ring & ORTHOGONAL).count_ones() >= 3 {
      continue;
    }
    // A trajectory keeps only dots that join two separate runs of the
    // player's dots around them, see `add_trajectory`, and that count only
    // changes with the dots around them. Nothing is played after the last
    // dot, so whether it and the dots before it qualify is known before it is
    // played.
    if depth == 0
      && (ring_runs_settled(field, pos, ring, player) < 2
        || field.moves[field.moves_count() + 1 - cur_depth as usize..]
          .iter()
          .any(|&prev| number_near_groups_with(field, prev, pos, player) < 2))
    {
      continue;
    }
    // Likewise one dot ahead of the last: a dot short of two runs needs the
    // last one next to it, giving it a second run and joining two runs
    // itself. Which neighbours give it a second run is read from its ring, so
    // most dots are ruled out without looking further.
    if depth == 1
      && ring_runs_settled(field, pos, ring, player) < 2
      && !(0..8).any(|direction| {
        ring >> direction & 1 == 0 && ring_groups(ring | 1 << direction) >= 2 && {
          let next = field.ring_pos(pos, direction);
          let back = 1 << ((direction + 4) & 7);
          field.cell(next).is_putting_allowed()
            && ring_runs_settled(field, next, field.live_ring(next, player) | back, player) >= 2
        }
      })
    {
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
    } else if depth == 1
      && let Some(next_moves) = last_moves_of_weak(field, pos, player, empty_board)
    {
      build_trajectories_rec(
        field,
        trajectories,
        player,
        cur_depth + 1,
        depth - 1,
        empty_board,
        pos,
        retain_far(field, last_pos, next_moves),
        ensure_pos,
        should_stop,
      );
    } else if depth > 0 {
      let mut marks = SmallVec::new();
      let next_moves = next_moves(field, pos, player, empty_board, &mut marks);
      let next_moves = retain_far(field, last_pos, next_moves);
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

pub fn build_trajectories<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  player: Player,
  depth: u32,
  empty_board: &mut [u32],
  should_stop: &SS,
) -> C
where
  [Pos; N]: Array<Item = Pos>,
{
  let mut trajectories = C::new();

  if depth == 0 {
    return trajectories;
  }

  // Every dot the walks pass is marked until the end, so each chain is walked
  // once, from the first of its dots; the rest are skipped without starting
  // a walk.
  let mut marks = SmallVec::with_capacity(field.moves_count());
  // By index rather than over a copy of the moves: every move played below
  // is undone before the next one, so the list is the same at every step.
  for i in 0..field.moves_count() {
    let pos = field.moves[i];
    if field.cell(pos).get_player() != player || empty_board[pos] != 0 {
      continue;
    }

    if should_stop() {
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

pub fn build_trajectories_from<const N: usize, SS: Fn() -> bool, C: VecLike<Trajectory<N>>>(
  field: &mut Field,
  pos: Pos,
  player: Player,
  depth: u32,
  empty_board: &mut [u32],
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
