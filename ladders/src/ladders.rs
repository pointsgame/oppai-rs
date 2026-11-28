//! Ladders solver.
//!
//! A ladder is a sequence of moves where every move of the attacker threatens
//! an immediate capture: the attacker puts one point of a two-point capturing
//! trajectory, so the other point captures with the next move. The attacker is
//! restricted to trajectories continuing from its last move, so the solver can
//! miss some ladders, but the defender is not restricted. A pending capture
//! can be prevented either by occupying the cell that closes the ring or by
//! capturing a point of the ring, so the defender tries blocking every
//! immediate capture of the attacker and every move that captures something.
//! Any other move lets the attacker capture the best of its immediate captures.
//!
//! A ladder has to capture regardless of the defender capturing something
//! else, so the defender's captures don't count: a counter capture matters
//! only if it disturbs the attacker's captures. The returned scores are the
//! current score plus what the ladder captures.

use oppai_common::common::is_last_move_stupid;
use oppai_common::trajectory::{Trajectory, build_trajectories, build_trajectories_from};
use oppai_field::field::wave_diag;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::iter;

/// The score reached by a line together with the number of the attacker's
/// moves before the capture. The search compares only scores, the depth is
/// kept for reporting: the attacker keeps the first line found with the best
/// score, the defender prefers longer lines among equal scores.
#[derive(Clone, Copy, Debug)]
struct Outcome {
  score: i32,
  depth: u32,
}

impl Outcome {
  fn new(score: i32, depth: u32) -> Self {
    Outcome { score, depth }
  }

  fn max(self, other: Outcome) -> Outcome {
    if other.score > self.score { other } else { self }
  }

  fn min(self, other: Outcome) -> Outcome {
    if other.score < self.score || other.score == self.score && other.depth > self.depth {
      other
    } else {
      self
    }
  }
}

/// Positions where a player captures with the next move. They are checked
/// again before use since later moves can spoil them.
type Captures = SmallVec<[Pos; 8]>;

fn mark_group(field: &mut Field, start_pos: Pos, player: Player, empty_board: &mut [u32], marks: &mut Vec<Pos>) {
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
      false
    }
  });
}

/// The number of the opponent's points the player has captured.
fn own_captures(field: &Field, player: Player) -> i32 {
  match player {
    Player::Red => field.score_red,
    Player::Black => field.score_black,
  }
}

/// Keeps the positions where the player still captures, calling `f` with the
/// score after each capture.
fn check_captures<F: FnMut(i32)>(field: &mut Field, player: Player, captures: &mut Captures, mut f: F) {
  captures.retain(|&mut pos| {
    if !field.put_point(pos, player) {
      return false;
    }
    let result = field.get_delta_score(player) > 0;
    if result {
      f(field.score(player));
    }
    field.undo();
    result
  });
}

struct Search<'a, SS: Fn() -> bool> {
  player: Player,
  /// Marks groups the chase has already passed so that trajectories are
  /// searched only around the new points.
  empty_board: Vec<u32>,
  marks: Vec<Pos>,
  /// A clean board for searching captures at the root.
  captures_board: Vec<u32>,
  /// Marks the defender's points whose neighbors are in the frontier.
  defender_board: Vec<u32>,
  defender_marks: Vec<Pos>,
  /// Empty cells next to the marked defender's points: a new point of the
  /// defender can close a ring at any of them, not only next to itself.
  /// `frontier_board` is 1 for cells in the frontier and 2 for candidates.
  frontier: Vec<Pos>,
  frontier_board: Vec<u8>,
  /// Cells of the frontier that can close a ring of the defender, see
  /// `is_candidate`. Only new points of the defender change that.
  candidates: Vec<Pos>,
  should_stop: &'a SS,
}

impl<'a, SS: Fn() -> bool> Search<'a, SS> {
  fn new(field: &Field, player: Player, should_stop: &'a SS) -> Self {
    Search {
      player,
      empty_board: iter::repeat_n(0, field.length()).collect(),
      marks: Vec::with_capacity(field.length()),
      captures_board: iter::repeat_n(0, field.length()).collect(),
      defender_board: iter::repeat_n(0, field.length()).collect(),
      defender_marks: Vec::new(),
      frontier: Vec::new(),
      frontier_board: iter::repeat_n(0, field.length()).collect(),
      candidates: Vec::new(),
      should_stop,
    }
  }

  fn clear_marks(&mut self, len: usize) {
    for &pos in &self.marks[len..] {
      self.empty_board[pos] = 0;
    }
    self.marks.truncate(len);
  }

  fn root_trajectories(&mut self, field: &mut Field) -> SmallVec<[Trajectory<2>; 8]> {
    let mut trajectories: SmallVec<[Trajectory<2>; 8]> =
      build_trajectories(field, self.player, 2, &mut self.empty_board, self.should_stop);
    trajectories.sort_unstable_by_key(|trajectory| -trajectory.score);
    trajectories
  }

  /// Immediate captures of the defender at the root.
  fn root_counters(&mut self, field: &mut Field) -> Captures {
    build_trajectories::<1, _, SmallVec<[_; 8]>>(
      field,
      self.player.next(),
      1,
      &mut self.captures_board,
      self.should_stop,
    )
    .into_iter()
    .map(|trajectory| trajectory.points[0])
    .collect()
  }

  /// Whether putting a point of the player at the empty `pos` can capture:
  /// it has to join two separate runs of the player's points around it unless
  /// captured points inside the ring come back to life.
  fn is_candidate(field: &Field, pos: Pos, player: Player) -> bool {
    field.number_near_groups(pos, player) >= 2
      || field.captured_count(player.next()) != 0 && field.captured_ring(pos, player) != 0
  }

  /// Marks the defender's groups containing `starts` and extends the frontier
  /// with the empty cells next to them.
  fn extend_frontier(&mut self, field: &mut Field, starts: &[Pos]) {
    let enemy = self.player.next();
    let board = &mut self.defender_board;
    let marks = &mut self.defender_marks;
    let mut cells = SmallVec::<[Pos; 16]>::new();
    for &start in starts {
      wave_diag(&mut field.q, field.stride, start, |pos| {
        if board[pos] != 0 {
          return false;
        }
        let cell = field.points[pos];
        if cell.is_live_players_point(enemy) {
          board[pos] = 1;
          marks.push(pos);
          true
        } else {
          if cell.is_putting_allowed() && !cell.is_players_empty_base(enemy) {
            cells.push(pos);
          }
          false
        }
      });
    }
    for pos in cells {
      if self.frontier_board[pos] == 0 {
        self.frontier_board[pos] = 1;
        self.frontier.push(pos);
      }
      if self.frontier_board[pos] == 1 && Search::<SS>::is_candidate(field, pos, enemy) {
        self.frontier_board[pos] = 2;
        self.candidates.push(pos);
      }
    }
  }

  fn truncate_frontier(&mut self, marks_len: usize, frontier_len: usize, candidates_len: usize) {
    for &pos in &self.candidates[candidates_len..] {
      self.frontier_board[pos] = 1;
    }
    self.candidates.truncate(candidates_len);
    for &pos in &self.frontier[frontier_len..] {
      self.frontier_board[pos] = 0;
    }
    self.frontier.truncate(frontier_len);
    for &pos in &self.defender_marks[marks_len..] {
      self.defender_board[pos] = 0;
    }
    self.defender_marks.truncate(marks_len);
  }

  /// Adds the candidates where the defender might capture now.
  fn add_candidates(&self, field: &Field, captures: &mut Captures) {
    let enemy = self.player.next();
    for &pos in &self.candidates {
      if !captures.contains(&pos)
        && field.cell(pos).is_putting_allowed()
        && Search::<SS>::is_candidate(field, pos, enemy)
      {
        captures.push(pos);
      }
    }
  }

  /// Puts the attacker's point that starts or continues a ladder, returning
  /// false if it's not worth it. A point in the defender's empty base is
  /// worth it only if it captures, otherwise it's captured itself, which
  /// `is_last_move_stupid` rules out.
  fn put_attacker_point(&self, field: &mut Field, pos: Pos) -> bool {
    if !field.put_point(pos, self.player) {
      return false;
    }
    if is_last_move_stupid(field, pos, self.player) {
      field.undo();
      return false;
    }
    true
  }

  /// Evaluates the attacker's move `pos` that was just put at `depth` as a
  /// part of a trajectory with the other point `other` and the score after
  /// capturing with it. A ladder has to
  /// capture regardless of the defender capturing something else, so scores
  /// ignore the defender's captures: `ignored` is what it captured along the
  /// line, added back to the field score. `cap` is the minimum of the best
  /// immediate captures of the attacker at the defender's turns along the
  /// line: the score never exceeds it since the defender could concede any of
  /// them. `threats` and
  /// `counters` are the known immediate captures of the attacker and the
  /// defender. Alpha-beta on scores: the result is exact if it's strictly
  /// between `alpha` and `beta`.
  fn evaluate(
    &mut self,
    field: &mut Field,
    pos: Pos,
    other: (Pos, i32),
    threats: &Captures,
    counters: &Captures,
    cap: i32,
    ignored: i32,
    depth: u32,
    alpha: i32,
    beta: i32,
  ) -> Outcome {
    if field.get_delta_score(self.player) > 0 {
      Outcome::new((field.score(self.player) + ignored).min(cap), depth)
    } else {
      self.defend(field, pos, other, threats, counters, cap, ignored, depth, alpha, beta)
    }
  }

  /// The attacker's turn after its move `last` at `depth - 1` and the
  /// defender's reply, see `evaluate` for the rest.
  fn attack(
    &mut self,
    field: &mut Field,
    last: Pos,
    threats: &Captures,
    counters: &Captures,
    cap: i32,
    ignored: i32,
    depth: u32,
    alpha: i32,
    beta: i32,
  ) -> Outcome {
    let player = self.player;

    let mut best = Outcome::new((field.score(player) + ignored).min(cap), depth);
    let mut threats = threats.clone();
    check_captures(field, player, &mut threats, |score| {
      best = best.max(Outcome::new((score + ignored).min(cap), depth));
    });
    if best.score >= beta || (self.should_stop)() {
      return best;
    }

    let mut trajectories: SmallVec<[Trajectory<2>; 4]> =
      build_trajectories_from(field, last, player, 2, &mut self.empty_board, self.should_stop);
    trajectories.sort_unstable_by_key(|trajectory| -trajectory.score);

    let marks_len = self.marks.len();
    mark_group(field, last, player, &mut self.empty_board, &mut self.marks);

    // The value of a line never exceeds the cap. A trajectory score doesn't
    // limit it since the move can create other threats as well.
    'outer: for trajectory in trajectories {
      if cap <= best.score.max(alpha) || (self.should_stop)() {
        break;
      }
      match *trajectory.points.as_slice() {
        [pos] => {
          if self.put_attacker_point(field, pos) {
            best = best.max(Outcome::new((field.score(player) + ignored).min(cap), depth));
            field.undo();
          }
        }
        [pos1, pos2] => {
          for (our_pos, other_pos) in [(pos1, pos2), (pos2, pos1)] {
            if !self.put_attacker_point(field, our_pos) {
              continue;
            }
            let outcome = self.evaluate(
              field,
              our_pos,
              (other_pos, trajectory.score),
              &threats,
              counters,
              cap,
              ignored,
              depth,
              best.score.max(alpha),
              beta,
            );
            field.undo();
            best = best.max(outcome);
            if best.score >= beta || (self.should_stop)() {
              break 'outer;
            }
          }
        }
        _ => unreachable!("Trajectory with {} points", trajectory.points.len()),
      }
    }

    self.clear_marks(marks_len);

    best
  }

  /// The defender's turn after the attacker's move `last` at `depth`, see
  /// `evaluate`.
  fn defend(
    &mut self,
    field: &mut Field,
    last: Pos,
    (other, other_score): (Pos, i32),
    threats: &Captures,
    counters: &Captures,
    cap: i32,
    ignored: i32,
    depth: u32,
    alpha: i32,
    beta: i32,
  ) -> Outcome {
    let player = self.player;
    let enemy = player.next();

    // The attacker's immediate captures: the known ones and the new ones whose
    // rings go through its last move. The other point of the trajectory
    // captures with the trajectory score, so it's not put again, which is
    // costly with a long chased chain.
    let mut threats = threats.clone();
    threats.retain(|&mut pos| pos != other);
    let mut best_threat = other_score;
    check_captures(field, player, &mut threats, |score| {
      best_threat = best_threat.max(score)
    });
    threats.push(other);
    self.captures_board[other] += 1;
    let new_threats: SmallVec<[Trajectory<1>; 4]> =
      build_trajectories_from(field, last, player, 1, &mut self.captures_board, self.should_stop);
    self.captures_board[other] -= 1;
    for trajectory in new_threats {
      let pos = trajectory.points[0];
      if !threats.contains(&pos) {
        threats.push(pos);
        best_threat = best_threat.max(trajectory.score);
      }
    }
    let cap = cap.min(best_threat + ignored);

    // any move other than blocking a threat or capturing lets the attacker
    // capture the best of them
    let mut worst = Outcome::new(cap, depth + 1);
    if worst.score <= alpha || (self.should_stop)() {
      return worst;
    }

    // a new capture of the defender goes through its last move or encloses
    // the attacker's last move, and its closing cell can be next to any part
    // of the ring, so the whole groups are added to the frontier
    let (marks_len, frontier_len, candidates_len) =
      (self.defender_marks.len(), self.frontier.len(), self.candidates.len());
    let mut starts = SmallVec::<[Pos; 9]>::new();
    if let Some(&prev) = field.moves.get(field.moves_count().wrapping_sub(2))
      && field.cell(prev).is_live_players_point(enemy)
    {
      starts.push(prev);
    }
    for pos in field.directions_diag(last) {
      if field.cell(pos).is_live_players_point(enemy) {
        starts.push(pos);
      }
    }
    self.extend_frontier(field, &starts);
    let mut candidates = counters.clone();
    self.add_candidates(field, &mut candidates);

    // A counter capture matters only if it disturbs the attacker's captures:
    // if the attacker can still capture as much as the cap, nothing can be
    // better for it, since the defender's captures are ignored. Only the
    // disturbing counter captures are searched.
    let mut counters = Captures::new();
    let mut disturbing = Captures::new();
    for pos in candidates {
      let before = own_captures(field, enemy);
      if !field.put_point(pos, enemy) {
        continue;
      }
      if field.get_delta_score(enemy) > 0 {
        counters.push(pos);
        let gain = own_captures(field, enemy) - before;
        let undisturbed = threats.iter().any(|&threat| {
          field.put_point(threat, player) && {
            let result = field.get_delta_score(player) > 0 && field.score(player) + ignored + gain >= cap;
            field.undo();
            result
          }
        });
        if !undisturbed {
          disturbing.push(pos);
        }
      }
      field.undo();
    }

    // the disturbing counter captures first since they are the most likely
    // refutations, then blocking the threats
    for &pos in &threats {
      if !disturbing.contains(&pos) {
        disturbing.push(pos);
      }
    }
    if worst.score > alpha && !(self.should_stop)() {
      for pos in disturbing {
        let before = own_captures(field, enemy);
        if !field.put_point(pos, enemy) {
          continue;
        }
        let gain = own_captures(field, enemy) - before;
        let outcome = self.attack(
          field,
          last,
          &threats,
          &counters,
          cap,
          ignored + gain,
          depth + 1,
          alpha,
          worst.score.min(beta),
        );
        field.undo();
        worst = worst.min(outcome);
        if worst.score <= alpha || (self.should_stop)() {
          break;
        }
      }
    }

    self.truncate_frontier(marks_len, frontier_len, candidates_len);

    worst
  }
}

pub fn ladders<SS: Fn() -> bool>(
  field: &mut Field,
  player: Player,
  should_stop: &SS,
) -> (Option<NonZeroPos>, i32, u32) {
  let mut search = Search::new(field, player, should_stop);
  let trajectories = search.root_trajectories(field);
  let counters = search.root_counters(field);

  info!("Solving ladders for {} trajectories.", trajectories.len());

  let mut best = Outcome::new(field.score(player), 0);
  let mut best_move = None;

  for trajectory in trajectories {
    if should_stop() {
      break;
    }

    match *trajectory.points.as_slice() {
      [pos] => {
        if trajectory.score > best.score {
          best = Outcome::new(trajectory.score, 0);
          best_move = NonZeroPos::new(pos);
        }
      }
      [pos1, pos2] => {
        for (our_pos, other_pos) in [(pos1, pos2), (pos2, pos1)] {
          if !search.put_attacker_point(field, our_pos) {
            continue;
          }
          let outcome = search.evaluate(
            field,
            our_pos,
            (other_pos, trajectory.score),
            &Captures::new(),
            &counters,
            i32::MAX,
            0,
            0,
            best.score,
            i32::MAX,
          );
          field.undo();
          if outcome.score > best.score && !should_stop() {
            best = outcome;
            best_move = NonZeroPos::new(our_pos);
          }
        }
      }
      _ => unreachable!("Trajectory with {} points", trajectory.points.len()),
    }
  }

  (best_move, best.score, best.depth)
}

pub fn ladder_moves<SS: Fn() -> bool>(field: &mut Field, player: Player, should_stop: &SS) -> HashSet<Pos> {
  let mut search = Search::new(field, player, should_stop);
  let trajectories = search.root_trajectories(field);
  let counters = search.root_counters(field);

  let current = field.score(player);

  let mut moves = HashSet::new();

  for trajectory in trajectories {
    if should_stop() {
      break;
    }

    match *trajectory.points.as_slice() {
      [pos] => {
        moves.insert(pos);
      }
      [pos1, pos2] => {
        for (our_pos, other_pos) in [(pos1, pos2), (pos2, pos1)] {
          if moves.contains(&our_pos) || !search.put_attacker_point(field, our_pos) {
            continue;
          }
          let outcome = search.evaluate(
            field,
            our_pos,
            (other_pos, trajectory.score),
            &Captures::new(),
            &counters,
            i32::MAX,
            0,
            0,
            current,
            current + 1,
          );
          field.undo();
          if outcome.score > current && !should_stop() {
            moves.insert(our_pos);
          }
        }
      }
      _ => unreachable!("Trajectory with {} points", trajectory.points.len()),
    }
  }

  moves
}
