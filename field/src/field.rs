use rand::Rng;
use smallvec::SmallVec;

use crate::cell::Cell;
use crate::player::Player;
use crate::points_vec::PointsVec;
use crate::zobrist::Zobrist;
use std::{collections::VecDeque, fmt, mem, num::NonZeroUsize, sync::Arc};

pub type Pos = usize;
pub type NonZeroPos = NonZeroUsize;

#[derive(Clone, PartialEq)]
struct FieldChange {
  score_red: i32,
  score_black: i32,
  hash: u64,
  points_changes: SmallVec<[(Pos, Cell); 5]>,
  #[cfg(feature = "dsu")]
  dsu_changes: SmallVec<[(Pos, Pos); 5]>,
  #[cfg(feature = "dsu")]
  dsu_size_change: Option<(Pos, u32)>,
}

#[derive(Clone, Copy, PartialEq)]
enum IntersectionState {
  None,
  Up,
  Target,
  Down,
}

#[derive(Clone)]
pub struct Field {
  pub width: u32,
  pub height: u32,
  pub score_red: i32,
  pub score_black: i32,
  pub moves: Vec<Pos>,
  pub points: PointsVec<Cell>,
  #[cfg(feature = "dsu")]
  dsu: PointsVec<Pos>,
  #[cfg(feature = "dsu")]
  dsu_size: PointsVec<u32>,
  changes: Vec<FieldChange>,
  zobrist: Arc<Zobrist>,
  pub hash: u64,
  chain: Vec<Pos>,
  pub q: VecDeque<Pos>,
}

impl PartialEq for Field {
  fn eq(&self, other: &Self) -> bool {
    self.hash == other.hash && self.width == other.width && self.height == other.height && self.moves == other.moves
  }
}

#[inline]
pub fn length(width: u32, height: u32) -> Pos {
  (width as Pos + 2) * (height as Pos + 2)
}

#[inline]
pub fn to_pos(width: u32, x: u32, y: u32) -> Pos {
  (y as Pos + 1) * (width as Pos + 2) + x as Pos + 1
}

#[inline]
pub fn to_x(width: u32, pos: Pos) -> u32 {
  (pos % (width as Pos + 2) - 1) as u32
}

#[inline]
pub fn to_y(width: u32, pos: Pos) -> u32 {
  (pos / (width as Pos + 2) - 1) as u32
}

#[inline]
pub fn n(width: u32, pos: Pos) -> Pos {
  pos - width as Pos - 2
}

#[inline]
pub fn s(width: u32, pos: Pos) -> Pos {
  pos + width as Pos + 2
}

#[inline]
pub fn w(pos: Pos) -> Pos {
  pos - 1
}

#[inline]
pub fn e(pos: Pos) -> Pos {
  pos + 1
}

#[inline]
pub fn nw(width: u32, pos: Pos) -> Pos {
  n(width, w(pos))
}

#[inline]
pub fn ne(width: u32, pos: Pos) -> Pos {
  n(width, e(pos))
}

#[inline]
pub fn sw(width: u32, pos: Pos) -> Pos {
  s(width, w(pos))
}

#[inline]
pub fn se(width: u32, pos: Pos) -> Pos {
  s(width, e(pos))
}

#[inline]
pub fn min_pos(width: u32) -> Pos {
  to_pos(width, 0, 0)
}

#[inline]
pub fn max_pos(width: u32, height: u32) -> Pos {
  to_pos(width, width - 1, height - 1)
}

pub fn is_near(width: u32, pos1: Pos, pos2: Pos) -> bool {
  n(width, pos1) == pos2
    || s(width, pos1) == pos2
    || w(pos1) == pos2
    || e(pos1) == pos2
    || nw(width, pos1) == pos2
    || ne(width, pos1) == pos2
    || sw(width, pos1) == pos2
    || se(width, pos1) == pos2
}

pub fn is_corner(width: u32, height: u32, pos: Pos) -> bool {
  let x = to_x(width, pos);
  let y = to_y(width, pos);
  (x == 0 || x == width - 1) && (y == 0 || y == height - 1)
}

fn get_intersection_state(width: u32, pos: Pos, next_pos: Pos) -> IntersectionState {
  let pos_x = to_x(width, pos);
  let pos_y = to_y(width, pos);
  let next_pos_x = to_x(width, next_pos);
  let next_pos_y = to_y(width, next_pos);
  if next_pos_x <= pos_x {
    match next_pos_y as i32 - pos_y as i32 {
      1 => IntersectionState::Up,
      0 => IntersectionState::Target,
      -1 => IntersectionState::Down,
      _ => IntersectionState::None,
    }
  } else {
    IntersectionState::None
  }
}

pub fn is_point_inside_ring(width: u32, pos: Pos, ring: &[Pos]) -> bool {
  let mut intersections = 0u32;
  let mut state = IntersectionState::None;
  for &next_pos in ring {
    match get_intersection_state(width, pos, next_pos) {
      IntersectionState::None => {
        state = IntersectionState::None;
      }
      IntersectionState::Up => {
        if state == IntersectionState::Down {
          intersections += 1;
        }
        state = IntersectionState::Up;
      }
      IntersectionState::Down => {
        if state == IntersectionState::Up {
          intersections += 1;
        }
        state = IntersectionState::Down;
      }
      IntersectionState::Target => {}
    }
  }
  if state == IntersectionState::Up || state == IntersectionState::Down {
    let mut iter = ring.iter();
    let mut begin_state = get_intersection_state(width, pos, *iter.next().unwrap());
    while begin_state == IntersectionState::Target {
      begin_state = get_intersection_state(width, pos, *iter.next().unwrap());
    }
    if state == IntersectionState::Up && begin_state == IntersectionState::Down
      || state == IntersectionState::Down && begin_state == IntersectionState::Up
    {
      intersections += 1;
    }
  }
  intersections % 2 == 1
}

#[inline]
pub fn skew_product(width: u32, pos1: Pos, pos2: Pos) -> i32 {
  (to_x(width, pos1) * to_y(width, pos2)) as i32 - (to_y(width, pos1) * to_x(width, pos2)) as i32
}

pub fn directions(width: u32, pos: Pos) -> [Pos; 4] {
  [n(width, pos), s(width, pos), w(pos), e(pos)]
}

pub fn directions_diag(width: u32, pos: Pos) -> [Pos; 8] {
  [
    n(width, pos),
    s(width, pos),
    w(pos),
    e(pos),
    nw(width, pos),
    ne(width, pos),
    sw(width, pos),
    se(width, pos),
  ]
}

pub fn wave<F: FnMut(Pos) -> bool>(q: &mut VecDeque<Pos>, width: u32, start_pos: Pos, mut cond: F) {
  if !cond(start_pos) {
    return;
  }
  q.clear();
  q.push_back(start_pos);
  while let Some(pos) = q.pop_front() {
    q.extend(directions(width, pos).iter().filter(|&&pos| cond(pos)))
  }
}

pub fn wave_diag<F: FnMut(Pos) -> bool>(q: &mut VecDeque<Pos>, width: u32, start_pos: Pos, mut cond: F) {
  if !cond(start_pos) {
    return;
  }
  q.clear();
  q.push_back(start_pos);
  while let Some(pos) = q.pop_front() {
    q.extend(directions_diag(width, pos).iter().filter(|&&pos| cond(pos)))
  }
}

#[inline]
pub fn manhattan(width: u32, pos1: Pos, pos2: Pos) -> u32 {
  (i32::abs(to_x(width, pos1) as i32 - to_x(width, pos2) as i32)
    + i32::abs(to_y(width, pos1) as i32 - to_y(width, pos2) as i32)) as u32
}

#[inline]
pub fn euclidean(width: u32, pos1: Pos, pos2: Pos) -> u32 {
  let a = to_x(width, pos1) as i32 - to_x(width, pos2) as i32;
  let b = to_y(width, pos1) as i32 - to_y(width, pos2) as i32;
  (a * a + b * b) as u32
}

impl Field {
  #[inline]
  pub fn to_pos(&self, x: u32, y: u32) -> Pos {
    to_pos(self.width, x, y)
  }

  #[inline]
  pub fn to_x(&self, pos: Pos) -> u32 {
    to_x(self.width, pos)
  }

  #[inline]
  pub fn to_y(&self, pos: Pos) -> u32 {
    to_y(self.width, pos)
  }

  #[inline]
  pub fn n(&self, pos: Pos) -> Pos {
    n(self.width, pos)
  }

  #[inline]
  pub fn s(&self, pos: Pos) -> Pos {
    s(self.width, pos)
  }

  #[inline]
  pub fn w(&self, pos: Pos) -> Pos {
    w(pos)
  }

  #[inline]
  pub fn e(&self, pos: Pos) -> Pos {
    e(pos)
  }

  #[inline]
  pub fn nw(&self, pos: Pos) -> Pos {
    nw(self.width, pos)
  }

  #[inline]
  pub fn ne(&self, pos: Pos) -> Pos {
    ne(self.width, pos)
  }

  #[inline]
  pub fn sw(&self, pos: Pos) -> Pos {
    sw(self.width, pos)
  }

  #[inline]
  pub fn se(&self, pos: Pos) -> Pos {
    se(self.width, pos)
  }

  #[inline]
  pub fn directions(&self, pos: Pos) -> [Pos; 4] {
    directions(self.width, pos)
  }

  #[inline]
  pub fn directions_diag(&self, pos: Pos) -> [Pos; 8] {
    directions_diag(self.width, pos)
  }

  #[inline]
  pub fn min_pos(&self) -> Pos {
    min_pos(self.width)
  }

  #[inline]
  pub fn max_pos(&self) -> Pos {
    max_pos(self.width, self.height)
  }

  #[inline]
  pub fn min_to_max(&self) -> &[Cell] {
    let min_pos = self.min_pos();
    let max_pos = self.max_pos();
    &self.points.0[min_pos..=max_pos]
  }

  #[inline]
  pub fn min_to_max_mut(&mut self) -> &mut [Cell] {
    let min_pos = self.min_pos();
    let max_pos = self.max_pos();
    &mut self.points.0[min_pos..=max_pos]
  }

  #[inline]
  pub fn is_near(&self, pos1: Pos, pos2: Pos) -> bool {
    is_near(self.width, pos1, pos2)
  }

  #[inline]
  pub fn cell(&self, pos: Pos) -> Cell {
    self.points[pos]
  }

  #[inline]
  pub fn length(&self) -> usize {
    self.points.0.len()
  }

  #[inline]
  pub fn is_putting_allowed(&self, pos: Pos) -> bool {
    pos < self.length() && self.cell(pos).is_putting_allowed()
  }

  pub fn has_near_points(&self, center_pos: Pos, player: Player) -> bool {
    self
      .directions(center_pos)
      .iter()
      .any(|&pos| self.cell(pos).is_live_players_point(player))
  }

  pub fn has_near_points_diag(&self, center_pos: Pos, player: Player) -> bool {
    self
      .directions_diag(center_pos)
      .iter()
      .any(|&pos| self.cell(pos).is_live_players_point(player))
  }

  pub fn number_near_points(&self, center_pos: Pos, player: Player) -> u32 {
    self
      .directions(center_pos)
      .iter()
      .filter(|&&pos| self.cell(pos).is_live_players_point(player))
      .count() as u32
  }

  pub fn number_near_points_diag(&self, center_pos: Pos, player: Player) -> u32 {
    self
      .directions_diag(center_pos)
      .iter()
      .filter(|&&pos| self.cell(pos).is_live_players_point(player))
      .count() as u32
  }

  pub fn number_near_groups(&self, center_pos: Pos, player: Player) -> u32 {
    let mut result = 0u32;
    if !self.cell(self.w(center_pos)).is_live_players_point(player)
      && (self.cell(self.nw(center_pos)).is_live_players_point(player)
        || self.cell(self.n(center_pos)).is_live_players_point(player))
    {
      result += 1;
    }
    if !self.cell(self.s(center_pos)).is_live_players_point(player)
      && (self.cell(self.sw(center_pos)).is_live_players_point(player)
        || self.cell(self.w(center_pos)).is_live_players_point(player))
    {
      result += 1;
    }
    if !self.cell(self.e(center_pos)).is_live_players_point(player)
      && (self.cell(self.se(center_pos)).is_live_players_point(player)
        || self.cell(self.s(center_pos)).is_live_players_point(player))
    {
      result += 1;
    }
    if !self.cell(self.n(center_pos)).is_live_players_point(player)
      && (self.cell(self.ne(center_pos)).is_live_players_point(player)
        || self.cell(self.e(center_pos)).is_live_players_point(player))
    {
      result += 1;
    }
    result
  }

  fn set_padding(&mut self) {
    let max_pos = self.max_pos();
    for x in 0..self.width as Pos + 2 {
      self.points[x].set_bad();
      self.points[max_pos + 2 + x].set_bad();
    }
    for y in 1..=self.height as Pos {
      self.points[y * (self.width as Pos + 2)].set_bad();
      self.points[(y + 1) * (self.width as Pos + 2) - 1].set_bad();
    }
  }

  pub fn new(width: u32, height: u32, zobrist: Arc<Zobrist>) -> Field {
    let length = length(width, height);
    assert!(zobrist.hashes.0.len() >= 2 * length);
    #[cfg(feature = "dsu")]
    let mut field = Field {
      width,
      height,
      score_red: 0,
      score_black: 0,
      moves: Vec::with_capacity(length),
      points: vec![Cell::new(false); length].into(),
      dsu: PointsVec((0..length).collect()),
      dsu_size: vec![1; length].into(),
      changes: Vec::with_capacity(length),
      zobrist,
      hash: 0,
      chain: Vec::with_capacity(length),
      q: VecDeque::with_capacity(length),
    };
    #[cfg(not(feature = "dsu"))]
    let mut field = Field {
      width,
      height,
      score_red: 0,
      score_black: 0,
      moves: Vec::with_capacity(length),
      points: vec![Cell::new(false); length].into(),
      changes: Vec::with_capacity(length),
      zobrist,
      hash: 0,
      chain: Vec::with_capacity(length),
      q: VecDeque::with_capacity(length),
    };
    field.set_padding();
    field
  }

  #[inline]
  pub fn new_from_rng<R: Rng>(width: u32, height: u32, rng: &mut R) -> Field {
    let zobrist = Arc::new(Zobrist::new(length(width, height) * 2, rng));
    Field::new(width, height, zobrist)
  }

  #[inline]
  fn save_pos_value(&mut self, pos: Pos) {
    let cell = self.cell(pos);
    self.changes.last_mut().unwrap().points_changes.push((pos, cell));
  }

  #[cfg(feature = "dsu")]
  #[inline]
  fn save_dsu_value(&mut self, pos: Pos) {
    self.changes.last_mut().unwrap().dsu_changes.push((pos, self.dsu[pos]));
  }

  #[cfg(feature = "dsu")]
  #[inline]
  fn save_dsu_size_value(&mut self, pos: Pos) {
    self.changes.last_mut().unwrap().dsu_size_change = Some((pos, self.dsu_size[pos]));
  }

  fn get_input_points(&self, center_pos: Pos, player: Player) -> SmallVec<[(Pos, Pos); 4]> {
    let mut inp_points = SmallVec::new();
    if !self.cell(self.w(center_pos)).is_live_players_point(player) {
      if self.cell(self.nw(center_pos)).is_live_players_point(player) {
        inp_points.push((self.nw(center_pos), self.w(center_pos)));
      } else if self.cell(self.n(center_pos)).is_live_players_point(player) {
        inp_points.push((self.n(center_pos), self.w(center_pos)));
      }
    }
    if !self.cell(self.s(center_pos)).is_live_players_point(player) {
      if self.cell(self.sw(center_pos)).is_live_players_point(player) {
        inp_points.push((self.sw(center_pos), self.s(center_pos)));
      } else if self.cell(self.w(center_pos)).is_live_players_point(player) {
        inp_points.push((self.w(center_pos), self.s(center_pos)));
      }
    }
    if !self.cell(self.e(center_pos)).is_live_players_point(player) {
      if self.cell(self.se(center_pos)).is_live_players_point(player) {
        inp_points.push((self.se(center_pos), self.e(center_pos)));
      } else if self.cell(self.s(center_pos)).is_live_players_point(player) {
        inp_points.push((self.s(center_pos), self.e(center_pos)));
      }
    }
    if !self.cell(self.n(center_pos)).is_live_players_point(player) {
      if self.cell(self.ne(center_pos)).is_live_players_point(player) {
        inp_points.push((self.ne(center_pos), self.n(center_pos)));
      } else if self.cell(self.e(center_pos)).is_live_players_point(player) {
        inp_points.push((self.e(center_pos), self.n(center_pos)));
      }
    }
    inp_points
  }

  #[inline]
  fn skew_product(&self, pos1: Pos, pos2: Pos) -> i32 {
    skew_product(self.width, pos1, pos2)
  }

  //  * . .   x . *   . x x   . . .
  //  . o .   x o .   . o .   . o x
  //  x x .   . . .   . . *   * . x
  //  o - center pos
  //  x - pos
  //  * - result
  fn get_first_next_pos(&self, center_pos: Pos, pos: Pos) -> Pos {
    if pos < center_pos {
      if pos == self.nw(center_pos) || pos == self.w(center_pos) {
        self.ne(center_pos)
      } else {
        self.se(center_pos)
      }
    } else if pos == self.e(center_pos) || pos == self.se(center_pos) {
      self.sw(center_pos)
    } else {
      self.nw(center_pos)
    }
  }

  //  . . .   * . .   x * .   . x *   . . x   . . .   . . .   . . .
  //  * o .   x o .   . o .   . o .   . o *   . o x   . o .   . o .
  //  x . .   . . .   . . .   . . .   . . .   . . *   . * x   * x .
  //  o - center pos
  //  x - pos
  //  * - result
  fn get_next_pos(&self, center_pos: Pos, pos: Pos) -> Pos {
    if pos < center_pos {
      if pos == self.nw(center_pos) {
        self.n(center_pos)
      } else if pos == self.n(center_pos) {
        self.ne(center_pos)
      } else if pos == self.ne(center_pos) {
        self.e(center_pos)
      } else {
        self.nw(center_pos)
      }
    } else if pos == self.e(center_pos) {
      self.se(center_pos)
    } else if pos == self.se(center_pos) {
      self.s(center_pos)
    } else if pos == self.s(center_pos) {
      self.sw(center_pos)
    } else {
      self.w(center_pos)
    }
  }

  fn build_chain(&mut self, start_pos: Pos, player: Player, direction_pos: Pos) -> bool {
    let mut pos = direction_pos;
    let mut center_pos = start_pos;
    let mut base_square = self.skew_product(center_pos, pos);
    self.chain.clear();
    self.chain.push(start_pos);
    loop {
      if self.cell(pos).is_tagged() {
        while *self.chain.last().unwrap() != pos {
          self.points[*self.chain.last().unwrap()].clear_tag();
          self.chain.pop();
        }
      } else {
        self.points[pos].set_tag();
        self.chain.push(pos);
      }
      mem::swap(&mut pos, &mut center_pos);
      pos = self.get_first_next_pos(center_pos, pos);
      while !self.cell(pos).is_live_players_point(player) {
        pos = self.get_next_pos(center_pos, pos);
      }
      base_square += self.skew_product(center_pos, pos);
      if pos == start_pos {
        break;
      }
    }
    for &pos in &self.chain {
      self.points[pos].clear_tag();
    }
    base_square < 0
  }

  fn find_chain(&mut self, start_pos: Pos, player: Player, direction_pos: Pos) -> bool {
    let mut pos = direction_pos;
    let mut center_pos = start_pos;
    let mut base_square = self.skew_product(center_pos, pos);
    self.chain.clear();
    self.chain.push(start_pos);
    loop {
      self.chain.push(pos);
      mem::swap(&mut pos, &mut center_pos);
      pos = self.get_first_next_pos(center_pos, pos);
      while !(self.cell(pos).is_live_players_point(player) && self.cell(pos).is_bound()) {
        pos = self.get_next_pos(center_pos, pos);
      }
      base_square += self.skew_product(center_pos, pos);
      if pos == start_pos {
        break;
      }
    }
    base_square < 0 && self.chain.len() > 2
  }

  #[inline]
  fn is_point_inside_chain(&self, pos: Pos) -> bool {
    is_point_inside_ring(self.width, pos, &self.chain)
  }

  #[inline]
  fn update_hash(&mut self, pos: Pos, player: Player) {
    self.hash ^= if player == Player::Red {
      self.zobrist.hashes[pos]
    } else {
      self.zobrist.hashes[self.length() + pos]
    };
  }

  fn capture(&mut self, inside_pos: Pos, player: Player) -> bool {
    let mut captured_count = 0i32;
    let mut freed_count = 0i32;
    let mut captured_points = Vec::new();
    for &pos in &self.chain {
      self.points[pos].set_tag();
    }
    wave(&mut self.q, self.width, inside_pos, |pos| {
      let cell = self.points[pos];
      if !cell.is_tagged() && !cell.is_bound_player(player) {
        self.points[pos].set_tag();
        captured_points.push(pos);
        if cell.is_put() {
          if cell.get_player() != player {
            captured_count += 1;
          } else if cell.is_captured() {
            freed_count += 1;
          }
        }
        true
      } else {
        false
      }
    });
    if captured_count > 0 {
      match player {
        Player::Red => {
          self.score_red += captured_count;
          self.score_black -= freed_count;
        }
        Player::Black => {
          self.score_black += captured_count;
          self.score_red -= freed_count;
        }
      }
      for &pos in self.chain.iter() {
        self.points[pos].clear_tag();
        self
          .changes
          .last_mut()
          .unwrap()
          .points_changes
          .push((pos, self.points[pos]));
        self.points[pos].set_bound();
      }
      for &pos in &captured_points {
        self.points[pos].clear_tag();
        self.save_pos_value(pos);
        let cell = self.cell(pos);
        if !cell.is_put() {
          if cell.is_captured() {
            self.update_hash(pos, player.next());
          } else {
            self.points[pos].set_captured();
          }
          self.points[pos].clear_empty_base();
          self.points[pos].set_player(player);
          self.update_hash(pos, player);
        } else if cell.get_player() != player {
          self.points[pos].set_captured();
          self.points[pos].clear_bound();
          self.update_hash(pos, player.next());
          self.update_hash(pos, player);
        } else if cell.is_captured() {
          self.points[pos].clear_captured();
          self.update_hash(pos, player.next());
          self.update_hash(pos, player);
        }
      }
      true
    } else {
      for &pos in self.chain.iter() {
        self.points[pos].clear_tag();
      }
      for &pos in &captured_points {
        self.points[pos].clear_tag();
        if !self.points[pos].is_put() {
          self.save_pos_value(pos);
          self.points[pos].set_empty_base_player(player);
        }
      }
      false
    }
  }

  #[cfg(feature = "dsu")]
  fn find_dsu_set(&mut self, pos: Pos) -> Pos {
    let dsu_value = self.dsu[pos];
    if dsu_value == pos {
      pos
    } else {
      let result = self.find_dsu_set(dsu_value);
      if result != dsu_value {
        self.save_dsu_value(pos);
        self.dsu[pos] = result;
      }
      result
    }
  }

  #[cfg(feature = "dsu")]
  fn union_dsu_sets(&mut self, sets: &[Pos]) -> Pos {
    let mut max_dsu_size = 0;
    let mut parent = 0;
    for &set in sets.iter() {
      if self.dsu_size[set] > max_dsu_size {
        max_dsu_size = self.dsu_size[set];
        parent = set;
      }
    }
    self.save_dsu_size_value(parent);
    for &set in sets {
      if self.dsu[set] != parent {
        self.save_dsu_value(set);
        self.dsu[set] = parent;
        self.dsu_size[parent] += self.dsu_size[set];
      }
    }
    parent
  }

  #[cfg(feature = "dsu")]
  fn find_captures(&mut self, pos: Pos, player: Player) -> bool {
    let input_points = self.get_input_points(pos, player);
    let input_points_count = input_points.len();
    if input_points_count > 1 {
      let mut sets: SmallVec<[_; 4]> = SmallVec::new();
      for &(chain_pos, _) in &input_points {
        sets.push(self.find_dsu_set(chain_pos));
      }
      let mut group: SmallVec<[_; 4]> = SmallVec::new();
      let mut result = false;
      for (i, &set) in sets.iter().enumerate() {
        group.clear();
        for j in i..input_points_count {
          if sets[j] == set {
            group.push(input_points[j]);
          }
        }
        let group_points_count = group.len() as u32;
        if group_points_count > 1 {
          let mut chains_count = 0u32;
          for &(chain_pos, captured_pos) in &group {
            if self.build_chain(pos, player, chain_pos) {
              self.capture(captured_pos, player);
              chains_count += 1;
              if chains_count == group_points_count - 1 {
                break;
              }
            }
          }
          if chains_count > 0 {
            result = true;
          }
          if group_points_count >= 3 {
            break;
          }
        }
      }
      let parent = self.union_dsu_sets(&sets);
      self.save_dsu_value(pos);
      self.dsu[pos] = parent;
      self.dsu_size[parent] += 1;
      result
    } else {
      if let Some(&(chain_pos, _)) = input_points.first() {
        let parent = self.find_dsu_set(chain_pos);
        self.save_dsu_value(pos);
        self.dsu[pos] = parent;
        self.save_dsu_size_value(parent);
        self.dsu_size[parent] += 1;
      }
      false
    }
  }

  #[cfg(not(feature = "dsu"))]
  fn find_captures(&mut self, pos: Pos, player: Player) -> bool {
    let input_points = self.get_input_points(pos, player);
    let input_points_count = input_points.len();
    if input_points_count > 1 {
      let mut chains_count = 0;
      for (chain_pos, captured_pos) in input_points {
        if self.build_chain(pos, player, chain_pos) {
          self.capture(captured_pos, player);
          chains_count += 1;
          if chains_count == input_points_count - 1 {
            break;
          }
        }
      }
      chains_count > 0
    } else {
      false
    }
  }

  #[inline]
  fn remove_empty_base(&mut self, start_pos: Pos) {
    wave(&mut self.q, self.width, start_pos, |pos| {
      if self.points[pos].is_empty_base() {
        self
          .changes
          .last_mut()
          .unwrap()
          .points_changes
          .push((pos, self.points[pos]));
        self.points[pos].clear_empty_base();
        true
      } else {
        false
      }
    })
  }

  pub fn put_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.is_putting_allowed(pos) {
      #[cfg(feature = "dsu")]
      let change = FieldChange {
        score_red: self.score_red,
        score_black: self.score_black,
        hash: self.hash,
        points_changes: SmallVec::new(),
        dsu_changes: SmallVec::new(),
        dsu_size_change: None,
      };
      #[cfg(not(feature = "dsu"))]
      let change = FieldChange {
        score_red: self.score_red,
        score_black: self.score_black,
        hash: self.hash,
        points_changes: SmallVec::new(),
      };
      self.changes.push(change);
      self.save_pos_value(pos);
      self.update_hash(pos, player);
      match self.cell(pos).get_empty_base_player() {
        Some(empty_base_player) => {
          self.points[pos].put_point(player);
          if empty_base_player == player {
            self.points[pos].clear_empty_base();
          } else if self.find_captures(pos, player) {
            self.remove_empty_base(pos);
          } else {
            let next_player = player.next();
            let mut bound_pos = pos;
            'outer: loop {
              bound_pos = self.w(bound_pos);
              while !self.cell(bound_pos).is_players_point(next_player) {
                bound_pos = self.w(bound_pos);
              }
              let input_points = self.get_input_points(bound_pos, next_player);
              for (chain_pos, captured_pos) in input_points {
                if self.build_chain(bound_pos, next_player, chain_pos) && self.is_point_inside_chain(pos) {
                  self.capture(captured_pos, next_player);
                  break 'outer;
                }
              }
            }
          }
        }
        None => {
          self.points[pos].put_point(player);
          self.find_captures(pos, player);
        }
      }
      self.moves.push(pos);
      true
    } else {
      false
    }
  }

  pub fn undo(&mut self) -> bool {
    if let Some(change) = self.changes.pop() {
      self.moves.pop();
      self.score_red = change.score_red;
      self.score_black = change.score_black;
      self.hash = change.hash;
      for (pos, cell) in change.points_changes.into_iter().rev() {
        self.points[pos] = cell;
      }
      #[cfg(feature = "dsu")]
      {
        for (pos, dsu_value) in change.dsu_changes.into_iter().rev() {
          self.dsu[pos] = dsu_value;
        }
        if let Some((pos, dsu_size)) = change.dsu_size_change {
          self.dsu_size[pos] = dsu_size;
        }
      }
      true
    } else {
      false
    }
  }

  pub fn undo_all(&mut self) {
    while self.undo() {}
  }

  pub fn get_last_chain(&mut self) -> Vec<Pos> {
    use std::cmp::Ordering;
    let pos = if let Some(&pos) = self.moves.last() {
      pos
    } else {
      return Vec::new();
    };
    let player = self.cell(pos).get_player();
    let delta_score = self.get_delta_score(player);
    match delta_score.cmp(&0) {
      Ordering::Greater => {
        let mut result = Vec::new();
        let input_points = self.get_input_points(pos, player);
        let input_points_count = input_points.len();
        let mut chains_count = 0;
        for (chain_pos, captured_pos) in input_points {
          if !(self.cell(captured_pos).is_captured() && self.cell(chain_pos).is_bound()) {
            continue;
          }
          if self.find_chain(pos, player, chain_pos) {
            result.append(&mut self.chain);
            chains_count += 1;
            if chains_count == input_points_count - 1 {
              break;
            }
          }
        }
        result
      }
      Ordering::Less => {
        let next_player = player.next();
        let mut bound_pos = pos;
        loop {
          bound_pos = self.w(bound_pos);
          while !self.cell(bound_pos).is_bound() {
            bound_pos = self.w(bound_pos);
          }
          let input_points = self.get_input_points(bound_pos, next_player);
          for (chain_pos, _) in input_points {
            if self.find_chain(bound_pos, next_player, chain_pos) && self.is_point_inside_chain(pos) {
              return self.chain.clone();
            }
          }
        }
      }
      Ordering::Equal => Vec::new(),
    }
  }

  #[inline]
  pub fn moves_count(&self) -> usize {
    self.moves.len()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.moves.is_empty()
  }

  #[inline]
  pub fn colored_moves(&self) -> impl ExactSizeIterator<Item = (Pos, Player)> + '_ {
    self.moves.iter().map(|&pos| (pos, self.cell(pos).get_player()))
  }

  #[inline]
  pub fn colored_hash(&self, player: Player) -> u64 {
    self.hash ^ player as u64
  }

  #[inline]
  pub fn hash_at(&self, move_number: usize) -> Option<u64> {
    use std::cmp::Ordering;
    let moves_count = self.moves_count();
    match move_number.cmp(&moves_count) {
      Ordering::Less => Some(self.changes[move_number].hash),
      Ordering::Equal => Some(self.hash),
      Ordering::Greater => None,
    }
  }

  #[inline]
  pub fn last_player(&self) -> Option<Player> {
    self.moves.last().map(|&pos| self.cell(pos).get_player())
  }

  #[inline]
  pub fn cur_player(&self) -> Player {
    self.last_player().unwrap_or(Player::Black).next()
  }

  #[inline]
  pub fn captured_count(&self, player: Player) -> i32 {
    match player {
      Player::Red => self.score_red,
      Player::Black => self.score_black,
    }
  }

  #[inline]
  pub fn score(&self, player: Player) -> i32 {
    match player {
      Player::Red => self.score_red - self.score_black,
      Player::Black => self.score_black - self.score_red,
    }
  }

  #[inline]
  pub fn get_delta_score(&self, player: Player) -> i32 {
    self.score(player)
      - self.changes.last().map_or(0, |change| match player {
        Player::Red => change.score_red - change.score_black,
        Player::Black => change.score_black - change.score_red,
      })
  }

  #[inline]
  pub fn zobrist(&self) -> &Zobrist {
    &self.zobrist
  }

  #[inline]
  pub fn zobrist_arc(&self) -> Arc<Zobrist> {
    self.zobrist.clone()
  }

  pub fn last_changed_cells(&self) -> impl Iterator<Item = (Pos, Cell)> + '_ {
    self
      .changes
      .last()
      .into_iter()
      .flat_map(|change| change.points_changes.iter())
      .cloned()
  }

  pub fn is_corner(&self, pos: Pos) -> bool {
    is_corner(self.width, self.height, pos)
  }

  fn non_grounded_points(&mut self) -> (u32, u32) {
    let mut result = (0, 0);
    for &pos in &self.moves {
      let player = self.cell(pos).get_owner().unwrap();
      let mut points = 0;
      let mut grounded = false;
      wave(&mut self.q, self.width, pos, |pos| {
        let cell = self.points[pos];
        grounded |= cell.is_bad();
        if !cell.is_tagged() && cell.is_owner(player) {
          if cell.is_put() {
            points += 1;
          }
          self.points[pos].set_tag();
          true
        } else {
          false
        }
      });
      if !grounded {
        match player {
          Player::Red => result.0 += points,
          Player::Black => result.1 += points,
        }
      }
    }
    for cell in self.min_to_max_mut() {
      cell.clear_tag();
    }
    result
  }

  pub fn is_game_over(&mut self) -> bool {
    let score = self.score(Player::Red);
    let non_grounded_points = self.non_grounded_points();
    score > non_grounded_points.0 as i32
      || score < -(non_grounded_points.1 as i32)
      || self
        .points
        .0
        .iter()
        .enumerate()
        .all(|(pos, cell)| !cell.is_putting_allowed() || cell.is_empty_base() || self.is_corner(pos))
  }

  pub fn clear(&mut self) {
    if self.moves_count() > self.width as usize * self.height as usize / 8 {
      for cell in self.min_to_max_mut() {
        *cell = Cell::new(false);
      }
      self.set_padding();
      self.changes.clear();
      self.moves.clear();
      self.score_red = 0;
      self.score_black = 0;
      self.hash = 0;
      #[cfg(feature = "dsu")]
      {
        for (i, dsu) in self.dsu.0.iter_mut().enumerate() {
          *dsu = i;
        }
        for dsu in self.dsu_size.0.iter_mut() {
          *dsu = 1;
        }
      }
    } else {
      while self.undo() {}
    }
  }

  pub fn winner(&self) -> Option<Player> {
    use std::cmp::Ordering;
    match self.score(Player::Red).cmp(&0) {
      Ordering::Greater => Some(Player::Red),
      Ordering::Less => Some(Player::Black),
      Ordering::Equal => None,
    }
  }
}

impl fmt::Display for Field {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    for y in 0..self.height {
      for x in 0..self.width {
        let pos = self.to_pos(x, y);
        let cell = self.cell(pos);
        match cell.get_players_point() {
          Some(Player::Red) if cell.is_captured() => write!(f, "x")?,
          Some(Player::Red) => write!(f, "X")?,
          Some(Player::Black) if cell.is_captured() => write!(f, "o")?,
          Some(Player::Black) => write!(f, "O")?,
          None => {
            if cell.is_captured() {
              write!(f, ",")?
            } else {
              write!(f, ".")?
            }
          }
        }
      }
      writeln!(f)?;
    }
    Ok(())
  }
}

impl fmt::Debug for Field {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{}", self)
  }
}
