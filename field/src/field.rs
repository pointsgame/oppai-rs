use rand::Rng;
use smallvec::SmallVec;

use crate::cell::Cell;
use crate::player::Player;
use crate::points_vec::PointsVec;
use crate::zobrist::Zobrist;
use std::{collections::VecDeque, fmt, num::NonZeroUsize, sync::Arc};

pub type Pos = usize;
pub type NonZeroPos = NonZeroUsize;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Neighbor(usize);

impl Neighbor {
  const N: Self = Self(0);
  const NE: Self = Self(1);
  const E: Self = Self(2);
  const SE: Self = Self(3);
  const S: Self = Self(4);
  const SW: Self = Self(5);
  const W: Self = Self(6);
  const NW: Self = Self(7);

  //  . . .   * . .   x * .   . x *   . . x   . . .   . . .   . . .
  //  * o .   x o .   . o .   . o .   . o *   . o x   . o .   . o .
  //  x . .   . . .   . . .   . . .   . . .   . . *   . * x   * x .
  //  o - center pos
  //  x - pos
  //  * - result
  #[inline(always)]
  fn next(self) -> Self {
    Self((self.0 + 1) & 7)
  }

  //  * . .   x . *   . x x   . . .
  //  . o .   x o .   . o .   . o x
  //  x x .   . . .   . . *   * . x
  //  o - center pos
  //  x - pos
  //  * - result
  #[inline(always)]
  fn first_next(self) -> Self {
    Self(((self.0 + 6) | 1) & 7)
  }

  #[inline(always)]
  fn apply(self, neighbor_offsets: &[isize; 8], pos: Pos) -> Pos {
    pos.overflowing_add_signed(neighbor_offsets[self.0]).0
  }
}

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
  pub stride: u32,
  pub score_red: i32,
  pub score_black: i32,
  pub moves: Vec<Pos>,
  pub points: PointsVec<Cell>,
  neighbor_offsets: [isize; 8],
  #[cfg(feature = "dsu")]
  dsu: PointsVec<Pos>,
  #[cfg(feature = "dsu")]
  dsu_size: PointsVec<u32>,
  changes: Vec<FieldChange>,
  zobrist: Arc<Zobrist>,
  pub hash: u64,
  chains: [(Vec<Pos>, Pos); 3],
  captured_points: Vec<Pos>,
  pub q: VecDeque<Pos>,
}

impl PartialEq for Field {
  fn eq(&self, other: &Self) -> bool {
    self.hash == other.hash
      && self.stride == other.stride
      && self.length() == other.length()
      && self.moves == other.moves
  }
}

#[inline]
pub fn length(width: u32, height: u32) -> Pos {
  (width as Pos + 1) * (height as Pos + 2) + 1
}

#[inline]
pub fn to_pos(stride: u32, x: u32, y: u32) -> Pos {
  (y as Pos + 1) * stride as Pos + x as Pos + 1
}

#[inline]
pub fn to_x(stride: u32, pos: Pos) -> u32 {
  (pos % stride as Pos - 1) as u32
}

#[inline]
pub fn to_xy(stride: u32, pos: Pos) -> (u32, u32) {
  (to_x(stride, pos), to_y(stride, pos))
}

#[inline]
pub fn to_y(stride: u32, pos: Pos) -> u32 {
  (pos / stride as Pos - 1) as u32
}

#[inline]
pub fn n(stride: u32, pos: Pos) -> Pos {
  pos - stride as Pos
}

#[inline]
pub fn s(stride: u32, pos: Pos) -> Pos {
  pos + stride as Pos
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
pub fn nw(stride: u32, pos: Pos) -> Pos {
  n(stride, w(pos))
}

#[inline]
pub fn ne(stride: u32, pos: Pos) -> Pos {
  n(stride, e(pos))
}

#[inline]
pub fn sw(stride: u32, pos: Pos) -> Pos {
  s(stride, w(pos))
}

#[inline]
pub fn se(stride: u32, pos: Pos) -> Pos {
  s(stride, e(pos))
}

pub fn is_near(stride: u32, pos1: Pos, pos2: Pos) -> bool {
  n(stride, pos1) == pos2
    || s(stride, pos1) == pos2
    || w(pos1) == pos2
    || e(pos1) == pos2
    || nw(stride, pos1) == pos2
    || ne(stride, pos1) == pos2
    || sw(stride, pos1) == pos2
    || se(stride, pos1) == pos2
}

pub fn is_corner(width: u32, height: u32, pos: Pos) -> bool {
  let x = to_x(width + 1, pos);
  let y = to_y(width + 1, pos);
  (x == 0 || x == width - 1) && (y == 0 || y == height - 1)
}

#[inline]
fn get_intersection_state(stride: u32, pos_x: u32, pos_y: u32, next_pos: Pos) -> IntersectionState {
  let next_pos_x = to_x(stride, next_pos);
  let next_pos_y = to_y(stride, next_pos);
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

pub fn is_point_inside_ring(stride: u32, pos: Pos, ring: &[Pos]) -> bool {
  let pos_x = to_x(stride, pos);
  let pos_y = to_y(stride, pos);
  let mut intersections = 0u32;
  let mut state = IntersectionState::None;
  for &next_pos in ring {
    match get_intersection_state(stride, pos_x, pos_y, next_pos) {
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
    let mut begin_state = get_intersection_state(stride, pos_x, pos_y, *iter.next().unwrap());
    while begin_state == IntersectionState::Target {
      begin_state = get_intersection_state(stride, pos_x, pos_y, *iter.next().unwrap());
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
pub fn skew_product(coord1: (u32, u32), coord2: (u32, u32)) -> i32 {
  (coord1.0 * coord2.1) as i32 - (coord1.1 * coord2.0) as i32
}

pub fn directions(stride: u32, pos: Pos) -> [Pos; 4] {
  [n(stride, pos), s(stride, pos), w(pos), e(pos)]
}

pub fn directions_diag(stride: u32, pos: Pos) -> [Pos; 8] {
  [
    n(stride, pos),
    s(stride, pos),
    w(pos),
    e(pos),
    nw(stride, pos),
    ne(stride, pos),
    sw(stride, pos),
    se(stride, pos),
  ]
}

pub fn wave<F: FnMut(Pos) -> bool>(q: &mut VecDeque<Pos>, stride: u32, start_pos: Pos, mut cond: F) {
  if !cond(start_pos) {
    return;
  }
  q.clear();
  q.push_back(start_pos);
  while let Some(pos) = q.pop_front() {
    q.extend(directions(stride, pos).into_iter().filter(|&pos| cond(pos)))
  }
}

pub fn wave_diag<F: FnMut(Pos) -> bool>(q: &mut VecDeque<Pos>, stride: u32, start_pos: Pos, mut cond: F) {
  if !cond(start_pos) {
    return;
  }
  q.clear();
  q.push_back(start_pos);
  while let Some(pos) = q.pop_front() {
    q.extend(directions_diag(stride, pos).into_iter().filter(|&pos| cond(pos)))
  }
}

#[inline]
pub fn manhattan(stride: u32, pos1: Pos, pos2: Pos) -> u32 {
  (i32::abs(to_x(stride, pos1) as i32 - to_x(stride, pos2) as i32)
    + i32::abs(to_y(stride, pos1) as i32 - to_y(stride, pos2) as i32)) as u32
}

#[inline]
pub fn euclidean(stride: u32, pos1: Pos, pos2: Pos) -> u32 {
  let a = to_x(stride, pos1) as i32 - to_x(stride, pos2) as i32;
  let b = to_y(stride, pos1) as i32 - to_y(stride, pos2) as i32;
  (a * a + b * b) as u32
}

fn get_input_points(
  stride: u32,
  points: &PointsVec<Cell>,
  center_pos: Pos,
  player: Player,
) -> SmallVec<[(Neighbor, Pos); 4]> {
  let mut inp_points = SmallVec::new();

  let pos_w = center_pos - 1;
  let pos_e = center_pos + 1;
  let pos_n = center_pos - stride as usize;
  let pos_s = center_pos + stride as usize;

  let has_w = points[pos_w].is_live_players_point(player);
  let has_e = points[pos_e].is_live_players_point(player);
  let has_n = points[pos_n].is_live_players_point(player);
  let has_s = points[pos_s].is_live_players_point(player);

  if !has_w {
    // NW
    if points[pos_n - 1].is_live_players_point(player) {
      inp_points.push((Neighbor::NW, pos_w));
    } else if has_n {
      inp_points.push((Neighbor::N, pos_w));
    }
  }
  if !has_s {
    // SW
    if points[pos_s - 1].is_live_players_point(player) {
      inp_points.push((Neighbor::SW, pos_s));
    } else if has_w {
      inp_points.push((Neighbor::W, pos_s));
    }
  }
  if !has_e {
    // SE
    if points[pos_s + 1].is_live_players_point(player) {
      inp_points.push((Neighbor::SE, pos_e));
    } else if has_s {
      inp_points.push((Neighbor::S, pos_e));
    }
  }
  if !has_n {
    // NE
    if points[pos_n + 1].is_live_players_point(player) {
      inp_points.push((Neighbor::NE, pos_n));
    } else if has_e {
      inp_points.push((Neighbor::E, pos_n));
    }
  }

  inp_points
}

fn build_chain(
  neighbor_offsets: &[isize; 8],
  stride: u32,
  points: &mut PointsVec<Cell>,
  chain: &mut Vec<Pos>,
  start_pos: Pos,
  player: Player,
  mut direction: Neighbor,
) -> bool {
  let mut center_pos = direction.apply(neighbor_offsets, start_pos);
  chain.clear();
  chain.push(start_pos);
  chain.push(center_pos);
  points[center_pos].set_tag();
  loop {
    direction = direction.first_next();
    let mut pos = direction.apply(neighbor_offsets, center_pos);
    while !points[pos].is_live_players_point(player) {
      if points[pos].is_bad() {
        for &pos in &*chain {
          points[pos].clear_tag();
        }
        chain.clear(); // used as a mark for reaching border
        return false;
      }
      direction = direction.next();
      pos = direction.apply(neighbor_offsets, center_pos);
    }
    if pos == start_pos {
      break;
    }
    if points[pos].is_tagged() {
      while *chain.last().unwrap() != pos {
        points[*chain.last().unwrap()].clear_tag();
        chain.pop();
      }
    } else {
      points[pos].set_tag();
      chain.push(pos);
    }
    center_pos = pos;
  }
  let mut center_coord = to_xy(stride, start_pos);
  let mut base_area = skew_product(to_xy(stride, center_pos), center_coord);
  for &pos in chain.iter().skip(1) {
    let pos_coord = to_xy(stride, pos);
    base_area += skew_product(center_coord, pos_coord);
    center_coord = pos_coord;
    points[pos].clear_tag();
  }
  base_area < 0
}

fn find_chain(
  neighbor_offsets: &[isize; 8],
  stride: u32,
  points: &mut PointsVec<Cell>,
  chain: &mut Vec<Pos>,
  start_pos: Pos,
  player: Player,
  mut direction: Neighbor,
) -> bool {
  let mut center_pos = direction.apply(neighbor_offsets, start_pos);
  let mut center_coord = to_xy(stride, center_pos);
  let mut base_area = skew_product(to_xy(stride, start_pos), center_coord);
  chain.clear();
  chain.push(start_pos);
  loop {
    chain.push(center_pos);
    direction = direction.first_next();
    let mut pos = direction.apply(neighbor_offsets, center_pos);
    while !(points[pos].is_live_players_point(player) && points[pos].is_bound()) {
      direction = direction.next();
      pos = direction.apply(neighbor_offsets, center_pos);
    }
    let pos_coord = to_xy(stride, pos);
    base_area += skew_product(center_coord, pos_coord);
    if pos == start_pos {
      break;
    }
    center_pos = pos;
    center_coord = pos_coord;
  }
  base_area < 0 && chain.len() > 2
}

impl Field {
  #[inline]
  pub fn width(&self) -> u32 {
    self.stride - 1
  }

  pub fn height(&self) -> u32 {
    self.length() as u32 / self.stride - 2
  }

  #[inline]
  pub fn to_pos(&self, x: u32, y: u32) -> Pos {
    to_pos(self.stride, x, y)
  }

  #[inline]
  pub fn to_x(&self, pos: Pos) -> u32 {
    to_x(self.stride, pos)
  }

  #[inline]
  pub fn to_xy(&self, pos: Pos) -> (u32, u32) {
    to_xy(self.stride, pos)
  }

  #[inline]
  pub fn to_y(&self, pos: Pos) -> u32 {
    to_y(self.stride, pos)
  }

  #[inline]
  pub fn n(&self, pos: Pos) -> Pos {
    n(self.stride, pos)
  }

  #[inline]
  pub fn s(&self, pos: Pos) -> Pos {
    s(self.stride, pos)
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
    nw(self.stride, pos)
  }

  #[inline]
  pub fn ne(&self, pos: Pos) -> Pos {
    ne(self.stride, pos)
  }

  #[inline]
  pub fn sw(&self, pos: Pos) -> Pos {
    sw(self.stride, pos)
  }

  #[inline]
  pub fn se(&self, pos: Pos) -> Pos {
    se(self.stride, pos)
  }

  #[inline]
  pub fn directions(&self, pos: Pos) -> [Pos; 4] {
    directions(self.stride, pos)
  }

  #[inline]
  pub fn directions_diag(&self, pos: Pos) -> [Pos; 8] {
    directions_diag(self.stride, pos)
  }

  #[inline]
  pub fn min_pos(&self) -> Pos {
    self.stride as usize + 1
  }

  #[inline]
  pub fn max_pos(&self) -> Pos {
    self.length() - self.stride as usize - 2
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
    is_near(self.stride, pos1, pos2)
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
      .into_iter()
      .any(|pos| self.cell(pos).is_live_players_point(player))
  }

  pub fn has_near_points_diag(&self, center_pos: Pos, player: Player) -> bool {
    self
      .directions_diag(center_pos)
      .into_iter()
      .any(|pos| self.cell(pos).is_live_players_point(player))
  }

  pub fn number_near_points(&self, center_pos: Pos, player: Player) -> u32 {
    self
      .directions(center_pos)
      .into_iter()
      .filter(|&pos| self.cell(pos).is_live_players_point(player))
      .count() as u32
  }

  pub fn number_near_points_diag(&self, center_pos: Pos, player: Player) -> u32 {
    self
      .directions_diag(center_pos)
      .into_iter()
      .filter(|&pos| self.cell(pos).is_live_players_point(player))
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
    let height = self.height();
    let last = self.length() - 2;
    self.points[last + 1].set_bad();
    for x in 0..self.stride as Pos {
      self.points[x].set_bad();
      self.points[last - x].set_bad();
    }
    for y in 1..=height as Pos {
      self.points[y * (self.stride as Pos)].set_bad();
    }
  }

  pub fn new(width: u32, height: u32, zobrist: Arc<Zobrist>) -> Field {
    let length = length(width, height);
    assert!(zobrist.hashes.0.len() >= 2 * length);
    let stride = width + 1;
    let s = stride as isize;
    let neighbor_offsets = [
      -s,     // N
      -s + 1, // NE
      1,      // E
      s + 1,  // SE
      s,      // S
      s - 1,  // SW
      -1,     // W
      -s - 1, // NW
    ];
    #[cfg(feature = "dsu")]
    let mut field = Field {
      stride,
      score_red: 0,
      score_black: 0,
      moves: Vec::with_capacity(length),
      points: vec![Cell::new(false); length].into(),
      neighbor_offsets,
      dsu: PointsVec((0..length).collect()),
      dsu_size: vec![1; length].into(),
      changes: Vec::with_capacity(length),
      zobrist,
      hash: 0,
      chains: [
        (Vec::with_capacity(length), 0),
        (Vec::with_capacity(length), 0),
        (Vec::with_capacity(length), 0),
      ],
      captured_points: Vec::with_capacity(length),
      q: VecDeque::with_capacity(length),
    };
    #[cfg(not(feature = "dsu"))]
    let mut field = Field {
      stride: width + 1,
      score_red: 0,
      score_black: 0,
      moves: Vec::with_capacity(length),
      points: vec![Cell::new(false); length].into(),
      neighbor_offsets,
      changes: Vec::with_capacity(length),
      zobrist,
      hash: 0,
      chains: [
        (Vec::with_capacity(length), 0),
        (Vec::with_capacity(length), 0),
        (Vec::with_capacity(length), 0),
      ],
      captured_points: Vec::with_capacity(length),
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

  #[inline]
  fn update_hash(&mut self, pos: Pos, player: Player) {
    self.hash ^= self.zobrist.hashes[self.length() * player as usize + pos]
  }

  fn capture(&mut self, chain_index: usize, inside_pos: Pos, player: Player) -> bool {
    let mut captured_count = 0i32;
    let mut freed_count = 0i32;
    self.captured_points.clear();
    for &pos in &self.chains[chain_index].0 {
      self.points[pos].set_tag();
    }
    wave(&mut self.q, self.stride, inside_pos, |pos| {
      let cell = self.points[pos];
      if !cell.is_tagged() && !cell.is_bound_player(player) {
        self.points[pos].set_tag();
        self.captured_points.push(pos);
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
      for &pos in self.chains[chain_index].0.iter() {
        self.points[pos].clear_tag();
        self
          .changes
          .last_mut()
          .unwrap()
          .points_changes
          .push((pos, self.points[pos]));
        self.points[pos].set_bound();
      }
      for &pos in &self.captured_points {
        self.points[pos].clear_tag();
        let cell = self.cell(pos);
        self.changes.last_mut().unwrap().points_changes.push((pos, cell));
        if !cell.is_put() {
          if cell.is_captured() {
            self.hash ^= self.zobrist.hashes[self.length() * player.next() as usize + pos];
          }
          self.points[pos].0 = self.points[pos].0 & !(Cell::EMPTY_BASE_BIT | Cell::PLAYER_BIT)
            | Cell::CAPTURED_BIT
            | player.to_bool() as u8;
          self.hash ^= self.zobrist.hashes[self.length() * player as usize + pos];
        } else if cell.get_player() != player {
          self.points[pos].0 = self.points[pos].0 & !Cell::BOUND_BIT | Cell::CAPTURED_BIT;
          self.hash ^= self.zobrist.hashes[self.length() * player.next() as usize + pos]
            ^ self.zobrist.hashes[self.length() * player as usize + pos];
        } else if cell.is_captured() {
          self.points[pos].clear_captured();
          self.hash ^= self.zobrist.hashes[self.length() * player.next() as usize + pos]
            ^ self.zobrist.hashes[self.length() * player as usize + pos];
        }
      }
      true
    } else {
      for &pos in self.chains[chain_index].0.iter() {
        self.points[pos].clear_tag();
      }
      for &pos in &self.captured_points {
        self.points[pos].clear_tag();
        if !self.points[pos].is_put() {
          let cell = self.cell(pos);
          self.changes.last_mut().unwrap().points_changes.push((pos, cell));
          self.points[pos].set_empty_base_player(player);
        }
      }
      false
    }
  }

  #[cfg(feature = "dsu")]
  fn find_dsu_set(&mut self, mut pos: Pos) -> Pos {
    let mut root = pos;
    while self.dsu[root] != root {
      root = self.dsu[root];
    }
    while self.dsu[pos] != root {
      let parent = self.dsu[pos];
      self.save_dsu_value(pos);
      self.dsu[pos] = root;
      pos = parent;
    }
    root
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
    let input_points = get_input_points(self.stride, &self.points, pos, player);
    let input_points_count = input_points.len();
    if input_points_count > 1 {
      let mut total_chains_count = 0;
      let sets: SmallVec<[_; 4]> = input_points
        .iter()
        .map(|&(chain_neighbor, _)| self.find_dsu_set(chain_neighbor.apply(&self.neighbor_offsets, pos)))
        .collect();
      let mut result = false;
      for (i, &set) in sets.iter().enumerate() {
        let mut group: SmallVec<[_; 4]> = SmallVec::new();
        for j in i..input_points_count {
          if sets[j] == set {
            group.push(input_points[j]);
          }
        }
        let mut group_points_count = group.len().saturating_sub(1) as u32;
        if group_points_count > 0 {
          let mut chains_count = 0u32;
          for &(chain_neighbor, captured_pos) in &group {
            if build_chain(
              &self.neighbor_offsets,
              self.stride,
              &mut self.points,
              &mut self.chains[total_chains_count].0,
              pos,
              player,
              chain_neighbor,
            ) {
              self.chains[total_chains_count].1 = captured_pos;
              chains_count += 1;
              total_chains_count += 1;
              if chains_count == group_points_count {
                break;
              }
            } else {
              let len = self.chains[total_chains_count].0.len();
              if len > 0 && len < 4 {
                // If a chain is short it can't form a valid chain when followed in reverse direction
                group_points_count -= 1;
                if chains_count == group_points_count {
                  break;
                }
              }
            }
          }
          if chains_count > 0 {
            result = true;
          }
          if group.len() >= 3 {
            break;
          }
        }
      }
      self.chains[0..total_chains_count].sort_unstable_by_key(|(chain, _)| chain.len());
      for chain_index in 0..total_chains_count {
        self.capture(chain_index, self.chains[chain_index].1, player);
      }
      let parent = self.union_dsu_sets(&sets);
      self.save_dsu_value(pos);
      self.dsu[pos] = parent;
      self.dsu_size[parent] += 1;
      result
    } else {
      if let Some(&(chain_neighbor, _)) = input_points.first() {
        let parent = self.find_dsu_set(chain_neighbor.apply(&self.neighbor_offsets, pos));
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
    let input_points = get_input_points(self.stride, &self.points, pos, player);
    let mut input_points_count = input_points.len().saturating_sub(1);
    if input_points_count > 0 {
      let mut chains_count = 0;
      for (chain_neighbor, captured_pos) in input_points {
        if build_chain(
          &self.neighbor_offsets,
          self.stride,
          &mut self.points,
          &mut self.chains[chains_count].0,
          pos,
          player,
          chain_neighbor,
        ) {
          self.chains[chains_count].1 = captured_pos;
          chains_count += 1;
          if chains_count == input_points_count {
            break;
          }
        } else {
          let len = self.chains[chains_count].0.len();
          if len > 0 && len < 4 {
            // If a chain is short it can't form a valid chain when followed in reverse direction
            input_points_count -= 1;
            if chains_count == input_points_count {
              break;
            }
          }
        }
      }
      self.chains[0..chains_count].sort_unstable_by_key(|(chain, _)| chain.len());
      for chain_index in 0..chains_count {
        self.capture(chain_index, self.chains[chain_index].1, player);
      }
      chains_count > 0
    } else {
      false
    }
  }

  #[inline]
  fn remove_empty_base(&mut self, start_pos: Pos) {
    wave(&mut self.q, self.stride, start_pos, |pos| {
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

            #[cfg(feature = "dsu")]
            if let Some(direction_pos) = self
              .directions_diag(pos)
              .into_iter()
              .find(|&pos| self.points[pos].is_live_players_point(player))
            {
              let parent = self.find_dsu_set(direction_pos);
              self.save_dsu_value(pos);
              self.dsu[pos] = parent;
              self.save_dsu_size_value(parent);
              self.dsu_size[parent] += 1;
            }
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
              let input_points = get_input_points(self.stride, &self.points, bound_pos, next_player);
              for (chain_pos, captured_pos) in input_points {
                if build_chain(
                  &self.neighbor_offsets,
                  self.stride,
                  &mut self.points,
                  &mut self.chains[0].0,
                  bound_pos,
                  next_player,
                  chain_pos,
                ) && is_point_inside_ring(self.stride, pos, &self.chains[0].0)
                {
                  self.capture(0, captured_pos, next_player);
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
        let input_points = get_input_points(self.stride, &self.points, pos, player);
        let input_points_count = input_points.len().saturating_sub(1);
        let mut chains_count = 0;
        for (chain_neighbor, captured_pos) in input_points {
          if !(self.cell(captured_pos).is_captured()
            && self.cell(chain_neighbor.apply(&self.neighbor_offsets, pos)).is_bound())
          {
            continue;
          }
          if find_chain(
            &self.neighbor_offsets,
            self.stride,
            &mut self.points,
            &mut self.chains[chains_count].0,
            pos,
            player,
            chain_neighbor,
          ) {
            result.append(&mut self.chains[chains_count].0);
            chains_count += 1;
            if chains_count == input_points_count {
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
          let input_points = get_input_points(self.stride, &self.points, bound_pos, next_player);
          for (chain_neighbor, _) in input_points {
            if !self
              .cell(chain_neighbor.apply(&self.neighbor_offsets, bound_pos))
              .is_bound()
            {
              continue;
            }
            if find_chain(
              &self.neighbor_offsets,
              self.stride,
              &mut self.points,
              &mut self.chains[0].0,
              bound_pos,
              next_player,
              chain_neighbor,
            ) && is_point_inside_ring(self.stride, pos, &self.chains[0].0)
            {
              return self.chains[0].0.clone();
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
    is_corner(self.width(), self.height(), pos)
  }

  fn non_grounded_points(&mut self) -> (u32, u32) {
    let mut result = (0, 0);
    for &pos in &self.moves {
      let player = self.cell(pos).get_owner().unwrap();
      let mut points = 0;
      let mut grounded = false;
      wave(&mut self.q, self.stride, pos, |pos| {
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
    if self.moves_count() > self.width() as usize * self.height() as usize / 8 {
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
    for y in 0..self.height() {
      for x in 0..self.width() {
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
