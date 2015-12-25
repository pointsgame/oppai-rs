use std::{mem, iter};
use std::collections::LinkedList;
use std::sync::Arc;
use player::Player;
use cell::Cell;
use zobrist::Zobrist;

pub type Pos = usize;

#[derive(Clone, PartialEq)]
struct FieldChange {
  score_red: i32,
  score_black: i32,
  hash: u64,
  points_changes: Vec<(Pos, Cell)>,
  dsu_changes: Vec<(Pos, Pos)>,
  dsu_size_change: Option<(Pos, u32)>
}

#[derive(Clone, Copy, PartialEq)]
enum IntersectionState {
  None,
  Up,
  Target,
  Down
}

#[derive(Clone, PartialEq)]
pub struct Field {
  width: u32,
  height: u32,
  length: Pos,
  score_red: i32,
  score_black: i32,
  points_seq: Vec<Pos>,
  points: Vec<Cell>,
  dsu: Vec<Pos>,
  dsu_size: Vec<u32>,
  changes: Vec<FieldChange>,
  zobrist: Arc<Zobrist>,
  hash: u64
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
  n(width, pos1)  == pos2 ||
  s(width, pos1)  == pos2 ||
  w(pos1)         == pos2 ||
  e(pos1)         == pos2 ||
  nw(width, pos1) == pos2 ||
  ne(width, pos1) == pos2 ||
  sw(width, pos1) == pos2 ||
  se(width, pos1) == pos2
}

fn get_intersection_state(width: u32, pos: Pos, next_pos: Pos) -> IntersectionState {
  let pos_x = to_x(width, pos);
  let pos_y = to_y(width, pos);
  let next_pos_x = to_x(width, next_pos);
  let next_pos_y = to_y(width, next_pos);
  if next_pos_x <= pos_x {
    match next_pos_y as i32 - pos_y as i32 {
      1  => IntersectionState::Up,
      0  => IntersectionState::Target,
      -1 => IntersectionState::Down,
      _  => IntersectionState::None
    }
  } else {
    IntersectionState::None
  }
}

pub fn is_point_inside_ring(width: u32, pos: Pos, ring: &LinkedList<Pos>) -> bool {
  let mut intersections = 0u32;
  let mut state = IntersectionState::None;
  for &next_pos in ring {
    match get_intersection_state(width, pos, next_pos) {
      IntersectionState::None => {
        state = IntersectionState::None;
      },
      IntersectionState::Up => {
        if state == IntersectionState::Down {
          intersections += 1;
        }
        state = IntersectionState::Up;
      },
      IntersectionState::Down => {
        if state == IntersectionState::Up {
          intersections += 1;
        }
        state = IntersectionState::Down;
      },
      IntersectionState::Target => { }
    }
  }
  if state == IntersectionState::Up || state == IntersectionState::Down {
    let mut iter = ring.iter();
    let mut begin_state = get_intersection_state(width, pos, *iter.next().unwrap());
    while begin_state == IntersectionState::Target {
      begin_state = get_intersection_state(width, pos, *iter.next().unwrap());
    }
    if state == IntersectionState::Up && begin_state == IntersectionState::Down || state == IntersectionState::Down && begin_state == IntersectionState::Up {
      intersections += 1;
    }
  }
  intersections % 2 == 1
}

#[inline]
pub fn square(width: u32, pos1: Pos, pos2: Pos) -> i32 {
  (to_x(width, pos1) * to_y(width, pos2)) as i32 - (to_y(width, pos1) * to_x(width, pos2)) as i32
}

pub fn wave<F: FnMut(Pos) -> bool>(width: u32, start_pos: Pos, mut cond: F) {
  if !cond(start_pos) {
    return;
  }
  let mut queue = LinkedList::new();
  queue.push_back(start_pos);
  while let Some(pos) = queue.pop_front() {
    let n_pos = n(width, pos);
    let s_pos = s(width, pos);
    let w_pos = w(pos);
    let e_pos = e(pos);
    if cond(n_pos) {
      queue.push_back(n_pos);
    }
    if cond(s_pos) {
      queue.push_back(s_pos);
    }
    if cond(w_pos) {
      queue.push_back(w_pos);
    }
    if cond(e_pos) {
      queue.push_back(e_pos);
    }
  }
}

#[inline]
pub fn manhattan(width: u32, pos1: Pos, pos2: Pos) -> u32 {
  (i32::abs(to_x(width, pos1) as i32 - to_x(width, pos2) as i32) + i32::abs(to_y(width, pos1) as i32 - to_y(width, pos2) as i32)) as u32
}

impl Field {
  #[inline]
  pub fn length(&self) -> Pos {
    self.length
  }

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
  pub fn min_pos(&self) -> Pos {
    min_pos(self.width)
  }

  #[inline]
  pub fn max_pos(&self) -> Pos {
    max_pos(self.width, self.height)
  }

  #[inline]
  pub fn is_near(&self, pos1: Pos, pos2: Pos) -> bool {
    is_near(self.width, pos1, pos2)
  }

  #[inline]
  pub fn get_player(&self, pos: Pos) -> Player {
    self.points[pos].get_player()
  }

  #[inline]
  fn set_player(&mut self, pos: Pos, player: Player) {
    self.points[pos].set_player(player)
  }

  #[inline]
  pub fn is_put(&self, pos: Pos) -> bool {
    self.points[pos].is_put()
  }

  #[inline]
  fn set_put(&mut self, pos: Pos) {
    self.points[pos].set_put()
  }

  #[inline]
  fn clear_put(&mut self, pos: Pos) {
    self.points[pos].clear_put()
  }

  #[inline]
  pub fn is_captured(&self, pos: Pos) -> bool {
    self.points[pos].is_captured()
  }

  #[inline]
  fn set_captured(&mut self, pos: Pos) {
    self.points[pos].set_captured()
  }

  #[inline]
  fn clear_captured(&mut self, pos: Pos) {
    self.points[pos].clear_captured()
  }

  #[inline]
  pub fn is_bound(&self, pos: Pos) -> bool {
    self.points[pos].is_bound()
  }

  #[inline]
  fn set_bound(&mut self, pos: Pos) {
    self.points[pos].set_bound()
  }

  #[inline]
  fn clear_bound(&mut self, pos: Pos) {
    self.points[pos].clear_bound()
  }

  #[inline]
  pub fn is_empty_base(&self, pos: Pos) -> bool {
    self.points[pos].is_empty_base()
  }

  #[inline]
  fn set_empty_base(&mut self, pos: Pos) {
    self.points[pos].set_empty_base()
  }

  #[inline]
  fn clear_empty_base(&mut self, pos: Pos) {
    self.points[pos].clear_empty_base()
  }

  #[inline]
  pub fn is_bad(&self, pos: Pos) -> bool {
    self.points[pos].is_bad()
  }

  #[inline]
  pub fn set_bad(&mut self, pos: Pos) {
    self.points[pos].set_bad()
  }

  #[inline]
  pub fn clear_bad(&mut self, pos: Pos) {
    self.points[pos].clear_bad()
  }

  #[inline]
  pub fn is_tagged(&self, pos: Pos) -> bool {
    self.points[pos].is_tagged()
  }

  #[inline]
  pub fn set_tag(&mut self, pos: Pos) {
    self.points[pos].set_tag()
  }

  #[inline]
  pub fn clear_tag(&mut self, pos: Pos) {
    self.points[pos].clear_tag()
  }

  #[inline]
  pub fn get_owner(&self, pos: Pos) -> Option<Player> {
    self.points[pos].get_owner()
  }

  #[inline]
  pub fn is_owner(&self, pos: Pos, player: Player) -> bool {
    self.points[pos].is_owner(player)
  }

  #[inline]
  pub fn get_players_point(&self, pos: Pos) -> Option<Player> {
    self.points[pos].get_players_point()
  }

  #[inline]
  pub fn is_players_point(&self, pos: Pos, player: Player) -> bool {
    self.points[pos].is_players_point(player)
  }

  #[inline]
  pub fn get_live_players_point(&self, pos: Pos) -> Option<Player> {
    self.points[pos].get_live_players_point()
  }

  #[inline]
  pub fn is_live_players_point(&self, pos: Pos, player: Player) -> bool {
    self.points[pos].is_live_players_point(player)
  }

  #[inline]
  pub fn is_players_empty_base(&self, pos: Pos, player: Player) -> bool {
    self.points[pos].is_players_empty_base(player)
  }

  #[inline]
  pub fn get_empty_base_player(&self, pos: Pos) -> Option<Player> {
    self.points[pos].get_empty_base_player()
  }

  #[inline]
  fn just_put_point(&mut self, pos: Pos, player: Player) {
    self.points[pos].put_point(player)
  }

  #[inline]
  fn set_empty_base_player(&mut self, pos: Pos, player: Player) {
    self.points[pos].set_empty_base_player(player)
  }

  #[inline]
  pub fn is_bound_player(&self, pos: Pos, player: Player) -> bool {
    self.points[pos].is_bound_player(player)
  }

  #[inline]
  pub fn is_putting_allowed(&self, pos: Pos) -> bool {
    pos < self.length && self.points[pos].is_putting_allowed()
  }

  pub fn has_near_points(&self, center_pos: Pos, player: Player) -> bool {
    self.is_live_players_point(self.n(center_pos), player)  ||
    self.is_live_players_point(self.s(center_pos), player)  ||
    self.is_live_players_point(self.w(center_pos), player)  ||
    self.is_live_players_point(self.e(center_pos), player)  ||
    self.is_live_players_point(self.nw(center_pos), player) ||
    self.is_live_players_point(self.ne(center_pos), player) ||
    self.is_live_players_point(self.sw(center_pos), player) ||
    self.is_live_players_point(self.se(center_pos), player)
  }

  pub fn number_near_points(&self, center_pos: Pos, player: Player) -> u32 {
    let mut result = 0u32;
    if self.is_live_players_point(self.n(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.s(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.w(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.e(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.nw(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.ne(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.sw(center_pos), player) { result += 1; }
    if self.is_live_players_point(self.se(center_pos), player) { result += 1; }
    result
  }

  pub fn number_near_groups(&self, center_pos: Pos, player: Player) -> u32 {
    let mut result = 0u32;
    if !self.is_live_players_point(self.w(center_pos), player) && (self.is_live_players_point(self.nw(center_pos), player) || self.is_live_players_point(self.n(center_pos), player)) { result += 1; }
    if !self.is_live_players_point(self.s(center_pos), player) && (self.is_live_players_point(self.sw(center_pos), player) || self.is_live_players_point(self.w(center_pos), player)) { result += 1; }
    if !self.is_live_players_point(self.e(center_pos), player) && (self.is_live_players_point(self.se(center_pos), player) || self.is_live_players_point(self.s(center_pos), player)) { result += 1; }
    if !self.is_live_players_point(self.n(center_pos), player) && (self.is_live_players_point(self.ne(center_pos), player) || self.is_live_players_point(self.e(center_pos), player)) { result += 1; }
    result
  }

  pub fn new(width: u32, height: u32, zobrist: Arc<Zobrist>) -> Field {
    let length = length(width, height);
    let mut field = Field {
      width: width,
      height: height,
      length: length,
      score_red: 0,
      score_black: 0,
      points_seq: Vec::with_capacity(length),
      points: iter::repeat(Cell::new(false)).take(length).collect(),
      dsu: (0 .. length).collect(),
      dsu_size: iter::repeat(1).take(length).collect(),
      changes: Vec::with_capacity(length),
      zobrist: zobrist,
      hash: 0
    };
    let max_pos = field.max_pos();
    for x in 0 .. width as Pos + 2 {
      field.set_bad(x);
      field.set_bad(max_pos + 1 + x);
    }
    for y in 1 .. height as Pos + 1 {
      field.set_bad(y * (width as Pos + 2));
      field.set_bad((y + 1) * (width as Pos + 2) - 1);
    }
    field
  }

  #[inline]
  fn save_pos_value(&mut self, pos: Pos) {
    self.changes.last_mut().unwrap().points_changes.push((pos, self.points[pos]));
  }

  #[inline]
  fn save_dsu_value(&mut self, pos: Pos) {
    self.changes.last_mut().unwrap().dsu_changes.push((pos, self.dsu[pos]));
  }

  #[inline]
  fn save_dsu_size_value(&mut self, pos: Pos) {
    self.changes.last_mut().unwrap().dsu_size_change = Some((pos, self.dsu_size[pos]));
  }

  fn get_input_points(&self, center_pos: Pos, player: Player) -> Vec<(Pos, Pos)> {
    let mut inp_points = Vec::with_capacity(4);
    if !self.is_live_players_point(self.w(center_pos), player) {
      if self.is_live_players_point(self.nw(center_pos), player) {
        inp_points.push((self.nw(center_pos), self.w(center_pos)));
      } else if self.is_live_players_point(self.n(center_pos), player) {
        inp_points.push((self.n(center_pos), self.w(center_pos)));
      }
    }
    if !self.is_live_players_point(self.s(center_pos), player) {
      if self.is_live_players_point(self.sw(center_pos), player) {
        inp_points.push((self.sw(center_pos), self.s(center_pos)));
      } else if self.is_live_players_point(self.w(center_pos), player) {
        inp_points.push((self.w(center_pos), self.s(center_pos)));
      }
    }
    if !self.is_live_players_point(self.e(center_pos), player) {
      if self.is_live_players_point(self.se(center_pos), player) {
        inp_points.push((self.se(center_pos), self.e(center_pos)));
      } else if self.is_live_players_point(self.s(center_pos), player) {
        inp_points.push((self.s(center_pos), self.e(center_pos)));
      }
    }
    if !self.is_live_players_point(self.n(center_pos), player) {
      if self.is_live_players_point(self.ne(center_pos), player) {
        inp_points.push((self.ne(center_pos), self.n(center_pos)));
      } else if self.is_live_players_point(self.e(center_pos), player) {
        inp_points.push((self.e(center_pos), self.n(center_pos)));
      }
    }
    inp_points
  }

  #[inline]
  fn square(&self, pos1: Pos, pos2: Pos) -> i32 {
    square(self.width, pos1, pos2)
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
    } else {
      if pos == self.e(center_pos) || pos == self.se(center_pos) {
        self.sw(center_pos)
      } else {
        self.nw(center_pos)
      }
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
    } else {
      if pos == self.e(center_pos) {
        self.se(center_pos)
      } else if pos == self.se(center_pos) {
        self.s(center_pos)
      } else if pos == self.s(center_pos) {
        self.sw(center_pos)
      } else {
        self.w(center_pos)
      }
    }
  }

  fn build_chain(&mut self, start_pos: Pos, player: Player, direction_pos: Pos) -> Option<LinkedList<Pos>> {
    let mut chain = LinkedList::new();
    chain.push_back(start_pos);
    let mut pos = direction_pos;
    let mut center_pos = start_pos;
    let mut base_square = self.square(center_pos, pos);
    loop {
      if self.is_tagged(pos) {
        while *chain.back().unwrap() != pos {
          self.clear_tag(*chain.back().unwrap());
          chain.pop_back();
        }
      } else {
        self.set_tag(pos);
        chain.push_back(pos);
      }
      mem::swap(&mut pos, &mut center_pos);
      pos = self.get_first_next_pos(center_pos, pos);
      while !self.is_live_players_point(pos, player) {
        pos = self.get_next_pos(center_pos, pos);
      }
      base_square += self.square(center_pos, pos);
      if pos == start_pos { break }
    }
    for &pos in &chain {
      self.clear_tag(pos);
    }
    if base_square < 0 && chain.len() > 2 {
      Some(chain)
    } else {
      None
    }
  }

  #[inline]
  pub fn is_point_inside_ring(&self, pos: Pos, ring: &LinkedList<Pos>) -> bool {
    is_point_inside_ring(self.width, pos, ring)
  }

  #[inline]
  fn update_hash(&mut self, pos: Pos, player: Player) {
    if player == Player::Red {
      self.hash ^= self.zobrist.get_hash(pos);
    } else {
      self.hash ^= self.zobrist.get_hash(self.length + pos);
    }
  }

  fn capture(&mut self, chain: &LinkedList<Pos>, inside_pos: Pos, player: Player) -> bool {
    let mut captured_count = 0i32;
    let mut freed_count = 0i32;
    let mut captured_points = LinkedList::new();
    for &pos in chain {
      self.set_tag(pos);
    }
    wave(self.width, inside_pos, |pos| {
      if !self.is_tagged(pos) && !self.is_bound_player(pos, player) {
        self.set_tag(pos);
        captured_points.push_back(pos);
        if self.is_put(pos) {
          if self.get_player(pos) != player {
            captured_count += 1;
          } else if self.is_captured(pos) {
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
        },
        Player::Black => {
          self.score_black += captured_count;
          self.score_red -= freed_count;
        }
      }
      for &pos in chain.iter() {
        self.clear_tag(pos);
        self.save_pos_value(pos);
        self.set_bound(pos);
      }
      for &pos in &captured_points {
        self.clear_tag(pos);
        self.save_pos_value(pos);
        if !self.is_put(pos) {
          if !self.is_captured(pos) {
            self.set_captured(pos);
          } else {
            self.update_hash(pos, player.next());
          }
          self.clear_empty_base(pos);
          self.set_player(pos, player);
          self.update_hash(pos, player);
        } else {
          if self.get_player(pos) != player {
            self.set_captured(pos);
            self.update_hash(pos, player.next());
            self.update_hash(pos, player);
          } else if self.is_captured(pos) {
            self.clear_captured(pos);
            self.update_hash(pos, player.next());
            self.update_hash(pos, player);
          }
        }
      }
      true
    } else {
      for &pos in chain.iter() {
        self.clear_tag(pos);
      }
      for &pos in &captured_points {
        self.clear_tag(pos);
        if !self.is_put(pos) {
          self.save_pos_value(pos);
          self.set_empty_base_player(pos, player);
        }
      }
      false
    }
  }

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

  fn find_captures(&mut self, pos: Pos, player: Player) -> bool {
    let input_points = self.get_input_points(pos, player);
    let input_points_count = input_points.len();
    if input_points_count > 1 {
      let mut sets = Vec::with_capacity(input_points_count);
      for &(chain_pos, _) in &input_points {
        sets.push(self.find_dsu_set(chain_pos));
      }
      let mut group = Vec::with_capacity(4);
        let mut result = false;
      for (i, &set) in sets.iter().enumerate() {
        group.clear();
        for j in i .. input_points_count {
          if sets[j] == set {
            group.push(input_points[j]);
          }
        }
        let group_points_count = group.len() as u32;
        if group_points_count > 1 {
          let mut chains_count = 0u32;
          for &(chain_pos, captured_pos) in &group {
            if let Some(chain) = self.build_chain(pos, player, chain_pos) {
              self.capture(&chain, captured_pos, player);
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

  #[inline]
  fn remove_empty_base(&mut self, start_pos: Pos) {
    wave(self.width, start_pos, |pos| {
      if self.is_empty_base(pos) {
        self.save_pos_value(pos);
        self.clear_empty_base(pos);
        true
      } else {
        false
      }
    })
  }

  pub fn put_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.is_putting_allowed(pos) {
      let change = FieldChange {
        score_red: self.score_red,
        score_black: self.score_black,
        hash: self.hash,
        points_changes: Vec::new(),
        dsu_changes: Vec::new(),
        dsu_size_change: None
      };
      self.changes.push(change);
      self.save_pos_value(pos);
      self.update_hash(pos, player);
      match self.get_empty_base_player(pos) {
        Some(empty_base_player) => {
          self.just_put_point(pos, player);
          if empty_base_player == player {
            self.clear_empty_base(pos);
          } else {
            if self.find_captures(pos, player) {
              self.remove_empty_base(pos);
            } else {
              let next_player = player.next();
              let mut bound_pos = pos;
              'outer: loop {
                bound_pos = self.w(bound_pos);
                while !self.is_players_point(bound_pos, next_player) {
                  bound_pos = self.w(bound_pos);
                }
                let input_points = self.get_input_points(bound_pos, next_player);
                for (chain_pos, captured_pos) in input_points {
                  match self.build_chain(bound_pos, next_player, chain_pos) {
                    Some(chain) => {
                      if self.is_point_inside_ring(pos, &chain) {
                        self.capture(&chain, captured_pos, next_player);
                        break 'outer
                      }
                    },
                    None => { }
                  }
                }
              }
            }
          }
        },
        None => {
          self.just_put_point(pos, player);
          self.find_captures(pos, player);
        }
      }
      self.points_seq.push(pos);
      true
    } else {
      false
    }
  }

  pub fn undo(&mut self) -> bool {
    if let Some(change) = self.changes.pop() {
      self.points_seq.pop();
      self.score_red = change.score_red;
      self.score_black = change.score_black;
      self.hash = change.hash;
      for (pos, cell) in change.points_changes.into_iter().rev() {
        self.points[pos] = cell;
      }
      for (pos, dsu_value) in change.dsu_changes.into_iter().rev() {
        self.dsu[pos] = dsu_value;
      }
      if let Some((pos, dsu_size)) = change.dsu_size_change {
        self.dsu_size[pos] = dsu_size;
      }
      true
    } else {
      false
    }
  }

  #[inline]
  pub fn moves_count(&self) -> usize {
    self.points_seq.len()
  }

  #[inline]
  pub fn points_seq(&self) -> &Vec<Pos> {
    &self.points_seq
  }

  #[inline]
  pub fn hash(&self) -> u64 {
    self.hash
  }

  #[inline]
  pub fn hash_at(&self, move_number: usize) -> Option<u64> {
    let moves_count = self.moves_count();
    if move_number < moves_count {
      Some(self.changes[move_number].hash)
    } else if move_number == moves_count {
      Some(self.hash)
    } else {
      None
    }
  }

  #[inline]
  pub fn last_player(&self) -> Option<Player> {
    self.points_seq.last().map(|&pos| self.get_player(pos))
  }

  #[inline]
  pub fn width(&self) -> u32 {
    self.width
  }

  #[inline]
  pub fn height(&self) -> u32 {
    self.height
  }

  #[inline]
  pub fn cur_player(&self) -> Player {
    self.last_player().unwrap_or(Player::Black).next()
  }

  #[inline]
  pub fn captured_count(&self, player: Player) -> i32 {
    match player {
      Player::Red => self.score_red,
      Player::Black => self.score_black
    }
  }

  #[inline]
  pub fn score(&self, player: Player) -> i32 {
    match player {
      Player::Red => self.score_red - self.score_black,
      Player::Black => self.score_black - self.score_red
    }
  }

  #[inline]
  pub fn get_delta_score(&self, player: Player) -> i32 {
    self.score(player) - self.changes.last().map_or(0, |change| {
      match player {
        Player::Red => change.score_red - change.score_black,
        Player::Black => change.score_black - change.score_red
      }
    })
  }

  #[inline]
  pub fn zobrist(&self) -> &Zobrist {
    &self.zobrist
  }

  #[inline]
  pub fn cell(&self, pos: Pos) -> Cell {
    self.points[pos]
  }
}
