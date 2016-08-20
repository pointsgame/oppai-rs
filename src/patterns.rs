use std::collections::{HashSet, HashMap};
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use std::fs::File;
use std::io::Cursor;
use std::cmp;
use rand::Rng;
use tar::Archive;
use spiral::Spiral;
use dfa::{Dfa, DfaState};
use player::Player;
use cell::Cell;
use field::{Pos, Field};
use rotate::*;

const PATTERNS_STR: &'static str = "patterns";

#[derive(Clone, Debug)]
struct Move {
  x: u32,
  y: u32,
  p: f64 // probability
}

struct Pattern {
  p: f64, // priority
  width: u32,
  height: u32,
  moves: Vec<Move>
}

pub struct Patterns {
  min_size: u32,
  dfa: Dfa<usize>,
  patterns: Vec<Pattern>
}

impl Patterns {
  pub fn empty() -> Patterns {
    Patterns {
      min_size: u32::max_value(),
      dfa: Dfa::empty(),
      patterns: Vec::new()
    }
  }

  fn read_header<T: BufRead>(name: &str, input: &mut T, s: &mut String) -> (u32, u32, f64) {
    s.clear();
    input.read_line(s).ok();
    s.pop();
    let mut split = s.split(' ').fuse();
    let width = u32::from_str(split.next().expect("Invalid pattern format: expected width.")).expect("Invalid pattern format: width must be u32.");
    if width < 2 {
      panic!("Minimum allowed width is 2 in the pattern '{}'.", name);
    }
    let height = u32::from_str(split.next().expect("Invalid pattern format: expected height.")).expect("Invalid pattern format: height must be u32.");
    if height < 2 {
      panic!("Minimum allowed height is 2 in the pattern '{}'.", name);
    }
    let priority = f64::from_str(split.next().expect("Invalid pattern format: expected priority.")).expect("Invalid pattern format: priority must be f64.");
    (width, height, priority)
  }

  fn read_pattern<T: BufRead>(input: &mut T, s: &mut String, width: u32, height: u32) {
    s.clear();
    for _ in 0 .. height {
      let last_len = s.len();
      input.read_line(s).ok();
      s.pop();
      assert_eq!((s.len() - last_len) as u32, width)
    }
  }

  fn read_moves<T: BufRead>(name: &str, input: &mut T, s: &mut String, pattern_moves: HashSet<(u32, u32)>) -> Vec<Move> {
    let moves_count = pattern_moves.len();
    let mut moves = Vec::with_capacity(moves_count as usize);
    let mut moves_set = HashSet::new();
    let mut priorities_sum = 0f64;
    for _ in 0 .. moves_count {
      s.clear();
      input.read_line(s).ok();
      s.pop();
      let mut split = s.split(' ').fuse();
      let x = u32::from_str(split.next().expect("Invalid pattern format: expected x coordinate.")).expect("Invalid pattern format: x coordinate must be u32.");
      let y = u32::from_str(split.next().expect("Invalid pattern format: expected x coordinate.")).expect("Invalid pattern format: x coordinate must be u32.");
      let p = f64::from_str(split.next().expect("Invalid pattern format: expected probability.")).expect("Invalid pattern format: probability must be f64.");
      moves_set.insert((x, y));
      priorities_sum += p;
      moves.push(Move {
        x: x,
        y: y,
        p: p
      });
    }
    if moves_set != pattern_moves {
      panic!("Moves list does not match moves in the pattern named '{}'.", name);
    }
    s.clear();
    input.read_line(s).ok();
    if !s.is_empty() {
      panic!("Pattern '{}' should end with empty line.", name);
    }
    for m in &mut moves {
      m.p /= priorities_sum;
    }
    moves
  }

  fn covering_spiral_length(side_of_square: u32) -> u32 {
    let x = side_of_square / 2 + 1;
    let y = (1 - side_of_square % 2) * side_of_square * 2;
    (8 * x - 13) * x + 6 - y
  }

  fn build_dfa(name: &str, width: u32, height: u32, pattern: usize, rotation: u32, s: &str) -> Dfa<usize> {
    let (rotated_width, rotated_height) = rotate_sizes(width, height, rotation);
    let center_x = (rotated_width - 1) / 2;
    let center_y = (rotated_height - 1) / 2;
    let spiral_length = Patterns::covering_spiral_length(cmp::max(width, height)) as usize;
    let mut states = Vec::with_capacity(spiral_length + 2);
    let fs = spiral_length; // "Found" state.
    let nfs = spiral_length + 1; // "Not found" state.
    let s_bytes = s.as_bytes();
    for (i, (shift_x, shift_y)) in Spiral::new().into_iter().take(spiral_length).enumerate() {
      let nxt = i + 1;
      let rotated_x = center_x as i32 + shift_x;
      let rotated_y = center_y as i32 + shift_y;
      let state = if rotated_x >= 0 && rotated_x < rotated_width as i32 && rotated_y >= 0 && rotated_y < rotated_height as i32 {
        let (x, y) = rotate_back(rotated_width, rotated_height, rotated_x as u32, rotated_y as u32, rotation);
        let pos = y * width + x;
        match s_bytes[pos as usize] as char {
          '.' | '+' => DfaState::new(nxt, nfs, nfs, nfs, false, HashSet::with_capacity(0)),
          '?' => DfaState::new(nxt, nxt, nxt, nfs, false, HashSet::with_capacity(0)),
          '*' => DfaState::new(nxt, nxt, nxt, nxt, false, HashSet::with_capacity(0)),
          'R' => DfaState::new(nfs, nxt, nfs, nfs, false, HashSet::with_capacity(0)),
          'B' => DfaState::new(nfs, nfs, nxt, nfs, false, HashSet::with_capacity(0)),
          'r' => DfaState::new(nxt, nxt, nfs, nfs, false, HashSet::with_capacity(0)),
          'b' => DfaState::new(nxt, nfs, nxt, nfs, false, HashSet::with_capacity(0)),
          '#' => DfaState::new(nfs, nfs, nfs, nxt, false, HashSet::with_capacity(0)),
          c => panic!("Invalid character in the pattern '{}': {}", name, c)
        }
      } else {
        DfaState::new(nxt, nxt, nxt, nxt, false, HashSet::with_capacity(0))
      };
      states.push(state);
    }
    let mut c = 0;
    for state in states.iter().rev().take(spiral_length - 1) {
      if state.empty == nfs || state.red == nfs || state.black == nfs || state.bad == nfs {
        break;
      }
      c += 1;
    }
    let new_fs = fs - c;
    let new_nfs = nfs - c;
    if c > 0 {
      states.truncate(spiral_length - c);
      for state in &mut states {
        if state.empty == nfs {
          state.empty = new_nfs;
        }
        if state.red == nfs {
          state.red = new_nfs;
        }
        if state.black == nfs {
          state.black = new_nfs;
        }
        if state.bad == nfs {
          state.bad = new_nfs;
        }
      }
    }
    let mut patterns = HashSet::with_capacity(1);
    patterns.insert(pattern);
    states.push(DfaState::new(new_fs, new_fs, new_fs, new_fs, true, patterns));
    states.push(DfaState::new(new_nfs, new_nfs, new_nfs, new_nfs, true, HashSet::with_capacity(0)));
    Dfa::new(states)
  }

  fn get_pattern_moves(name: &str, width: u32, s: &str) -> HashSet<(u32, u32)> {
    let mut pattern_moves = HashSet::new();
    for i in s.chars().enumerate().filter(|&(_, c)| c == '+').map(|(i, _)| i as u32) {
      pattern_moves.insert((i % width, i / width));
    }
    if pattern_moves.is_empty() {
      panic!("Moves are not defined in the pattern '{}'.", name);
    }
    pattern_moves
  }

  fn add<T: BufRead>(&mut self, name: &str, input: &mut T, s: &mut String, pattern_s: &mut String) {
    let (width, height, priority) = Patterns::read_header(name, input, s);
    if width < self.min_size {
      self.min_size = width;
    }
    if height < self.min_size {
      self.min_size = height;
    }
    Patterns::read_pattern(input, pattern_s, width, height);
    let pattern_moves = Patterns::get_pattern_moves(name, width, pattern_s);
    let moves = Patterns::read_moves(name, input, s, pattern_moves);
    let mut dfa = Dfa::empty();
    for rotation in 0 .. 8 {
      let cur_dfa = Patterns::build_dfa(name, width, height, self.patterns.len(), rotation, pattern_s);
      dfa = dfa.product(&cur_dfa);
      info!(target: PATTERNS_STR, "DFA total size: {}.", self.dfa.states_count());
      let (rotated_width, rotated_height) = rotate_sizes(width, height, rotation);
      self.patterns.push(Pattern {
        p: priority,
        width: rotated_width,
        height: rotated_height,
        moves: moves.iter().map(|m| {
          let (x, y) = rotate(width, height, m.x, m.y, rotation);
          Move {
            x: x,
            y: y,
            p: m.p
          }
        }).collect()
      });
    }
    self.dfa = self.dfa.product(&dfa);
  }

  pub fn add_str(&mut self, string: &str) {
    let mut s = String::new();
    let mut pattern_s = String::new();
    self.add("<none>", &mut Cursor::new(string.as_bytes()), &mut s, &mut pattern_s);
  }

  pub fn add_tar(&mut self, file: File) {
    let mut archive = Archive::new(file);
    let mut s = String::new();
    let mut pattern_s = String::new();
    let iter = archive.entries().expect("Reading of tar archive is failed.").into_iter().map(|file| file.expect("Reading of file in tar archive is failed."));
    for file in iter.filter(|file| file.header().entry_type().is_file()) {
      let name = file.header().path().ok()
        .and_then(|path| path.to_str().map(|s| s.to_owned()))
        .unwrap_or_else(|| "<unknown>".to_owned());
      info!(target: PATTERNS_STR, "Loading pattern '{}'", name);
      let mut input = BufReader::new(file);
      self.add(&name, &mut input, &mut s, &mut pattern_s);
    }
  }

  pub fn find(&self, field: &Field, player: Player, first_match: bool) -> Vec<(Pos, f64)> {
    if self.dfa.is_empty() || field.width() < self.min_size - 2 || field.height() < self.min_size - 2 {
      return Vec::with_capacity(0);
    }
    let mut priorities_sum = 0f64;
    let mut moves_count = 0usize;
    let mut matched = Vec::new();
    let left_border = (self.min_size as i32 - 1) / 2 - 1;
    let right_border = self.min_size as i32 / 2 - 1;
    let inv_color = player == Player::Black;
    for y in left_border .. field.height() as i32 - right_border {
      for x in left_border .. field.width() as i32 - right_border {
        let patterns = self.dfa.run(&mut Spiral::new().into_iter().map(|(shift_x, shift_y)| {
          let cur_x = x + shift_x;
          let cur_y = y + shift_y;
          if cur_x >= 0 && cur_x < field.width() as i32 && cur_y >= 0 && cur_y < field.height() as i32 {
            let pos = field.to_pos(cur_x as u32, cur_y as u32);
            field.cell(pos)
          } else {
            Cell::new(true)
          }
        }), inv_color, first_match);
        for &pattern_number in patterns {
          let pattern = &self.patterns[pattern_number];
          info!(target: PATTERNS_STR, "Found pattern {} ({} with rotation {}) at ({}, {}).", pattern_number, pattern_number / 8, pattern_number % 8, x - (pattern.width as i32 - 1) / 2, y - (pattern.height as i32 - 1) / 2);
          priorities_sum += pattern.p;
          moves_count += pattern.moves.len();
          matched.push((pattern_number, x, y));
        }
      }
    }
    let mut result = Vec::with_capacity(moves_count);
    for (pattern_number, center_x, center_y) in matched {
      let pattern = &self.patterns[pattern_number];
      for &Move { x, y, p: probability } in &pattern.moves {
        let move_x = (center_x - (pattern.width as i32 - 1) / 2 + x as i32) as u32;
        let move_y = (center_y - (pattern.height as i32 - 1) / 2 + y as i32) as u32;
        result.push((field.to_pos(move_x, move_y), probability * pattern.p / priorities_sum));
      }
    }
    info!(target: PATTERNS_STR, "Found moves: {:?}.", result.iter().map(|&(pos, p)| (field.to_x(pos), field.to_y(pos), p)).collect::<Vec<(u32, u32, f64)>>());
    result
  }

  pub fn find_sorted(&self, field: &Field, player: Player, first_match: bool) -> Vec<(Pos, f64)> {
    let moves = self.find(field, player, first_match);
    let mut map = HashMap::with_capacity(moves.len());
    for (pos, p) in moves {
      let sum_p = map.get(&pos).cloned().unwrap_or(0f64) + p;
      map.insert(pos, sum_p);
    }
    let mut result = map.into_iter().collect::<Vec<(Pos, f64)>>();
    result.sort_by(|&(_, a), &(_, b)| b.partial_cmp(&a).expect("Cann't compare f64 types."));
    info!(target: PATTERNS_STR, "Found sorted moves: {:?}.", result.iter().map(|&(pos, p)| (field.to_x(pos), field.to_y(pos), p)).collect::<Vec<(u32, u32, f64)>>());
    result
  }

  pub fn find_foreground(&self, field: &Field, player: Player, first_match: bool) -> Option<Pos> {
    self.find_sorted(field, player, first_match).first().map(|&(pos, _)| pos)
  }

  pub fn find_rand<T: Rng>(&self, field: &Field, player: Player, first_match: bool, rng: &mut T) -> Option<Pos> {
    let moves = self.find_sorted(field, player, first_match);
    if moves.is_empty() {
      return None;
    }
    let rand = rng.gen();
    let mut sum = 0f64;
    let mut idx = 0;
    while sum < rand && idx < moves.len() {
      sum += moves[idx].1;
      idx += 1;
    }
    Some(moves[idx - 1].0)
  }
}
