use crate::dfa::{Dfa, DfaState};
use crate::rotate::*;
use crate::spiral::Spiral;
use oppai_field::cell::Cell;
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use rand::Rng;
use std::{
  cmp,
  collections::{HashMap, HashSet},
  fs::File,
  io::Read,
  str::FromStr,
};
use tar::Archive;

const PATTERNS_STR: &str = "patterns";

#[derive(Clone, Debug)]
struct Move {
  x: i32,
  y: i32,
  p: f64, // priority
}

pub struct Patterns {
  min_size: u32,
  dfa: Dfa<Move>,
}

impl Patterns {
  pub fn empty() -> Patterns {
    Patterns {
      min_size: u32::max_value(),
      dfa: Dfa::empty(),
    }
  }

  fn parse_header(name: &str, string: &str) -> (u32, u32, f64) {
    let mut split = string.split_whitespace().fuse();
    let width = split
      .next()
      .and_then(|s| u32::from_str(s).ok())
      .unwrap_or_else(|| panic!("Invalid format of pattern '{}': expected width of type u32.", name));
    if width < 2 {
      panic!("Minimum allowed width is 2 in the pattern '{}'.", name);
    }
    let height = split
      .next()
      .and_then(|s| u32::from_str(s).ok())
      .unwrap_or_else(|| panic!("Invalid format of pattern '{}': expected height of type u32.", name));
    if height < 2 {
      panic!("Minimum allowed height is 2 in the pattern '{}'.", name);
    }
    let priority = split
      .next()
      .and_then(|s| f64::from_str(s).ok())
      .unwrap_or_else(|| panic!("Invalid format of pattern '{}': expected priority of type f64.", name));
    (width, height, priority)
  }

  fn parse_move(name: &str, string: &str) -> Move {
    let mut split = string.split(' ').fuse();
    let x = split.next().and_then(|s| u32::from_str(s).ok()).unwrap_or_else(|| {
      panic!(
        "Invalid format of pattern '{}': expected x coordinate of type u32.",
        name
      )
    });
    let y = split.next().and_then(|s| u32::from_str(s).ok()).unwrap_or_else(|| {
      panic!(
        "Invalid format of pattern '{}': expected y coordinate of type u32.",
        name
      )
    });
    let p = split.next().and_then(|s| f64::from_str(s).ok()).unwrap_or_else(|| {
      panic!(
        "Invalid format of pattern '{}': expected probability of type f64.",
        name
      )
    });
    Move {
      x: x as i32,
      y: y as i32,
      p,
    }
  }

  fn covering_spiral_length(side_of_square: u32) -> u32 {
    let x = side_of_square / 2 + 1;
    let y = (1 - side_of_square % 2) * side_of_square * 2;
    (8 * x - 13) * x + 6 - y
  }

  fn build_dfa(name: &str, width: u32, height: u32, moves: &[Move], rotation: u32, s: &str) -> Dfa<Move> {
    let (rotated_width, rotated_height) = rotate_sizes(width, height, rotation);
    let center_x = (rotated_width - 1) / 2;
    let center_y = (rotated_height - 1) / 2;
    let spiral_length = Patterns::covering_spiral_length(cmp::max(width, height)) as usize;
    let mut states = Vec::with_capacity(spiral_length + 2);
    let fs = spiral_length; // "Found" state.
    let nfs = spiral_length + 1; // "Not found" state.
    let s_bytes = s.as_bytes();
    for (i, (shift_x, shift_y)) in Spiral::new().take(spiral_length).enumerate() {
      let nxt = i + 1;
      let rotated_x = center_x as i32 + shift_x;
      let rotated_y = center_y as i32 + shift_y;
      let state =
        if rotated_x >= 0 && rotated_x < rotated_width as i32 && rotated_y >= 0 && rotated_y < rotated_height as i32 {
          let (x, y) = rotate_back(
            rotated_width,
            rotated_height,
            rotated_x as u32,
            rotated_y as u32,
            rotation,
          );
          let pos = y * width + x;
          match s_bytes[pos as usize] as char {
            '.' | '+' => DfaState::new(nxt, nfs, nfs, nfs, false, Vec::with_capacity(0)),
            '?' => DfaState::new(nxt, nxt, nxt, nfs, false, Vec::with_capacity(0)),
            '*' => DfaState::new(nxt, nxt, nxt, nxt, false, Vec::with_capacity(0)),
            'R' => DfaState::new(nfs, nxt, nfs, nfs, false, Vec::with_capacity(0)),
            'B' => DfaState::new(nfs, nfs, nxt, nfs, false, Vec::with_capacity(0)),
            'r' => DfaState::new(nxt, nxt, nfs, nfs, false, Vec::with_capacity(0)),
            'b' => DfaState::new(nxt, nfs, nxt, nfs, false, Vec::with_capacity(0)),
            '#' => DfaState::new(nfs, nfs, nfs, nxt, false, Vec::with_capacity(0)),
            c => panic!("Invalid character in the pattern '{}': {}", name, c),
          }
        } else {
          DfaState::new(nxt, nxt, nxt, nxt, false, Vec::with_capacity(0))
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
    let rotated_moves = moves
      .iter()
      .map(|m| {
        let (x, y) = rotate(width, height, m.x as u32, m.y as u32, rotation);
        Move {
          x: x as i32 - center_x as i32,
          y: y as i32 - center_y as i32,
          p: m.p,
        }
      })
      .collect();
    states.push(DfaState::new(new_fs, new_fs, new_fs, new_fs, true, rotated_moves));
    states.push(DfaState::new(
      new_nfs,
      new_nfs,
      new_nfs,
      new_nfs,
      true,
      Vec::with_capacity(0),
    ));
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

  pub fn union(&self, patterns: &Patterns) -> Patterns {
    Patterns {
      dfa: self.dfa.product(&patterns.dfa),
      min_size: cmp::min(self.min_size, patterns.min_size),
    }
  }

  pub fn from_str(name: &str, string: &str) -> Patterns {
    let mut split = string.split('\n').map(str::trim).filter(|line| !line.is_empty());
    if let Some(header_str) = split.next() {
      // Read header from input string.
      let (width, height, priority) = Patterns::parse_header(name, header_str);
      // Read pattern from input string.
      let mut pattern_str = String::new();
      for _ in 0..height {
        let s = split
          .next()
          .unwrap_or_else(|| panic!("Unexpected end of pattern '{}'.", name));
        assert_eq!(s.len() as u32, width);
        pattern_str.push_str(s);
      }
      // Get moves set from pattern.
      let pattern_moves = Patterns::get_pattern_moves(name, width, &pattern_str);
      // Read and verify moves from input string.
      let mut moves = Vec::with_capacity(pattern_moves.len());
      let mut moves_set = HashSet::new();
      let mut priorities_sum = 0f64;
      for s in split {
        let m = Patterns::parse_move(name, s);
        priorities_sum += m.p;
        moves_set.insert((m.x as u32, m.y as u32));
        moves.push(m);
      }
      if moves_set != pattern_moves {
        panic!("Moves list does not match moves in the pattern named '{}'.", name);
      }
      for m in &mut moves {
        m.p = m.p * priority / priorities_sum;
      }
      // Build DFA for each rotation.
      let mut dfa = Dfa::empty();
      for rotation in 0..8 {
        let cur_dfa = Patterns::build_dfa(name, width, height, &moves, rotation, &pattern_str);
        dfa = dfa.product(&cur_dfa);
      }
      Patterns {
        dfa,
        min_size: cmp::min(width, height),
      }
    } else {
      panic!("Empty pattern '{}'.", name)
    }
  }

  fn from_strings(strings: &[(String, String)]) -> Patterns {
    let len = strings.len();
    if let [(ref name, ref pattern_str)] = *strings {
      Patterns::from_str(name, pattern_str)
    } else {
      let split_idx = len / 2;
      let left = &strings[0..split_idx];
      let right = &strings[split_idx..len];
      let (left_patterns, right_patterns) =
        rayon::join(|| Patterns::from_strings(left), || Patterns::from_strings(right));
      left_patterns.union(&right_patterns)
    }
  }

  pub fn from_tar(file: File) -> Patterns {
    let mut archive = Archive::new(file);
    let mut strings = Vec::new();
    let iter = archive
      .entries()
      .expect("Reading of tar archive is failed.")
      .map(|file| file.expect("Reading of file in tar archive is failed."));
    for mut file in iter.filter(|file| file.header().entry_type().is_file()) {
      let name = file
        .header()
        .path()
        .ok()
        .and_then(|path| path.to_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "<unknown>".to_owned());
      info!(target: PATTERNS_STR, "Loading pattern '{}'", name);
      let mut s = String::new();
      file.read_to_string(&mut s).ok();
      strings.push((name, s));
    }
    let patterns = Patterns::from_strings(&strings);
    info!(target: PATTERNS_STR, "DFA total size: {}.", patterns.dfa.states_count());
    patterns
  }

  pub fn find(&self, field: &Field, player: Player, first_match: bool) -> Vec<(Pos, f64)> {
    if self.dfa.is_empty() || field.width() < self.min_size - 2 || field.height() < self.min_size - 2 {
      return Vec::with_capacity(0);
    }
    let mut priorities_sum = 0f64;
    let mut matched = Vec::new();
    let left_border = (self.min_size as i32 - 1) / 2 - 1;
    let right_border = self.min_size as i32 / 2 - 1;
    let inv_color = player == Player::Black;
    for y in left_border..field.height() as i32 - right_border {
      for x in left_border..field.width() as i32 - right_border {
        let moves = self.dfa.run(
          &mut Spiral::new().map(|(shift_x, shift_y)| {
            let cur_x = x + shift_x;
            let cur_y = y + shift_y;
            if cur_x >= 0 && cur_x < field.width() as i32 && cur_y >= 0 && cur_y < field.height() as i32 {
              let pos = field.to_pos(cur_x as u32, cur_y as u32);
              field.cell(pos)
            } else {
              Cell::new(true)
            }
          }),
          inv_color,
          first_match,
        );
        for m in moves {
          let move_x = (x as i32 + m.x) as u32;
          let move_y = (y as i32 + m.y) as u32;
          priorities_sum += m.p;
          matched.push((field.to_pos(move_x, move_y), m.p));
        }
      }
    }
    for &mut (_, ref mut p) in &mut matched {
      *p /= priorities_sum;
    }
    info!(
      target: PATTERNS_STR,
      "Found moves: {:?}.",
      matched
        .iter()
        .map(|&(pos, p)| (field.to_x(pos), field.to_y(pos), p))
        .collect::<Vec<(u32, u32, f64)>>()
    );
    matched
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
    info!(
      target: PATTERNS_STR,
      "Found sorted moves: {:?}.",
      result
        .iter()
        .map(|&(pos, p)| (field.to_x(pos), field.to_y(pos), p))
        .collect::<Vec<(u32, u32, f64)>>()
    );
    result
  }

  pub fn find_foreground(&self, field: &Field, player: Player, first_match: bool) -> Option<Pos> {
    self
      .find_sorted(field, player, first_match)
      .first()
      .map(|&(pos, _)| pos)
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
