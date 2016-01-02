use std::collections::{HashSet, HashMap};
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use std::fs::File;
use std::cmp;
use tar::Archive;
use spiral::Spiral;
use dfa::{Dfa, DfaState};
use player::Player;
use cell::Cell;
use field::Field;

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
  dfa: Dfa,
  patterns: Vec<Pattern>
}

impl Patterns {
  fn read_header<T: BufRead>(input: &mut T, s: &mut String) -> (u32, u32, u32, f64) {
    s.clear();
    input.read_line(s).ok();
    s.pop();
    let mut split = s.split(' ').fuse();
    let width = u32::from_str(split.next().expect("Invalid pattern format: expected width.")).expect("Invalid pattern format: width must be u32."); //TODO: validate all this.
    let height = u32::from_str(split.next().expect("Invalid pattern format: expected height.")).expect("Invalid pattern format: height must be u32.");
    let moves_count = u32::from_str(split.next().expect("Invalid pattern format: expected moves count.")).expect("Invalid pattern format: moves count must be u32.");
    let priority = f64::from_str(split.next().expect("Invalid pattern format: expected priority.")).expect("Invalid pattern format: priority must be f64.");
    (width, height, moves_count, priority)
  }

  fn read_pattern<T: BufRead>(input: &mut T, s: &mut String, width: u32, height: u32) {
    s.clear();
    for y in 0 .. height {
      input.read_line(s).ok(); //TODO: check sizes.
      s.pop();
    }
  }

  fn read_moves<T: BufRead>(input: &mut T, s: &mut String, moves_count: u32) -> Vec<Move> {
    let mut moves = Vec::with_capacity(moves_count as usize);
    for _ in 0 .. moves_count {
      s.clear();
      input.read_line(s).ok();
      s.pop();
      let mut split = s.split(' ').fuse();
      let x = u32::from_str(split.next().expect("Invalid pattern format: expected x coordinate.")).expect("Invalid pattern format: x coordinate must be u32.");
      let y = u32::from_str(split.next().expect("Invalid pattern format: expected x coordinate.")).expect("Invalid pattern format: x coordinate must be u32.");
      let p = f64::from_str(split.next().expect("Invalid pattern format: expected probability.")).expect("Invalid pattern format: probability must be f64.");
      moves.push(Move {
        x: x,
        y: y,
        p: p
      });
    }
    moves
  }

  fn covering_spiral_length(side_of_square: u32) -> u32 {
    let x = side_of_square / 2 + 1;
    let y = (1 - side_of_square % 2) * side_of_square * 2;
    (8 * x - 13) * x + 6 - y
  }

  fn rotate(width: u32, height: u32, x: i32, y: i32, rotation: u32) -> (i32, i32) {
    match rotation {
      0 => (x, y),
      1 => (width as i32 - x - 1, y),
      2 => (x, height as i32 - y - 1),
      3 => (width as i32 - x - 1, height as i32 - y - 1),
      4 => (y, x),
      5 => (height as i32 - y - 1, x),
      6 => (y, width as i32 - x - 1),
      7 => (height as i32 - y - 1, width as i32 - x - 1),
      r => panic!("Invalid rotation number: {}", r)
    }
  }

  fn build_dfa(width: u32, height: u32, pattern: usize, rotation: u32, s: &str) -> Dfa {
    let center_x = (width - 1) / 2;
    let center_y = (height - 1) / 2;
    let spiral_length = Patterns::covering_spiral_length(cmp::max(width, height)) as usize;
    let mut states = Vec::with_capacity(spiral_length + 2);
    let fs = spiral_length; // "Found" state.
    let nfs = spiral_length + 1; // "Not found" state.
    let mut i = 0;
    for (shift_x, shift_y) in Spiral::new().into_iter().take(spiral_length) {
      i += 1;
      let (x, y) = Patterns::rotate(width, height, center_x as i32 + shift_x, center_y as i32 + shift_y, rotation);
      let state = if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
        let pos = y as u32 * width + x as u32;
        match s.char_at(pos as usize) {
          '.' => DfaState::new(i, nfs, nfs, nfs, false, HashSet::with_capacity(0)),
          '?' => DfaState::new(i, i, i, nfs, false, HashSet::with_capacity(0)),
          '*' => DfaState::new(i, i, i, i, false, HashSet::with_capacity(0)),
          'R' => DfaState::new(nfs, i, nfs, nfs, false, HashSet::with_capacity(0)),
          'B' => DfaState::new(nfs, nfs, i, nfs, false, HashSet::with_capacity(0)),
          'r' => DfaState::new(i, i, nfs, nfs, false, HashSet::with_capacity(0)),
          'b' => DfaState::new(i, nfs, i, nfs, false, HashSet::with_capacity(0)),
          '#' => DfaState::new(nfs, nfs, nfs, i, false, HashSet::with_capacity(0)),
          c   => panic!("Invalid character in pattern: {}", c)
        }
      } else {
        DfaState::new(i, i, i, i, false, HashSet::with_capacity(0))
      };
      states.push(state);
    }
    let mut patterns = HashSet::with_capacity(1);
    patterns.insert(pattern);
    states.push(DfaState::new(fs, fs, fs, fs, true, patterns));
    states.push(DfaState::new(nfs, nfs, nfs, nfs, true, HashSet::with_capacity(0)));
    Dfa::new(states)
  }

  pub fn load(file: File) -> Patterns {
    let archive = Archive::new(file);
    let mut s = String::new();
    let mut pattern_s = String::new();
    let mut patterns = Vec::new();
    let mut iter = archive.files().expect("Reading of tar archive is failed.").into_iter().map(|file| file.expect("Reading of file in tar archive is failed."));
    let mut dfa = Dfa::empty();
    let mut min_size = u32::max_value();
    for file in iter {
      let mut input = BufReader::new(file);
      let (width, height, moves_count, priority) = Patterns::read_header(&mut input, &mut s);
      if width < min_size {
        min_size = width;
      }
      if height < min_size {
        min_size = height;
      }
      Patterns::read_pattern(&mut input, &mut pattern_s, width, height);
      let moves = Patterns::read_moves(&mut input, &mut s, moves_count);
      for i in 0 .. 8 {
        let cur_dfa = Patterns::build_dfa(width, height, patterns.len(), i, &pattern_s);
        dfa = dfa.product(&cur_dfa);
        patterns.push(Pattern {
          p: priority,
          width: width,
          height: height,
          moves: moves.iter().map(|m| {
            let (x, y) = Patterns::rotate(width, height, m.x as i32, m.y as i32, i);
            Move {
              x: x as u32,
              y: y as u32,
              p: m.p
            }
          }).collect()
        });
      }
    }
    dfa.minimize(); //TODO: does it work?
    Patterns {
      min_size: min_size,
      dfa: dfa,
      patterns: patterns
    }
  }

  pub fn find(&self, field: &Field, player: Player, first_match: bool) -> Vec<Move> {
    if self.dfa.is_empty() {
      return Vec::with_capacity(0);
    }
    let mut priorities_sum = 0f64;
    let mut moves_count = 0usize;
    let mut matched = Vec::new();
    let left_border = (self.min_size - 1) / 2;
    let right_border = self.min_size / 2;
    let inv_color = player == Player::Black;
    for y in left_border .. field.height() - right_border {
      for x in left_border .. field.width() - right_border {
        let patterns = self.dfa.run(&mut Spiral::new().into_iter().map(|(shift_x, shift_y)| {
          let cur_x = x as i32 + shift_x;
          let cur_y = y as i32 + shift_y;
          if cur_x >= 0 && cur_x < field.width() as i32 && cur_y >= 0 && cur_y < field.height() as i32 {
            let pos = field.to_pos(cur_x as u32, cur_y as u32);
            field.cell(pos)
          } else {
            Cell::new(true)
          }
        }), inv_color, first_match);
        for &pattern_number in patterns {
          let pattern = &self.patterns[pattern_number];
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
        let move_x = center_x - (pattern.width - 1) / 2 + x;
        let move_y = center_y - (pattern.height - 1) / 2 + y;
        result.push(Move { x: move_x, y: move_y, p: probability * pattern.p / priorities_sum });
      }
    }
    result
  }

  pub fn find_sorted(&self, field: &Field, player: Player, first_match: bool) -> Vec<Move> {
    let moves = self.find(field, player, first_match);
    let mut map = HashMap::with_capacity(moves.len());
    for Move { x, y, p } in moves {
      let coord = (x, y);
      let sum_p = map.get(&coord).cloned().unwrap_or(0f64) + p;
      map.insert(coord, sum_p);
    }
    let mut result = map.into_iter().map(|((x, y), p)| Move { x: x, y: y, p: p }).collect::<Vec<Move>>();
    result.sort_by(|a, b| b.p.partial_cmp(&a.p).expect("Cann't compare f64 types."));
    result
  }
}
