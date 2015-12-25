use std::io::{BufReader, BufRead};
use std::str::FromStr;
use std::fs::File;
use std::cmp;
use tar::Archive;
use spiral::Spiral;
use dfa::{Dfa, DfaState};
use field::Field;

struct Move {
  x: u32,
  y: u32,
  p: f64 // probability
}

struct Pattern {
  p: f64, // priority (probability = p / sum(p))
  moves: Vec<Move>
}

pub struct Patterns {
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
    let x = side_of_square / 2;
    (8 * x - 13) * x + 6
  }

  fn build_dfa(width: u32, height: u32, pattern: u32, s: &str) -> Dfa {
    let center_x = width / 2;
    let center_y = height / 2;
    let spiral_length = Patterns::covering_spiral_length(cmp::max(width, height)) as usize;
    let mut states = Vec::with_capacity(spiral_length + 1);
    let mut i = 0;
    for (shift_x, shift_y) in Spiral::new().into_iter().take(spiral_length) {
      i += 1;
      let x = center_x as i32 + shift_x;
      let y = center_y as i32 + shift_y;
      let state = if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
        let pos = y as u32 * width + x as u32;
        match s.char_at(pos as usize) {
          '.' => DfaState::new(i, -1, -1, -1, -1),
          '?' => DfaState::new(i, i, i, i, -1),
          'R' => DfaState::new(-1, i, -1, -1, -1),
          'B' => DfaState::new(-1, -1, i, -1, -1),
          'r' => DfaState::new(i, i, -1, -1, -1),
          'b' => DfaState::new(i, -1, i, -1, -1),
          '*' => DfaState::new(-1, -1, -1, i, -1),
          c   => panic!("Invalid character in pattern: {}", c)
        }
      } else {
        DfaState::new(-1, -1, -1, -1, -1)
      };
      states.push(state);
    }
    states.push(DfaState::new(-1, -1, -1, -1, pattern as i32));
    Dfa::new(states)
  }

  pub fn load(file: File) -> Patterns {
    let archive = Archive::new(file);
    let mut s = String::new();
    let mut patterns = Vec::new();
    let mut iter = archive.files().expect("Reading of tar archive is failed.").into_iter().map(|file| file.expect("Reading of file in tar archive is failed."));
    let mut dfa = {
      let file = iter.next().expect("Archive should contain at least one pattern.");
      let mut input = BufReader::new(file);
      let (width, height, moves_count, priority) = Patterns::read_header(&mut input, &mut s);
      Patterns::read_pattern(&mut input, &mut s, width, height);
      let cur_dfa = Patterns::build_dfa(width, height, 0, &s);
      let moves = Patterns::read_moves(&mut input, &mut s, moves_count);
      patterns.push(Pattern {
        p: priority,
        moves: moves
      });
      cur_dfa
    };
    for file in iter {
      let mut input = BufReader::new(file);
      let (width, height, moves_count, priority) = Patterns::read_header(&mut input, &mut s);
      Patterns::read_pattern(&mut input, &mut s, width, height);
      let cur_dfa = Patterns::build_dfa(width, height, patterns.len() as u32, &s);
      dfa = dfa.product(&cur_dfa);
      dfa.delete_non_reachable();
      let moves = Patterns::read_moves(&mut input, &mut s, moves_count);
      patterns.push(Pattern {
        p: priority,
        moves: moves
      });
    }
    dfa.minimize();
    Patterns {
      dfa: dfa,
      patterns: patterns
    }
  }

  pub fn find(&self, field: &Field) -> Vec<(i32, i32, f64)> {
    unimplemented!()
  }
}
