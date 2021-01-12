use crate::dfa::{Dfa, DfaState};
use crate::rotate::*;
use crate::spiral::Spiral;
use oppai_field::cell::Cell;
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use std::{
  cmp,
  fs::File,
  io::{BufRead, BufReader},
};

#[derive(Clone, Debug)]
struct Move {
  x: i32,
  y: i32,
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

  fn covering_spiral_length(side_of_square: u32) -> u32 {
    let x = side_of_square / 2 + 1;
    let y = (1 - side_of_square % 2) * side_of_square * 2;
    (8 * x - 13) * x + 6 - y
  }

  fn build_dfa(
    width: u32,
    height: u32,
    moves: &[Move],
    rotation: u32,
    chars: &[char],
  ) -> Result<Dfa<Move>, &'static str> {
    let (rotated_width, rotated_height) = rotate_sizes(width, height, rotation);
    let center_x = (rotated_width - 1) / 2;
    let center_y = (rotated_height - 1) / 2;
    let spiral_length = Patterns::covering_spiral_length(cmp::max(width, height)) as usize;
    let mut states = Vec::with_capacity(spiral_length + 2);
    let fs = spiral_length; // "Found" state.
    let nfs = spiral_length + 1; // "Not found" state.
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
          match chars[pos as usize] {
            '.' | '+' => DfaState::new(nxt, nfs, nfs, nfs, false, Vec::new()),
            '?' => DfaState::new(nxt, nxt, nxt, nfs, false, Vec::new()),
            '*' => DfaState::new(nxt, nxt, nxt, nxt, false, Vec::new()),
            'X' => DfaState::new(nfs, nxt, nfs, nfs, false, Vec::new()),
            'O' => DfaState::new(nfs, nfs, nxt, nfs, false, Vec::new()),
            'x' => DfaState::new(nxt, nxt, nfs, nfs, false, Vec::new()),
            'o' => DfaState::new(nxt, nfs, nxt, nfs, false, Vec::new()),
            '#' => DfaState::new(nfs, nfs, nfs, nxt, false, Vec::new()),
            _ => return Err("Invalid character in the pattern."),
          }
        } else {
          DfaState::new(nxt, nxt, nxt, nxt, false, Vec::new())
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
        }
      })
      .collect();
    states.push(DfaState::new(new_fs, new_fs, new_fs, new_fs, true, rotated_moves));
    states.push(DfaState::new(new_nfs, new_nfs, new_nfs, new_nfs, true, Vec::new()));
    Ok(Dfa::new(states))
  }

  fn get_pattern_moves(width: u32, chars: &[char]) -> Result<Vec<Move>, &'static str> {
    let moves = chars
      .iter()
      .enumerate()
      .filter(|&(_, &c)| c == '+')
      .map(|(i, _)| Move {
        x: (i as u32 % width) as i32,
        y: (i as u32 / width) as i32,
      })
      .collect::<Vec<_>>();

    if moves.is_empty() {
      Err("Moves are not defined.")
    } else {
      Ok(moves)
    }
  }

  pub fn union(&self, patterns: &Patterns) -> Patterns {
    Patterns {
      dfa: self.dfa.product(&patterns.dfa),
      min_size: cmp::min(self.min_size, patterns.min_size),
    }
  }

  pub fn from_str(string: &str) -> Result<Patterns, &'static str> {
    let mut split = string
      .split('\n')
      .map(str::trim)
      .filter(|line| !line.is_empty())
      .peekable();
    let width = if let Some(first) = split.peek() {
      first.len() as u32
    } else {
      return Err("Empty pattern.");
    };
    let height = split.count() as u32;
    let chars = string.chars().filter(|c| !c.is_whitespace()).collect::<Vec<_>>();

    let moves = Patterns::get_pattern_moves(width, &chars)?;

    let mut dfa = Dfa::empty();
    for rotation in 0..8 {
      let cur_dfa = Patterns::build_dfa(width, height, &moves, rotation, &chars)?;
      dfa = dfa.product(&cur_dfa);
    }
    Ok(Patterns {
      dfa,
      min_size: cmp::min(width, height),
    })
  }

  fn from_strings(strings: &[String]) -> Patterns {
    let len = strings.len();
    if let [ref pattern_str] = *strings {
      match Patterns::from_str(pattern_str) {
        Ok(patterns) => patterns,
        Err(e) => {
          error!("Failed to parse pattern: {}\n{}", e, pattern_str);
          Patterns::empty()
        }
      }
    } else {
      let split_idx = len / 2;
      let left = &strings[0..split_idx];
      let right = &strings[split_idx..len];
      let (left_patterns, right_patterns) =
        rayon::join(|| Patterns::from_strings(left), || Patterns::from_strings(right));
      left_patterns.union(&right_patterns)
    }
  }

  pub fn from_files<T: Iterator<Item = File>>(files: T) -> std::io::Result<Patterns> {
    let mut strings = Vec::new();
    let mut string = String::new();

    for file in files {
      let mut reader = BufReader::new(file);
      loop {
        let len = reader.read_line(&mut string)?;
        if len <= 1 {
          if string.len() > 1 {
            strings.push(string);
            string = String::new();
          } else {
            string.clear();
          }
        }
        if len == 0 {
          break;
        }
      }
    }

    let patterns = Patterns::from_strings(&strings);
    info!("DFA total size: {}.", patterns.dfa.states_count());
    Ok(patterns)
  }

  pub fn find(&self, field: &Field, player: Player, first_match: bool) -> Vec<Pos> {
    if self.dfa.is_empty() || field.width() < self.min_size - 2 || field.height() < self.min_size - 2 {
      return Vec::new();
    }
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
          matched.push(field.to_pos(move_x, move_y));
        }
        if first_match && !matched.is_empty() {
          info!(
            "Found first matched moves: {:?}.",
            matched
              .iter()
              .map(|&pos| (field.to_x(pos), field.to_y(pos)))
              .collect::<Vec<(u32, u32)>>()
          );
          return matched;
        }
      }
    }
    info!(
      "Found moves: {:?}.",
      matched
        .iter()
        .map(|&pos| (field.to_x(pos), field.to_y(pos)))
        .collect::<Vec<(u32, u32)>>()
    );
    matched
  }
}
