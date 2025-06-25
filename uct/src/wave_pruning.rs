use oppai_field::field::{self, Field, Pos};
use std::iter;

#[derive(Clone)]
pub struct WavePruning {
  moves: Vec<Pos>,
  moves_field: Vec<Pos>,
}

impl WavePruning {
  pub fn new(length: Pos) -> WavePruning {
    WavePruning {
      moves: Vec::with_capacity(length),
      moves_field: iter::repeat_n(0, length).collect(),
    }
  }

  pub fn moves(&self) -> &Vec<Pos> {
    &self.moves
  }

  pub fn clear(&mut self) {
    self.moves.clear();
    for i in &mut self.moves_field {
      *i = 0;
    }
  }

  pub fn init(&mut self, field: &Field, radius: u32) {
    let width = field.width();
    for &start_pos in field.moves() {
      field::wave(width, start_pos, |pos| {
        if pos == start_pos && self.moves_field[pos] == 0 {
          self.moves_field[pos] = 1;
          true
        } else if self.moves_field[pos] != start_pos
          && field.cell(pos).is_putting_allowed()
          && field::manhattan(width, start_pos, pos) <= radius
        {
          if self.moves_field[pos] == 0 {
            self.moves.push(pos);
          }
          self.moves_field[pos] = start_pos;
          true
        } else {
          false
        }
      });
      self.moves_field[start_pos] = 0;
    }
  }

  pub fn update(&mut self, field: &Field, last_moves_count: usize, radius: u32) -> Vec<Pos> {
    let moves_field = &mut self.moves_field;
    let moves = &mut self.moves;
    moves.retain(|&pos| {
      if field.cell(pos).is_putting_allowed() {
        true
      } else {
        moves_field[pos] = 0;
        false
      }
    });
    let width = field.width();
    let mut added_moves = Vec::new();
    for &next_pos in field.moves().iter().skip(last_moves_count) {
      field::wave(width, next_pos, |pos| {
        if pos == next_pos && moves_field[pos] == 0 {
          moves_field[pos] = 1;
          true
        } else if moves_field[pos] != next_pos
          && field.cell(pos).is_putting_allowed()
          && field::manhattan(width, next_pos, pos) <= radius
        {
          if moves_field[pos] == 0 && pos != next_pos {
            moves.push(pos);
            added_moves.push(pos);
          }
          moves_field[pos] = next_pos;
          true
        } else {
          false
        }
      });
      moves_field[next_pos] = 0;
    }
    added_moves
  }
}
