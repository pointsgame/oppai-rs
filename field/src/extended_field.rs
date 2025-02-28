use rand::Rng;

use crate::{
  field::{Field, Pos},
  player::Player,
  zobrist::Zobrist,
};
use std::sync::Arc;

/// Field that stores additional information useful for rendering purposes.
#[derive(Clone, PartialEq)]
pub struct ExtendedField {
  /// The player for the next turn.
  pub player: Player,
  /// The game field.
  pub field: Field,
  /// A list of surrounding chains with the turn number when the capturing took place.
  pub captures: Vec<(Vec<Pos>, Player, usize)>,
  /// Contains the turn number when a cell was captured.
  pub captured: Vec<usize>,
}

impl From<Field> for ExtendedField {
  fn from(mut field: Field) -> Self {
    let points = field
      .moves()
      .iter()
      .map(|&pos| (pos, field.cell(pos).get_player()))
      .collect::<Vec<_>>();
    let captured = vec![0; field.length()];
    field.undo_all();
    let mut result = ExtendedField {
      player: Player::Red,
      field,
      captures: Vec::new(),
      captured,
    };
    result.put_points(points);
    result
  }
}

impl ExtendedField {
  pub fn new(width: u32, height: u32, zobrist: Arc<Zobrist>) -> Self {
    let field = Field::new(width, height, zobrist);
    let length = field.length();
    Self {
      player: Player::Red,
      field,
      captures: Vec::new(),
      captured: vec![0; length],
    }
  }

  pub fn new_from_rng<R: Rng>(width: u32, height: u32, rng: &mut R) -> Self {
    let field = Field::new_from_rng(width, height, rng);
    let length = field.length();
    Self {
      player: Player::Red,
      field,
      captures: Vec::new(),
      captured: vec![0; length],
    }
  }

  pub fn put_points<I>(&mut self, points: I) -> bool
  where
    I: IntoIterator<Item = (Pos, Player)>,
  {
    for (pos, player) in points {
      if !self.put_players_point(pos, player) {
        return false;
      }
    }
    true
  }

  pub fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.field.put_point(pos, player) {
      let last_chain = self.field.get_last_chain();
      if let Some(&pos) = last_chain.first() {
        let player = self.field.cell(pos).get_player();
        self.captures.push((last_chain, player, self.field.moves_count()));
        for (pos, _) in self.field.last_changed_cells() {
          if self.captured[pos] == 0 && self.field.cell(pos).is_captured() {
            self.captured[pos] = self.field.moves_count();
          }
        }
      }

      let n = self.field.n(pos);
      let s = self.field.s(pos);
      let w = self.field.w(pos);
      let e = self.field.e(pos);
      let nw = self.field.nw(pos);
      let ne = self.field.ne(pos);
      let sw = self.field.sw(pos);
      let se = self.field.se(pos);

      let mut check = |pos1: Pos, pos2: Pos| {
        if self.field.cell(pos1).get_players_point() == Some(player)
          && self.field.cell(pos2).get_players_point() == Some(player)
        {
          self
            .captures
            .push((vec![pos, pos1, pos2], player, self.field.moves_count()));
          true
        } else {
          false
        }
      };

      let _ = !check(s, e) && (check(s, se) || check(e, se));
      let _ = !check(e, n) && (check(e, ne) || check(n, ne));
      let _ = !check(n, w) && (check(n, nw) || check(w, nw));
      let _ = !check(w, s) && (check(w, sw) || check(s, sw));

      self.player = player.next();

      true
    } else {
      false
    }
  }

  pub fn put_point(&mut self, pos: Pos) -> bool {
    self.put_players_point(pos, self.player)
  }

  pub fn undo(&mut self) -> bool {
    if let Some(player) = self.field.last_player() {
      let moves_count = self.field.moves_count();
      for (pos, _) in self.field.last_changed_cells() {
        if self.captured[pos] == moves_count {
          self.captured[pos] = 0;
        }
      }

      self.field.undo();
      self.player = player;

      while self
        .captures
        .last()
        .is_some_and(|&(_, _, c)| c > self.field.moves_count())
      {
        self.captures.pop();
      }

      true
    } else {
      false
    }
  }

  pub fn clear(&mut self) {
    while self.undo() {}
  }
}
