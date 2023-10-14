use oppai_bot::field::{self, Field, Pos};
use oppai_bot::player::Player;
use oppai_bot::zobrist::Zobrist;
use oppai_initial::initial::InitialPosition;
use rand::Rng;
use std::sync::Arc;

#[derive(Debug)]
pub struct ExtendedField {
  pub player: Player,
  pub field: Field,
  pub captures: Vec<(Vec<Pos>, Player, usize)>,
  pub captured: Vec<usize>,
}

impl ExtendedField {
  pub fn new<R: Rng>(width: u32, height: u32, rng: &mut R) -> Self {
    let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, rng));
    let field = Field::new(width, height, zobrist);
    let length = field.length();
    Self {
      player: Player::Red,
      field,
      captures: Vec::new(),
      captured: vec![0; length],
    }
  }

  fn put_points<I>(&mut self, points: I) -> bool
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

  pub fn from_moves<R, I>(width: u32, height: u32, rng: &mut R, moves: I) -> Option<Self>
  where
    R: Rng,
    I: IntoIterator<Item = (Pos, Player)>,
  {
    let mut result = Self::new(width, height, rng);
    if result.put_points(moves) {
      if let Some(&pos) = result.field.points_seq().last() {
        result.player = result.field.cell(pos).get_player().next();
      }
      Some(result)
    } else {
      None
    }
  }

  pub fn place_initial_position(&mut self, initial_position: InitialPosition) {
    self.put_points(initial_position.points(self.field.width(), self.field.height(), self.player));
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

      true
    } else {
      false
    }
  }

  pub fn put_point(&mut self, pos: Pos) -> bool {
    if self.put_players_point(pos, self.player) {
      self.player = self.player.next();
      true
    } else {
      false
    }
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
        .map_or(false, |&(_, _, c)| c > self.field.moves_count())
      {
        self.captures.pop();
      }

      true
    } else {
      false
    }
  }
}
