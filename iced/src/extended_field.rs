use oppai_field::field::{self, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use rand::Rng;
use sgf_parser::{Action, Color, Game, GameNode, GameTree, SgfToken};
use std::sync::Arc;

#[derive(Debug)]
pub struct ExtendedField {
  pub player: Player,
  pub field: Field,
  pub captures: Vec<(Vec<Pos>, Player, usize)>,
  pub captured: Vec<usize>,
}

impl ExtendedField {
  pub fn new<G: Rng>(width: u32, height: u32, rng: &mut G) -> Self {
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

  fn parse_root(root: &GameNode) -> Option<(u32, u32)> {
    let mut game = None;
    let mut size = None;
    for token in &root.tokens {
      if game.is_some() && size.is_some() {
        break;
      }
      match token {
        SgfToken::Game(g) => game = Some(g == &Game::Other(40)),
        SgfToken::Size(w, h) => size = Some((*w, *h)),
        _ => {}
      }
    }
    if game == Some(true) {
      size
    } else {
      None
    }
  }

  fn color_to_player(color: Color) -> Player {
    match color {
      Color::White => Player::Red,
      Color::Black => Player::Black,
    }
  }

  fn to_coordinate(c: u8) -> u8 {
    if c > 96 {
      c - 97
    } else {
      c - 39
    }
  }

  fn parse_node(node: &GameNode) -> Option<(Player, u32, u32)> {
    for token in &node.tokens {
      match token {
        SgfToken::Move { color, action } => match action {
          Action::Pass => return None,
          Action::Move(x, y) => {
            let player = ExtendedField::color_to_player(*color);
            return Some((player, (*x - 1) as u32, (*y - 1) as u32));
          }
        },
        SgfToken::Invalid((ident, value)) => {
          let player = if ident == "W" {
            Player::Red
          } else if ident == "B" {
            Player::Black
          } else {
            continue;
          };
          if value.len() < 2 {
            return None;
          }
          let x = ExtendedField::to_coordinate(value.as_bytes()[0]);
          let y = ExtendedField::to_coordinate(value.as_bytes()[1]);
          return Some((player, x as u32, y as u32));
        }
        _ => {}
      }
    }
    None
  }

  pub fn from_sgf<G: Rng>(game_tree: GameTree, rng: &mut G) -> Option<ExtendedField> {
    let (width, height) = if let Some(size) = game_tree.nodes.first().and_then(ExtendedField::parse_root) {
      size
    } else {
      return None;
    };

    let mut extended_field = ExtendedField::new(width, height, rng);

    for node in &game_tree.nodes[1..] {
      if let Some((player, x, y)) = ExtendedField::parse_node(node) {
        let pos = extended_field.field.to_pos(x, y);
        if !extended_field.put_players_point(pos, player) {
          return None;
        }
      }
    }

    if let Some(player) = extended_field.field.last_player() {
      extended_field.player = player.next();
    }

    Some(extended_field)
  }
}
