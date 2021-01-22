use crate::extended_field::ExtendedField;
use oppai_field::player::Player;
use rand::Rng;
use sgf_parser::{Action, Color, Game, GameNode, GameTree, SgfToken};

struct Root {
  width: u32,
  height: u32,
  moves: Vec<(Player, u32, u32)>,
}

fn color_to_player(color: Color) -> Player {
  match color {
    Color::White => Player::Red,
    Color::Black => Player::Black,
  }
}

fn parse_root(root: &GameNode) -> Option<Root> {
  let mut game = false;
  let mut size = None;
  let mut moves = Vec::new();
  for token in &root.tokens {
    match token {
      SgfToken::Game(g) => game = g == &Game::Other(40),
      SgfToken::Size(w, h) => size = Some((*w, *h)),
      SgfToken::Add {
        color,
        coordinate: (x, y),
      } => moves.push((color_to_player(*color), (x - 1) as u32, (y - 1) as u32)),
      _ => {}
    }
  }
  if game {
    size.map(|(width, height)| Root { width, height, moves })
  } else {
    None
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
          let player = color_to_player(*color);
          return Some((player, (x - 1) as u32, (y - 1) as u32));
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
        let x = to_coordinate(value.as_bytes()[0]);
        let y = to_coordinate(value.as_bytes()[1]);
        return Some((player, x as u32, y as u32));
      }
      _ => {}
    }
  }
  None
}

pub fn from_sgf<G: Rng>(game_tree: GameTree, rng: &mut G) -> Option<ExtendedField> {
  let root = if let Some(root) = game_tree.nodes.first().and_then(parse_root) {
    root
  } else {
    return None;
  };

  let mut extended_field = ExtendedField::new(root.width, root.height, rng);

  for (player, x, y) in root.moves {
    let pos = extended_field.field.to_pos(x, y);
    if !extended_field.put_players_point(pos, player) {
      return None;
    }
  }

  for node in &game_tree.nodes[1..] {
    if let Some((player, x, y)) = parse_node(node) {
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
