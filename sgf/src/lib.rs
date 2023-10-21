#[cfg(test)]
mod test;

use oppai_field::{any_field::AnyField, field::Field, player::Player};
use rand::Rng;
use sgf_parse::{serialize, unknown_game::Prop, GameTree, SgfNode};
use std::iter;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Move {
  Pass,
  Move(u8, u8, Vec<(u8, u8)>),
}

fn to_coordinate(c: u8) -> u8 {
  if c > 96 {
    c - 97
  } else {
    c - 39
  }
}

fn from_coordinate(c: u8) -> u8 {
  if c > 26 {
    c + 39
  } else {
    c + 97
  }
}

fn parse_move(s: &str) -> Option<Move> {
  if s.is_empty() {
    return Some(Move::Pass);
  } else if s.len() == 1 {
    return None;
  }
  let x = to_coordinate(s.as_bytes()[0]);
  let y = to_coordinate(s.as_bytes()[1]);
  let mut chain = Vec::new();
  if s.len() > 2 {
    if s.as_bytes()[2] != b'.' || s.len() % 2 == 0 {
      return None;
    }

    let mut i = 3;
    while i < s.len() {
      let x = to_coordinate(s.as_bytes()[i]);
      let y = to_coordinate(s.as_bytes()[i + 1]);
      chain.push((x, y));
      i += 2;
    }
  }
  Some(Move::Move(x, y, chain))
}

pub fn from_sgf<F: AnyField, R: Rng>(sgf: &str, rng: &mut R) -> Option<F> {
  let trees = sgf_parse::parse(sgf).ok()?;
  let node = trees.iter().find_map(|tree| match tree {
    GameTree::Unknown(node) => Some(node),
    GameTree::GoGame(_) => None,
  })?;

  if node.get_property("GM")? != &Prop::GM(40) {
    return None;
  };

  let (width, height) = if let Prop::SZ(size) = node.get_property("SZ")? {
    *size
  } else {
    return None;
  };

  let mut field = <F as AnyField>::new_from_rng(width as u32, height as u32, rng);

  let mut handle = |player: Player, s: &str| -> bool {
    if let Some(Move::Move(x, y, chain)) = parse_move(s) {
      let pos = field.field().to_pos(x as u32, y as u32);
      let result = field.put_players_point(pos, player);
      if !chain.into_iter().all(|(x, y)| {
        let pos = field.field().to_pos(x as u32, y as u32);
        field.field().cell(pos).is_bound_player(player)
      }) {
        log::warn!("Surrounding chain doesn't match the game rules, the position might be inaccurate.");
      }
      result
    } else {
      false
    }
  };

  'outer: for node in node.main_variation() {
    for prop in node.properties() {
      match prop {
        Prop::B(s) => {
          if !handle(Player::Black, s) {
            break 'outer;
          }
        }
        Prop::W(s) => {
          if !handle(Player::Red, s) {
            break 'outer;
          }
        }
        Prop::AB(set) => {
          for s in set {
            if !handle(Player::Black, s) {
              break 'outer;
            }
          }
        }
        Prop::AW(set) => {
          for s in set {
            if !handle(Player::Red, s) {
              break 'outer;
            }
          }
        }
        _ => {}
      };
    }
  }

  Some(field)
}

pub fn to_sgf(field: &Field) -> Option<String> {
  if field.width() > 52 || field.height() > 52 {
    return None;
  }

  let mut node = SgfNode::new(Vec::new(), Vec::new(), false);
  for &pos in field.points_seq().iter().rev() {
    let x = field.to_x(pos) as u8;
    let y = field.to_y(pos) as u8;
    let m = format!("{}{}", from_coordinate(x) as char, from_coordinate(y) as char);
    match field.cell(pos).get_player() {
      Player::Red => node.properties.push(Prop::W(m)),
      Player::Black => node.properties.push(Prop::B(m)),
    }
    node = SgfNode::new(Vec::new(), vec![node], false);
  }
  node.properties.push(Prop::GM(40));
  node
    .properties
    .push(Prop::SZ((field.width() as u8, field.height() as u8)));
  node.is_root = true;
  let tree = GameTree::Unknown(node);

  Some(serialize(iter::once(&tree)))
}
