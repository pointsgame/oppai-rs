#[cfg(test)]
mod test;

use oppai_field::{any_field::AnyField, extended_field::ExtendedField, player::Player};
use rand::Rng;
use sgf_parse::{serialize, unknown_game::Prop, GameTree, SgfNode};
use std::{fmt::Display, iter};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Move {
  Pass,
  Move(u8, u8, Vec<Vec<(u8, u8)>>),
}

impl Display for Move {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Move::Pass => Ok(()),
      Move::Move(x, y, chains) => {
        write!(f, "{}{}", from_coordinate(*x) as char, from_coordinate(*y) as char)?;
        for chain in chains {
          write!(f, ".")?;
          for (x, y) in chain {
            write!(f, "{}{}", from_coordinate(*x) as char, from_coordinate(*y) as char)?;
          }
        }
        Ok(())
      }
    }
  }
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
  let mut chains = Vec::new();
  if s.len() > 2 {
    for c in s.split('.').skip(1) {
      let mut chain = Vec::new();
      if c.len() % 2 == 1 {
        return None;
      }

      for i in 0..c.len() / 2 {
        let x = to_coordinate(c.as_bytes()[i * 2]);
        let y = to_coordinate(c.as_bytes()[i * 2 + 1]);
        chain.push((x, y));
      }
      chains.push(chain);
    }
  }
  Some(Move::Move(x, y, chains))
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
    if let Some(Move::Move(x, y, chains)) = parse_move(s) {
      let pos = field.field().to_pos(x as u32, y as u32);
      let result = field.put_players_point(pos, player);
      if !chains.into_iter().flat_map(|chain| chain.into_iter()).all(|(x, y)| {
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

pub fn to_sgf(field: &ExtendedField) -> Option<String> {
  if field.field().width() > 52 || field.field().height() > 52 {
    return None;
  }

  let mut node = SgfNode::new(Vec::new(), Vec::new(), false);
  let mut i = field.captures.len();
  for (n, &pos) in field.field().moves().iter().enumerate().rev() {
    let x = field.field().to_x(pos) as u8;
    let y = field.field().to_y(pos) as u8;
    let player = field.field().cell(pos).get_player();
    let mut chains = Vec::new();
    while i > 0 && field.captures[i - 1].2 > n + 1 {
      i -= 1;
    }
    if i > 0
      && field.captures[i - 1].0.len() > 3
      && field.captures[i - 1].1 == player
      && field.captures[i - 1].2 == n + 1
    {
      chains.push(
        field.captures[i - 1]
          .0
          .iter()
          .map(|&pos| (field.field().to_x(pos) as u8, field.field().to_y(pos) as u8))
          .collect(),
      )
    }
    if n > 0 && field.field().cell(field.field().moves()[n - 1]).get_player() == player.next() {
      for j in 0..2 {
        if i > j
          && field.captures[i - j - 1].0.len() > 3
          && field.captures[i - j - 1].1 == player
          && field.captures[i - j - 1].2 == n
        {
          chains.push(
            field.captures[i - j - 1]
              .0
              .iter()
              .map(|&pos| (field.field().to_x(pos) as u8, field.field().to_y(pos) as u8))
              .collect(),
          )
        }
      }
    }
    let m = Move::Move(x, y, chains);
    let m = format!("{}", m);
    match player {
      Player::Red => node.properties.push(Prop::W(m)),
      Player::Black => node.properties.push(Prop::B(m)),
    }
    node = SgfNode::new(Vec::new(), vec![node], false);
  }
  node.properties.push(Prop::GM(40));
  node
    .properties
    .push(Prop::SZ((field.field().width() as u8, field.field().height() as u8)));
  node.properties.push(Prop::RU("russian".into()));
  node.is_root = true;
  let tree = GameTree::Unknown(node);

  Some(serialize(iter::once(&tree)))
}
