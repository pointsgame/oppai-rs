use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Player {
  Red,
  Black
}

impl Player {
  pub fn next(self) -> Player {
    match self {
      Player::Red => Player::Black,
      Player::Black => Player::Red
    }
  }

  pub fn from_bool(b: bool) -> Player {
    if b { Player::Black } else { Player::Red }
  }

  pub fn to_bool(self) -> bool {
    self == Player::Black
  }
}

impl Display for Player {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match self {
      &Player::Red => write!(f, "Red"),
      &Player::Black => write!(f, "Black")
    }
  }
}
