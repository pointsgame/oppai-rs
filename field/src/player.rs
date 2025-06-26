use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum Player {
  #[default]
  Red,
  Black,
}

impl Player {
  #[inline]
  pub fn next(self) -> Player {
    match self {
      Player::Red => Player::Black,
      Player::Black => Player::Red,
    }
  }

  #[inline]
  pub fn from_bool(b: bool) -> Player {
    if b { Player::Black } else { Player::Red }
  }

  #[inline]
  pub fn to_bool(self) -> bool {
    self == Player::Black
  }
}

impl Display for Player {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match *self {
      Player::Red => write!(f, "Red"),
      Player::Black => write!(f, "Black"),
    }
  }
}
