#[derive(Clone, Copy, PartialEq)]
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
