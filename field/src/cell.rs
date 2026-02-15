use crate::player::Player;

type CellValue = u8;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell(pub CellValue);

impl Cell {
  pub const PLAYER_BIT: CellValue = 1;

  pub const PUT_BIT: CellValue = 2;

  pub const CAPTURED_BIT: CellValue = 4;

  pub const BOUND_BIT: CellValue = 8;

  pub const EMPTY_BASE_BIT: CellValue = 16;

  pub const BAD_BIT: CellValue = 32;

  pub const TAG_BIT: CellValue = 64;

  #[inline]
  pub fn new(bad: bool) -> Cell {
    Cell(if bad { Self::BAD_BIT } else { 0 })
  }

  #[inline]
  pub fn get_player(self) -> Player {
    Player::from_bool(self.0 & Self::PLAYER_BIT != 0)
  }

  #[inline]
  pub fn set_player(&mut self, player: Player) {
    self.0 = self.0 & !Self::PLAYER_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn is_put(self) -> bool {
    self.0 & Self::PUT_BIT != 0
  }

  #[inline]
  pub fn set_put(&mut self) {
    self.0 |= Self::PUT_BIT
  }

  #[inline]
  pub fn clear_put(&mut self) {
    self.0 &= !Self::PUT_BIT
  }

  #[inline]
  pub fn is_captured(self) -> bool {
    self.0 & Self::CAPTURED_BIT != 0
  }

  #[inline]
  pub fn set_captured(&mut self) {
    self.0 |= Self::CAPTURED_BIT
  }

  #[inline]
  pub fn clear_captured(&mut self) {
    self.0 &= !Self::CAPTURED_BIT
  }

  #[inline]
  pub fn is_bound(self) -> bool {
    self.0 & Self::BOUND_BIT != 0
  }

  #[inline]
  pub fn set_bound(&mut self) {
    self.0 |= Self::BOUND_BIT
  }

  #[inline]
  pub fn clear_bound(&mut self) {
    self.0 &= !Self::BOUND_BIT
  }

  #[inline]
  pub fn is_empty_base(self) -> bool {
    self.0 & Self::EMPTY_BASE_BIT != 0
  }

  #[inline]
  pub fn set_empty_base(&mut self) {
    self.0 |= Self::EMPTY_BASE_BIT
  }

  #[inline]
  pub fn clear_empty_base(&mut self) {
    self.0 &= !Self::EMPTY_BASE_BIT
  }

  #[inline]
  pub fn is_bad(self) -> bool {
    self.0 & Self::BAD_BIT != 0
  }

  #[inline]
  pub fn set_bad(&mut self) {
    self.0 |= Self::BAD_BIT
  }

  #[inline]
  pub fn clear_bad(&mut self) {
    self.0 &= !Self::BAD_BIT
  }

  #[inline]
  pub fn is_tagged(self) -> bool {
    self.0 & Self::TAG_BIT != 0
  }

  #[inline]
  pub fn set_tag(&mut self) {
    self.0 |= Self::TAG_BIT
  }

  #[inline]
  pub fn clear_tag(&mut self) {
    self.0 &= !Self::TAG_BIT
  }

  #[inline]
  pub fn get_owner(self) -> Option<Player> {
    if self.is_captured() {
      if self.is_put() {
        Some(self.get_player().next())
      } else {
        Some(self.get_player())
      }
    } else if self.is_put() {
      Some(self.get_player())
    } else {
      None
    }
  }

  #[inline]
  pub fn is_owner(self, player: Player) -> bool {
    self.get_owner() == Some(player)
  }

  #[inline]
  pub fn get_players_point(self) -> Option<Player> {
    if self.is_put() { Some(self.get_player()) } else { None }
  }

  #[inline]
  pub fn is_players_point(self, player: Player) -> bool {
    self.0 & (Self::PUT_BIT | Self::PLAYER_BIT) == Self::PUT_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn get_live_players_point(self) -> Option<Player> {
    if self.is_put() && !self.is_captured() {
      Some(self.get_player())
    } else {
      None
    }
  }

  #[inline]
  pub fn is_live_players_point(self, player: Player) -> bool {
    self.0 & (Self::PUT_BIT | Self::CAPTURED_BIT | Self::PLAYER_BIT) == Self::PUT_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn is_players_empty_base(self, player: Player) -> bool {
    self.0 & (Self::EMPTY_BASE_BIT | Self::PLAYER_BIT) == Self::EMPTY_BASE_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn get_empty_base_player(self) -> Option<Player> {
    if self.is_empty_base() {
      Some(self.get_player())
    } else {
      None
    }
  }

  #[inline]
  pub fn put_point(&mut self, player: Player) {
    self.0 = self.0 & !Self::PLAYER_BIT | player.to_bool() as CellValue | Self::PUT_BIT
  }

  #[inline]
  pub fn set_empty_base_player(&mut self, player: Player) {
    self.0 = self.0 & !Self::PLAYER_BIT | player.to_bool() as CellValue | Self::EMPTY_BASE_BIT
  }

  #[inline]
  pub fn is_bound_player(self, player: Player) -> bool {
    self.0 & (Self::PUT_BIT | Self::PLAYER_BIT | Self::BOUND_BIT)
      == Self::PUT_BIT | Self::BOUND_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn is_putting_allowed(self) -> bool {
    self.0 & (Self::PUT_BIT | Self::CAPTURED_BIT | Self::BAD_BIT) == 0
  }
}
