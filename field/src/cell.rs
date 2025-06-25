use crate::player::Player;

type CellValue = u8;

const PLAYER_BIT: CellValue = 1;

const PUT_BIT: CellValue = 2;

const CAPTURED_BIT: CellValue = 4;

const BOUND_BIT: CellValue = 8;

const EMPTY_BASE_BIT: CellValue = 16;

const BAD_BIT: CellValue = 32;

const TAG_BIT: CellValue = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
  value: CellValue,
}

impl Cell {
  #[inline]
  pub fn new(bad: bool) -> Cell {
    Cell {
      value: if bad { BAD_BIT } else { 0 },
    }
  }

  #[inline]
  pub fn get_player(self) -> Player {
    Player::from_bool(self.value & PLAYER_BIT != 0)
  }

  #[inline]
  pub fn set_player(&mut self, player: Player) {
    self.value = self.value & !PLAYER_BIT | player.to_bool() as CellValue
  }

  #[inline]
  pub fn is_put(self) -> bool {
    self.value & PUT_BIT != 0
  }

  #[inline]
  pub fn set_put(&mut self) {
    self.value |= PUT_BIT
  }

  #[inline]
  pub fn clear_put(&mut self) {
    self.value &= !PUT_BIT
  }

  #[inline]
  pub fn is_captured(self) -> bool {
    self.value & CAPTURED_BIT != 0
  }

  #[inline]
  pub fn set_captured(&mut self) {
    self.value |= CAPTURED_BIT
  }

  #[inline]
  pub fn clear_captured(&mut self) {
    self.value &= !CAPTURED_BIT
  }

  #[inline]
  pub fn is_bound(self) -> bool {
    self.value & BOUND_BIT != 0
  }

  #[inline]
  pub fn set_bound(&mut self) {
    self.value |= BOUND_BIT
  }

  #[inline]
  pub fn clear_bound(&mut self) {
    self.value &= !BOUND_BIT
  }

  #[inline]
  pub fn is_empty_base(self) -> bool {
    self.value & EMPTY_BASE_BIT != 0
  }

  #[inline]
  pub fn set_empty_base(&mut self) {
    self.value |= EMPTY_BASE_BIT
  }

  #[inline]
  pub fn clear_empty_base(&mut self) {
    self.value &= !EMPTY_BASE_BIT
  }

  #[inline]
  pub fn is_bad(self) -> bool {
    self.value & BAD_BIT != 0
  }

  #[inline]
  pub fn set_bad(&mut self) {
    self.value |= BAD_BIT
  }

  #[inline]
  pub fn clear_bad(&mut self) {
    self.value &= !BAD_BIT
  }

  #[inline]
  pub fn is_tagged(self) -> bool {
    self.value & TAG_BIT != 0
  }

  #[inline]
  pub fn set_tag(&mut self) {
    self.value |= TAG_BIT
  }

  #[inline]
  pub fn clear_tag(&mut self) {
    self.value &= !TAG_BIT
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
    self.is_put() && self.get_player() == player
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
    self.is_put() && !self.is_captured() && self.get_player() == player
  }

  #[inline]
  pub fn is_players_empty_base(self, player: Player) -> bool {
    self.is_empty_base() && self.get_player() == player
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
    self.set_player(player);
    self.value |= PUT_BIT
  }

  #[inline]
  pub fn set_empty_base_player(&mut self, player: Player) {
    self.value = self.value & !PLAYER_BIT | player.to_bool() as CellValue | EMPTY_BASE_BIT
  }

  #[inline]
  pub fn is_bound_player(self, player: Player) -> bool {
    self.is_bound() && self.is_players_point(player)
  }

  #[inline]
  pub fn is_putting_allowed(self) -> bool {
    !self.is_put() && !self.is_captured() && !self.is_bad()
  }
}
