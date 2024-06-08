use crate::{
  extended_field::ExtendedField,
  field::{Field, Pos},
  player::Player,
};
use rand::Rng;

pub trait AnyField {
  fn new_from_rng<R: Rng>(width: u32, height: u32, rng: &mut R) -> Self;
  fn put_players_point(&mut self, pos: Pos, player: Player) -> bool;
  fn undo(&mut self) -> bool;
  fn clear(&mut self);
  fn field(&self) -> &Field;
}

impl AnyField for Field {
  fn new_from_rng<R: Rng>(width: u32, height: u32, rng: &mut R) -> Self {
    Field::new_from_rng(width, height, rng)
  }
  fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    self.put_point(pos, player)
  }
  fn undo(&mut self) -> bool {
    self.undo()
  }
  fn clear(&mut self) {
    self.clear()
  }
  fn field(&self) -> &Field {
    self
  }
}

impl AnyField for ExtendedField {
  fn new_from_rng<R: Rng>(width: u32, height: u32, rng: &mut R) -> Self {
    ExtendedField::new_from_rng(width, height, rng)
  }
  fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    self.put_players_point(pos, player)
  }
  fn undo(&mut self) -> bool {
    self.undo()
  }
  fn clear(&mut self) {
    self.clear()
  }
  fn field(&self) -> &Field {
    &self.field
  }
}
