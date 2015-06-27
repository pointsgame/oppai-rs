use std::sync::Arc;
use rand::XorShiftRng;
use types::{Coord, Time};
use player::Player;
use zobrist::Zobrist;
use field;
use field::Field;
use uct::UctRoot;

pub struct Bot {
  rng: XorShiftRng,
  zobrist: Arc<Zobrist>,
  field: Field,
  uct: UctRoot
}

impl Bot {
  pub fn new(width: Coord, height: Coord) -> Bot {
    let length = field::length(width, height);
    let mut rng = XorShiftRng::new_unseeded();
    let zobrist = Arc::new(Zobrist::new(length * 2, &mut rng));
    let field_zobrist = zobrist.clone();
    Bot {
      rng: rng,
      zobrist: zobrist,
      field: Field::new(width, height, field_zobrist),
      uct: UctRoot::new(length)
    }
  }

  pub fn best_move_with_time(&mut self, player: Player, time: Time) -> Option<(Coord, Coord)> {
    self.uct.best_move_with_time(&self.field, player, &mut self.rng, time).map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
  }

  pub fn best_move_with_complexity(&mut self, player: Player, complexity: u8) -> Option<(Coord, Coord)> {
    self.uct.best_move_with_iterations_count(&self.field, player, &mut self.rng, 250000).map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
  }

  pub fn put_point(&mut self, x: Coord, y: Coord, player: Player) -> bool {
    let pos = self.field.to_pos(x, y);
    self.field.put_point(pos, player)
  }

  pub fn undo(&mut self) -> bool {
    self.field.undo()
  }
}
