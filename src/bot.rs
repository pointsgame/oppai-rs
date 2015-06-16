use std::sync::*;
use rand::*;
use types::*;
use player::*;
use zobrist::*;
use field::*;
use uct::*;

pub struct Bot {
  rng: XorShiftRng,
  zobrist: Arc<Zobrist>,
  field: Field,
  uct: UctRoot
}

impl Bot {
  pub fn new(width: Coord, height: Coord) -> Bot {
    let length = length(width, height);
    let mut rng = XorShiftRng::new_unseeded();
    let zobrist = Arc::new(Zobrist::new(length, &mut rng));
    let field_zobrist = zobrist.clone();
    Bot {
      rng: rng,
      zobrist: zobrist,
      field: Field::new(width, height, field_zobrist),
      uct: UctRoot::new(length)
    }
  }

  pub fn best_move(&mut self, player: Player, time: Time) -> Option<(Coord, Coord)> {
    self.uct.best_move(&self.field, player, &mut self.rng, time).map(|pos| (self.field.to_x(pos), self.field.to_y(pos)))
  }

  pub fn put_point(&mut self, x: Coord, y: Coord, player: Player) -> bool {
    let pos = self.field.to_pos(x, y);
    self.field.put_point(pos, player)
  }

  pub fn undo(&mut self) -> bool {
    self.field.undo()
  }
}
