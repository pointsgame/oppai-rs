use types::Coord;

pub enum UctLog {
  BestMove(Coord, Coord, f32),
  Estimation(Coord, Coord, f32, usize, usize, usize)
}
