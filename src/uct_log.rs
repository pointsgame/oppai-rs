use types::Pos;

pub enum UctLog {
  BestMove(Pos, f32),
  Estimation(Pos, f32, usize, usize, usize)
}
