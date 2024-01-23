use oppai_field::field::Pos;
use oppai_field::player::Player;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Request {
  PutPoint(Pos, Player),
  Undo,
  UndoAll,
  BestMove(Player),
  New(u32, u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Response {
  BestMove(Pos),
  Init,
}
