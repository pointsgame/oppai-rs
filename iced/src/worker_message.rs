use oppai_bot::field::Pos;
use oppai_bot::player::Player;
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
