use serde::{Serialize, Deserialize};
use uuid::Uuid;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConnectionId(pub Uuid);

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlayerId(pub Uuid);

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GameId(pub Uuid);
