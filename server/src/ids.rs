use derive_more::{Display, From, Into};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Display, PartialEq, Eq, Clone, Copy, Hash, Default, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConnectionId(pub Uuid);

#[derive(Debug, Display, PartialEq, Eq, Clone, Copy, Hash, Default, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlayerId(pub Uuid);

#[derive(Debug, Display, PartialEq, Eq, Clone, Copy, Hash, Default, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GameId(pub Uuid);
