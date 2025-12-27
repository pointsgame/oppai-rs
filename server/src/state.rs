use crate::{ids::*, message::Response};
use anyhow::Result;
use futures::channel::mpsc::Sender;
use imbl::HashSet as ImHashSet;
use oppai_field::{field::Field, player::Player};
use papaya::{Compute, HashMap, Operation};
use std::{
  sync::Arc,
  time::{Duration, SystemTime},
};
use tokio::sync::{Mutex, RwLock};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FieldSize {
  pub width: u32,
  pub height: u32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameTime {
  pub total: Duration,
  pub increment: Duration,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameConfig {
  pub size: FieldSize,
  pub time: GameTime,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OpenGame {
  pub player_id: PlayerId,
  pub config: GameConfig,
}

#[derive(Debug, PartialEq, Clone)]
pub struct GameState {
  pub field: Field,
  pub red_time: Duration,
  pub black_time: Duration,
  pub last_move_time: SystemTime,
  pub draw_offer: Option<Player>,
}

#[derive(Debug, Clone)]
pub struct Game {
  pub red_player_id: PlayerId,
  pub black_player_id: PlayerId,
  pub config: GameConfig,
  pub state: Arc<RwLock<GameState>>,
}

impl Game {
  pub fn color(&self, player_id: PlayerId) -> Option<Player> {
    if self.red_player_id == player_id {
      Some(Player::Red)
    } else if self.black_player_id == player_id {
      Some(Player::Black)
    } else {
      None
    }
  }
}

#[derive(Debug, Clone, Default)]
pub struct State {
  /// Sender is behind a lock since we need exclusive access to send a message.
  /// Also it's useful to make sure we don't send any updates before initial message is sent
  /// but at the same time don't lose that updates.
  pub connections: HashMap<ConnectionId, Arc<Mutex<Sender<Response>>>>,
  /// Immutable set allows to use CAS loop which is useful to avoid races when player is deleted.
  pub players: HashMap<PlayerId, ImHashSet<ConnectionId>>,
  /// Open games are never mutated, they can be only created or removed.
  pub open_games: HashMap<GameId, OpenGame>,
  /// Games have mutable state inside.
  pub games: HashMap<GameId, Game>,
  /// Immutable set just allows to avoid lock here.
  pub watchers: HashMap<GameId, ImHashSet<ConnectionId>>,
}

impl State {
  pub fn insert_players_connection(&self, player_id: PlayerId, connection_id: ConnectionId) {
    self.players.pin().compute(player_id, |entry| match entry {
      Some((_, connections)) if connections.contains(&connection_id) => Operation::Abort(()),
      Some((_, connections)) => Operation::Insert(connections.update(connection_id)),
      None => Operation::Insert(ImHashSet::unit(connection_id)),
    });
  }

  pub fn remove_players_connection(&self, player_id: PlayerId, connection_id: ConnectionId) -> bool {
    let pin = self.players.pin();
    let result = pin.compute(player_id, |entry| match entry {
      Some((_, connections)) if connections.contains(&connection_id) => {
        let new_connections = connections.without(&connection_id);
        if new_connections.is_empty() {
          Operation::Remove
        } else {
          Operation::Insert(new_connections)
        }
      }
      _ => Operation::Abort(()),
    });
    matches!(result, Compute::Removed(_, _))
  }

  pub fn subscribe(&self, connection_id: ConnectionId, game_id: GameId) {
    self.watchers.pin().compute(game_id, |entry| match entry {
      Some((_, connections)) if connections.contains(&connection_id) => Operation::Abort(()),
      Some((_, connections)) => Operation::Insert(connections.update(connection_id)),
      None => Operation::Insert(ImHashSet::unit(connection_id)),
    });
  }

  pub fn unsubscribe(&self, connection_id: ConnectionId, game_id: GameId) {
    self.watchers.pin().compute(game_id, |entry| match entry {
      Some((_, connections)) if connections.contains(&connection_id) => {
        let new_connections = connections.without(&connection_id);
        if new_connections.is_empty() {
          Operation::Remove
        } else {
          Operation::Insert(new_connections)
        }
      }
      _ => Operation::Abort(()),
    });
  }

  pub async fn send_to_connection(&self, connection_id: ConnectionId, response: Response) -> Result<()> {
    let connection = if let Some(connection) = self.connections.pin().get(&connection_id) {
      connection.clone()
    } else {
      anyhow::bail!("no connection {}", connection_id);
    };
    let mut connection = connection.lock().await;

    connection.try_send(response).map_err(From::from)
  }

  pub async fn send_to_player(&self, player_id: PlayerId, response: Response) {
    if let Some(connections) = self.players.pin_owned().get(&player_id) {
      for &connection_id in connections {
        if let Err(error) = self.send_to_connection(connection_id, response.clone()).await {
          self.connections.pin().remove(&connection_id);
          log::warn!("failed to send message to connection {}: {}", connection_id, error);
        }
      }
    }
  }

  pub async fn send_to_watchers(&self, game_id: GameId, response: Response) {
    if let Some(connections) = self.watchers.pin_owned().get(&game_id) {
      for &connection_id in connections {
        if let Err(error) = self.send_to_connection(connection_id, response.clone()).await {
          self.connections.pin().remove(&connection_id);
          log::warn!("failed to send message to connection {}: {}", connection_id, error);
        }
      }
    }
  }

  pub async fn send_to_all(&self, response: Response) {
    let pin = self.connections.pin_owned();
    for (connection_id, connection) in pin.iter() {
      let mut connection = connection.lock().await;
      if let Err(error) = connection.try_send(response.clone()) {
        pin.remove(connection_id);
        log::warn!("failed to send message to connection {}: {}", connection_id, error);
      }
    }
  }

  pub async fn send_to_all_except(&self, except: ConnectionId, response: Response) {
    let pin = self.connections.pin_owned();
    for (&connection_id, connection) in pin.iter() {
      if except == connection_id {
        continue;
      }
      let mut connection = connection.lock().await;
      if let Err(error) = connection.try_send(response.clone()) {
        pin.remove(&connection_id);
        log::warn!("failed to send message to connection {}: {}", connection_id, error);
      }
    }
  }
}
