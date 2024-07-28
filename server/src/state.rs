use crate::{ids::*, message::Response};
use anyhow::Result;
use futures::channel::mpsc::Sender;
use futures_util::SinkExt;
use im::HashSet as ImHashSet;
use oppai_field::{field::Field, player::Player};
use papaya::{HashMap, Operation};
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FieldSize {
  pub width: u32,
  pub height: u32,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OpenGame {
  pub player_id: PlayerId,
  pub size: FieldSize,
}

#[derive(Debug, Clone)]
pub struct Game {
  pub red_player_id: PlayerId,
  pub black_player_id: PlayerId,
  pub size: FieldSize,
  pub field: Arc<RwLock<Field>>,
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
  pub connections: HashMap<ConnectionId, Arc<RwLock<Sender<Response>>>>,
  /// Immutable set allows to use CAS loop which is useful to avoid races when player is deleted.
  pub players: HashMap<PlayerId, ImHashSet<ConnectionId>>,
  /// Open games are never muated, they can be only created or removed.
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

  pub fn remove_players_connection(&self, player_id: PlayerId, connection_id: ConnectionId) {
    self.players.pin().compute(player_id, |entry| match entry {
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

  async fn send_to(tx: &mut Sender<Response>, response: Response) -> Result<()> {
    tokio::time::timeout(Duration::from_millis(10), tx.send(response)).await??;
    Ok(())
  }

  pub async fn send_to_connection(&self, connection_id: ConnectionId, response: Response) -> Result<()> {
    let connection = if let Some(connection) = self.connections.pin().get(&connection_id) {
      connection.clone()
    } else {
      anyhow::bail!("no connection {}", connection_id);
    };
    let mut connection = connection.write().await;

    Self::send_to(&mut connection, response).await
  }

  pub async fn send_to_player(&self, player_id: PlayerId, response: Response) {
    if let Some(connections) = self.players.pin_owned().get(&player_id) {
      for &connection_id in connections {
        if let Err(_) = self.send_to_connection(connection_id, response.clone()).await {
          // TODO: log
        }
      }
    }
  }

  pub async fn send_to_watchers(&self, game_id: GameId, response: Response) {
    if let Some(connections) = self.watchers.pin_owned().get(&game_id) {
      for &connection_id in connections {
        if let Err(_) = self.send_to_connection(connection_id, response.clone()).await {
          // TODO: log
        }
      }
    }
  }

  pub async fn send_to_all(&self, response: Response) {
    for connection in self.connections.pin_owned().values() {
      let mut connection = connection.write().await;
      if let Err(_) = Self::send_to(&mut connection, response.clone()).await {
        // TODO: log
      }
    }
  }
}
