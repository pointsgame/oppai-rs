use super::*;

use anyhow::{Result, anyhow};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use time::PrimitiveDateTime;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Internal state container for the InMemoryDb
#[derive(Default)]
struct DbState {
  /// Maps User ID -> Player Data
  players: HashMap<Uuid, Player>,
  /// Maps OIDC Subject -> User ID (for efficient lookup in get_or_create)
  oidc_lookup: HashMap<String, Uuid>,
  /// Maps Game ID -> Game Data
  games: HashMap<Uuid, Game>,
  /// Maps Game ID -> List of Moves
  moves: HashMap<Uuid, Vec<Move>>,
  /// Maps Game ID -> List of Draw Offers
  draw_offers: HashMap<Uuid, Vec<DrawOffer>>,
  /// Maps Game ID -> Result Data
  results: HashMap<Uuid, (PrimitiveDateTime, GameResult)>,
}

#[derive(Clone, Default)]
pub struct InMemoryDb {
  state: Arc<RwLock<DbState>>,
}

impl Db for InMemoryDb {
  async fn get_or_create_player<R: Rng>(&self, oidc_player: OidcPlayer, rng: &mut R) -> Result<Player> {
    let mut state = self.state.write().await;

    if let Some(player_id) = state.oidc_lookup.get(&oidc_player.subject) {
      if let Some(player) = state.players.get(player_id) {
        return Ok(player.clone());
      }
    }

    let id = Uuid::new_v4();

    let player = Player {
      id,
      nickname: oidc_player.sanitized_nickname(rng),
    };

    state.oidc_lookup.insert(oidc_player.subject.clone(), id);
    state.players.insert(id, player.clone());

    Ok(player)
  }

  #[cfg(feature = "test")]
  async fn get_or_create_test_player(&self, name: String) -> Result<Player> {
    let mut state = self.state.write().await;

    if let Some(player) = state.players.values().find(|player| player.nickname == name) {
      return Ok(player.clone());
    }

    let id = Uuid::new_v4();
    let player = Player { id, nickname: name };

    state.players.insert(id, player.clone());

    Ok(player)
  }

  async fn get_player(&self, player_id: Uuid) -> Result<Player> {
    let state = self.state.read().await;
    state
      .players
      .get(&player_id)
      .cloned()
      .ok_or_else(|| anyhow!("Player with ID {} not found", player_id))
  }

  async fn get_players(&self, player_ids: &[Uuid]) -> Result<Vec<Player>> {
    let state = self.state.read().await;
    let mut results = Vec::with_capacity(player_ids.len());

    for id in player_ids {
      if let Some(player) = state.players.get(id) {
        results.push(player.clone());
      }
    }

    Ok(results)
  }

  async fn create_game(&self, game: Game) -> Result<()> {
    let mut state = self.state.write().await;

    if state.games.contains_key(&game.id) {
      return Err(anyhow!("Game with ID {} already exists", game.id));
    }

    state.moves.insert(game.id, Vec::new());
    state.draw_offers.insert(game.id, Vec::new());

    state.games.insert(game.id, game);

    Ok(())
  }

  async fn create_move(&self, m: Move) -> Result<()> {
    let mut state = self.state.write().await;

    if !state.games.contains_key(&m.game_id) {
      return Err(anyhow!("Game ID {} not found for move", m.game_id));
    }

    state.moves.entry(m.game_id).or_default().push(m);

    Ok(())
  }

  async fn create_draw_offer(&self, draw_offer: DrawOffer) -> Result<()> {
    let mut state = self.state.write().await;

    if !state.games.contains_key(&draw_offer.game_id) {
      return Err(anyhow!("Game ID {} not found for draw offer", draw_offer.game_id));
    }

    state
      .draw_offers
      .entry(draw_offer.game_id)
      .or_default()
      .push(draw_offer);

    Ok(())
  }

  async fn set_result(&self, game_id: Uuid, finish_time: PrimitiveDateTime, result: GameResult) -> Result<()> {
    let mut state = self.state.write().await;

    if !state.games.contains_key(&game_id) {
      return Err(anyhow!("Game ID {} not found to set result", game_id));
    }

    state.results.insert(game_id, (finish_time, result));

    Ok(())
  }

  async fn update_player_nickname(&self, player_id: Uuid, nickname: String) -> Result<()> {
    let mut state = self.state.write().await;

    if let Some(player) = state.players.get_mut(&player_id) {
      player.nickname = nickname;
      Ok(())
    } else {
      Err(anyhow!("Player with ID {} not found", player_id))
    }
  }

  async fn is_nickname_available(&self, nickname: String) -> Result<bool> {
    let state = self.state.read().await;
    Ok(!state.players.values().any(|player| player.nickname == nickname))
  }

  async fn get_game(&self, game_id: Uuid) -> Result<GameWithMoves> {
    let state = self.state.read().await;

    if let Some(game) = state.games.get(&game_id) {
      let moves = state.moves.get(&game_id).cloned().unwrap_or_default();
      let result = state.results.get(&game_id).cloned();

      Ok(GameWithMoves {
        game: Game {
          id: game.id,
          red_player_id: game.red_player_id,
          black_player_id: game.black_player_id,
          start_time: game.start_time,
          width: game.width,
          height: game.height,
          total_time_ms: game.total_time_ms,
          increment_ms: game.increment_ms,
        },
        moves,
        result,
      })
    } else {
      Err(anyhow!("Game with ID {} not found", game_id))
    }
  }
}
