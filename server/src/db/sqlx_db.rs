use super::*;

use anyhow::Result;
use derive_more::{From, Into};
use rand::Rng;
use sqlx::{Pool, Postgres};
use time::PrimitiveDateTime;
use uuid::Uuid;

#[derive(From, Into)]
pub struct SqlxDb {
  pool: Pool<Postgres>,
}

impl Db for SqlxDb {
  async fn get_or_create_player<R: Rng>(&self, oidc_player: OidcPlayer, rng: &mut R) -> Result<Player> {
    let mut tx = self.pool.begin().await?;

    let player: Option<Player> = sqlx::query_as(
      "
WITH updated AS (
  UPDATE oidc_players
  SET email = $1, email_verified = $2, name = $3, nickname = $4, preferred_username = $5
  WHERE subject = $6
  RETURNING player_id
)
SELECT players.id, players.nickname FROM updated
JOIN players ON updated.player_id = players.id
",
    )
    .bind(oidc_player.email.as_ref())
    .bind(oidc_player.email_verified)
    .bind(oidc_player.name.as_ref())
    .bind(oidc_player.nickname.as_ref())
    .bind(oidc_player.preferred_username.as_ref())
    .bind(oidc_player.subject.as_str())
    .fetch_optional(&mut *tx)
    .await?;

    if let Some(player) = player {
      tx.commit().await?;
      return Ok(player);
    }

    let player: Option<Player> = if let Some(email) = oidc_player.email.as_ref() {
      if oidc_player.email_verified == Some(true) {
        sqlx::query_as(
          "
SELECT players.id, players.nickname FROM oidc_players
JOIN players ON oidc_players.player_id = players.id
WHERE oidc_players.email = $1
LIMIT 1
",
        )
        .bind(email)
        .fetch_optional(&mut *tx)
        .await?
      } else {
        None
      }
    } else {
      None
    };

    let nickname = oidc_player.sanitized_nickname(rng);

    let player = if let Some(player) = player {
      player
    } else {
      sqlx::query_as(
        "
INSERT INTO players (id, nickname, registration_time)
VALUES (gen_random_uuid(), unique_nickname($1), now())
RETURNING id, nickname
",
      )
      .bind(nickname)
      .fetch_one(&mut *tx)
      .await?
    };

    sqlx::query(
      "
INSERT INTO oidc_players (player_id, subject, email, email_verified, \"name\", nickname, preferred_username)
VALUES ($1, $2, $3, $4, $5, $6, $7)
",
    )
    .bind(player.id)
    .bind(&oidc_player.subject)
    .bind(&oidc_player.email)
    .bind(oidc_player.email_verified)
    .bind(&oidc_player.name)
    .bind(&oidc_player.nickname)
    .bind(&oidc_player.preferred_username)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(player)
  }

  #[cfg(feature = "test")]
  async fn get_or_create_test_player(&self, name: String) -> Result<Player> {
    let player_id = Uuid::new_v5(&Uuid::default(), name.as_bytes());

    sqlx::query(
      "
INSERT INTO players (id, nickname, registration_time)
VALUES ($1, unique_nickname($2), now())
ON CONFLICT DO NOTHING
",
    )
    .bind(player_id)
    .bind(&name)
    .execute(&self.pool)
    .await?;

    Ok(Player {
      id: player_id,
      nickname: name,
    })
  }

  async fn get_player(&self, player_id: Uuid) -> Result<Player> {
    sqlx::query_as(
      "
SELECT id, nickname
FROM players
WHERE id = $1
",
    )
    .bind(player_id)
    .fetch_one(&self.pool)
    .await
    .map_err(From::from)
  }

  async fn get_players(&self, player_ids: &[Uuid]) -> Result<Vec<Player>> {
    sqlx::query_as(
      "
SELECT id, nickname
FROM players
WHERE id IN (SELECT unnest($1::uuid[]))
",
    )
    .bind(player_ids)
    .fetch_all(&self.pool)
    .await
    .map_err(From::from)
  }

  async fn create_game(&self, game: Game) -> Result<()> {
    sqlx::query(
      "
INSERT INTO games (id, red_player_id, black_player_id, start_time, width, height, total_time_ms, increment_ms)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
",
    )
    .bind(game.id)
    .bind(game.red_player_id)
    .bind(game.black_player_id)
    .bind(game.start_time)
    .bind(game.width)
    .bind(game.height)
    .bind(game.total_time_ms)
    .bind(game.increment_ms)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn create_move(&self, m: Move) -> Result<()> {
    sqlx::query(
      "
INSERT INTO moves (game_id, player, \"number\", x, y, \"timestamp\")
VALUES ($1, $2, $3, $4, $5, $6)
",
    )
    .bind(m.game_id)
    .bind(m.player)
    .bind(m.number)
    .bind(m.x)
    .bind(m.y)
    .bind(m.timestamp)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn create_draw_offer(&self, draw_offer: DrawOffer) -> Result<()> {
    sqlx::query(
      "
INSERT INTO draw_offers (game_id, player, offer, \"timestamp\")
VALUES ($1, $2, $3, $4)
",
    )
    .bind(draw_offer.game_id)
    .bind(draw_offer.player)
    .bind(draw_offer.offer)
    .bind(draw_offer.timestamp)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn set_result(&self, game_id: Uuid, finish_time: PrimitiveDateTime, result: GameResult) -> Result<()> {
    sqlx::query(
      "
UPDATE games SET \"result\" = $1, finish_time = $2
WHERE id = $3 AND \"result\" IS NULL
",
    )
    .bind(result)
    .bind(finish_time)
    .bind(game_id)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn update_player_nickname(&self, player_id: Uuid, nickname: String) -> Result<()> {
    sqlx::query(
      "
UPDATE players SET nickname = $1
WHERE id = $2
",
    )
    .bind(nickname)
    .bind(player_id)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn is_nickname_available(&self, nickname: String) -> Result<bool> {
    let count: i64 = sqlx::query_scalar(
      "
SELECT COUNT(*) FROM players WHERE nickname = $1
",
    )
    .bind(nickname)
    .fetch_one(&self.pool)
    .await?;
    Ok(count == 0)
  }

  async fn get_game(&self, game_id: Uuid) -> Result<GameWithMoves> {
    let game = sqlx::query_as::<_, Game>(
      "SELECT id, red_player_id, black_player_id, start_time, width, height, total_time_ms, increment_ms, result, finish_time FROM games WHERE id = $1"
    )
    .bind(game_id)
    .fetch_one(&self.pool)
    .await?;

    let moves = sqlx::query_as::<_, Move>(
      "SELECT game_id, player, \"number\", x, y, \"timestamp\" FROM moves WHERE game_id = $1 ORDER BY \"number\" ASC",
    )
    .bind(game_id)
    .fetch_all(&self.pool)
    .await?;

    Ok(GameWithMoves { game, moves })
  }
}
