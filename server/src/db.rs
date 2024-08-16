use anyhow::Result;
use derive_more::{From, Into};
use rand::{distributions::Alphanumeric, Rng};
use sqlx::{Pool, Postgres};
use time::PrimitiveDateTime;
use uuid::Uuid;

#[derive(sqlx::FromRow)]
pub struct Player {
  pub id: Uuid,
  pub nickname: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "provider")]
#[sqlx(rename_all = "lowercase")]
pub enum Provider {
  Portier,
  Google,
  GitLab,
}

pub struct OidcPlayer {
  pub provider: Provider,
  pub subject: String,
  pub email: Option<String>,
  pub email_verified: Option<bool>,
  pub name: Option<String>,
  pub nickname: Option<String>,
  pub preferred_username: Option<String>,
}

impl OidcPlayer {
  fn nickname(&self) -> Option<&str> {
    self
      .preferred_username
      .as_deref()
      .or(self.nickname.as_deref())
      .or(self.name.as_deref())
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "provider")]
#[sqlx(rename_all = "lowercase")]
pub enum GameResult {
  ResignedRed,
  ResignedBlack,
  GroundedRed,
  GroundedBlack,
  TimeOutRed,
  TimeOutBlack,
  DrawAgreement,
  DrawGrounded,
}

pub struct Game {
  pub id: Uuid,
  pub red_player_id: Uuid,
  pub black_player_id: Uuid,
  pub start_time: PrimitiveDateTime,
}

pub struct Move {
  pub game_id: Uuid,
  pub player_id: Uuid,
  pub number: i16,
  pub x: i16,
  pub y: i16,
  pub putting_time: PrimitiveDateTime,
}

pub trait Db {
  async fn get_or_create_player<R: Rng>(&self, oidc_player: OidcPlayer, rng: &mut R) -> Result<Player>;
  #[cfg(feature = "test")]
  async fn get_or_create_test_player(&self, name: String) -> Result<Player>;
  async fn get_player(&self, player_id: Uuid) -> Result<Player>;
  async fn get_players(&self, player_ids: &[Uuid]) -> Result<Vec<Player>>;
  async fn create_game(&self, game: Game) -> Result<()>;
  async fn create_move(&self, m: Move) -> Result<()>;
  async fn set_result(&self, game_id: Uuid, result: GameResult) -> Result<()>;
}

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
  WHERE subject = $6 AND provider = $7
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
    .bind(oidc_player.provider)
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

    let nickname = oidc_player
      .nickname()
      .map(|nickname| nickname.to_string())
      .unwrap_or_else(|| {
        format!(
          "player_{}",
          rng
            .sample_iter(&Alphanumeric)
            .map(|n| n as char)
            .take(4)
            .collect::<String>()
        )
      });

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
INSERT INTO oidc_players (player_id, provider, subject, email, email_verified, \"name\", nickname, preferred_username)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
",
    )
    .bind(player.id)
    .bind(oidc_player.provider)
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
INSERT INTO games (id, red_player_id, black_player_id, start_time)
VALUES ($1, $2, $3, $4)
",
    )
    .bind(game.id)
    .bind(game.red_player_id)
    .bind(game.black_player_id)
    .bind(game.start_time)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn create_move(&self, m: Move) -> Result<()> {
    sqlx::query(
      "
INSERT INTO moves (game_id, player_id, \"number\", x, y, putting_time)
VALUES ($1, $2, $3, $4, $5, $6)
",
    )
    .bind(m.game_id)
    .bind(m.player_id)
    .bind(m.number)
    .bind(m.x)
    .bind(m.y)
    .bind(m.putting_time)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }

  async fn set_result(&self, game_id: Uuid, result: GameResult) -> Result<()> {
    sqlx::query(
      "
UPDATE games SET \"result\" = $1
WHERE id = $2 AND \"result\" IS NULL
",
    )
    .bind(result)
    .bind(game_id)
    .execute(&self.pool)
    .await
    .map_err(From::from)
    .map(|_| ())
  }
}
