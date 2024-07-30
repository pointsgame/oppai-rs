use anyhow::Result;
use derive_more::{From, Into};
use sqlx::{Pool, Postgres};
use uuid::Uuid;

// pub struct Player {
//   pub id: Uuid,
//   pub registration_time: Instant,
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "provider")]
#[sqlx(rename_all = "lowercase")]
pub enum Provider {
  Google,
}

pub struct OidcPlayer {
  pub provider: Provider,
  pub subject: String,
  pub email: Option<String>,
  pub name: Option<String>,
  pub nickname: Option<String>,
  pub preferred_username: Option<String>,
}

#[derive(From, Into)]
pub struct SqlxDb {
  pool: Pool<Postgres>,
}

impl SqlxDb {
  pub async fn get_or_create_player(&self, oidc_player: OidcPlayer) -> Result<Uuid> {
    let mut tx = self.pool.begin().await?;

    let player_id: Option<(Uuid,)> = sqlx::query_as(
      "
WITH updated AS (
  UPDATE oidc_players
  SET email = $1, name = $2, nickname = $3, preferred_username = $4
  WHERE subject = $5 AND provider = $6
  RETURNING player_id
)
SELECT players.id FROM updated
JOIN players ON updated.player_id = players.id
",
    )
    .bind(oidc_player.email.as_ref())
    .bind(oidc_player.name.as_ref())
    .bind(oidc_player.nickname.as_ref())
    .bind(oidc_player.preferred_username.as_ref())
    .bind(oidc_player.subject.as_str())
    .bind(oidc_player.provider)
    .fetch_optional(&mut *tx)
    .await?;

    if let Some((player_id,)) = player_id {
      tx.commit().await?;
      return Ok(player_id);
    }

    let player_id = if let Some(email) = oidc_player.email.as_ref() {
      let player_id: Option<(Uuid,)> = sqlx::query_as(
        "
SELECT players.id FROM oidc_players
JOIN players ON oidc_players.player_id = players.id
WHERE oidc_players.email = $1
LIMIT 1
",
      )
      .bind(email)
      .fetch_optional(&mut *tx)
      .await?;
      player_id.map(|(player_id,)| player_id)
    } else {
      None
    };

    let player_id = if let Some(player_id) = player_id {
      player_id
    } else {
      let (player_id,) = sqlx::query_as(
        "
INSERT INTO players (id, registration_time)
VALUES (gen_random_uuid(), now())
RETURNING id
",
      )
      .fetch_one(&mut *tx)
      .await?;
      player_id
    };

    sqlx::query(
      "
INSERT INTO oidc_players (player_id, provider, subject, email, \"name\", nickname, preferred_username)
VALUES ($1, $2, $3, $4, $5, $6, $7)
",
    )
    .bind(player_id)
    .bind(oidc_player.provider)
    .bind(oidc_player.subject)
    .bind(oidc_player.email)
    .bind(oidc_player.name)
    .bind(oidc_player.nickname)
    .bind(oidc_player.preferred_username)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(player_id)
  }
}
