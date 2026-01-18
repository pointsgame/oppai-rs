use anyhow::Result;
use oppai_field::player::Player as OppaiPlayer;
use rand::{Rng, distr::Alphanumeric};
use time::PrimitiveDateTime;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "provider")]
#[sqlx(rename_all = "lowercase")]
pub enum Color {
  Red,
  Black,
}

impl From<OppaiPlayer> for Color {
  fn from(player: OppaiPlayer) -> Self {
    match player {
      OppaiPlayer::Red => Color::Red,
      OppaiPlayer::Black => Color::Black,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct Player {
  pub id: Uuid,
  pub nickname: String,
}

pub struct OidcPlayer {
  pub subject: String,
  pub email: Option<String>,
  pub email_verified: Option<bool>,
  pub name: Option<String>,
  pub nickname: Option<String>,
  pub preferred_username: Option<String>,
}

impl OidcPlayer {
  pub fn nickname(&self) -> Option<&str> {
    self
      .preferred_username
      .as_deref()
      .or(self.nickname.as_deref())
      .or(self.name.as_deref())
  }

  pub fn sanitized_nickname<R: Rng>(&self, rng: &mut R) -> String {
    self
      .nickname()
      .map(|nickname| {
        // Sanitize the nickname by replacing invalid characters with underscores
        nickname
          .chars()
          .map(|c| if c.is_alphanumeric() { c } else { '_' })
          .collect::<String>()
      })
      .unwrap_or_else(|| {
        format!(
          "player_{}",
          rng
            .sample_iter(&Alphanumeric)
            .map(|n| n as char)
            .take(4)
            .collect::<String>()
        )
      })
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

#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct Game {
  pub id: Uuid,
  pub red_player_id: Uuid,
  pub black_player_id: Uuid,
  pub start_time: PrimitiveDateTime,
  pub width: i32,
  pub height: i32,
  pub total_time_ms: i64,
  pub increment_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct Move {
  pub game_id: Uuid,
  pub player: Color,
  pub number: i16,
  pub x: i16,
  pub y: i16,
  pub timestamp: PrimitiveDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
pub struct DrawOffer {
  pub game_id: Uuid,
  pub player: Color,
  pub offer: bool,
  pub timestamp: PrimitiveDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameWithMoves {
  pub game: Game,
  pub moves: Vec<Move>,
  pub result: Option<(PrimitiveDateTime, GameResult)>,
}

pub trait Db {
  async fn get_or_create_player<R: Rng>(&self, oidc_player: OidcPlayer, rng: &mut R) -> Result<Player>;
  #[cfg(feature = "test")]
  async fn get_or_create_test_player(&self, name: String) -> Result<Player>;
  async fn get_player(&self, player_id: Uuid) -> Result<Player>;
  async fn get_players(&self, player_ids: &[Uuid]) -> Result<Vec<Player>>;
  async fn create_game(&self, game: Game) -> Result<()>;
  async fn create_move(&self, m: Move) -> Result<()>;
  async fn create_draw_offer(&self, draw_offer: DrawOffer) -> Result<()>;
  async fn set_result(&self, game_id: Uuid, finish_time: PrimitiveDateTime, result: GameResult) -> Result<()>;
  async fn update_player_nickname(&self, player_id: Uuid, nickname: String) -> Result<()>;
  async fn is_nickname_available(&self, nickname: String) -> Result<bool>;
  async fn get_game(&self, game_id: Uuid) -> Result<GameWithMoves>;
}
