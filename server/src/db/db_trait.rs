use anyhow::Result;
use oppai_field::player::Player as OppaiPlayer;
use rand::Rng;
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
  pub player: Color,
  pub number: i16,
  pub x: i16,
  pub y: i16,
  pub timestamp: PrimitiveDateTime,
}

pub struct DrawOffer {
  pub game_id: Uuid,
  pub player: Color,
  pub offer: bool,
  pub timestamp: PrimitiveDateTime,
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
}
