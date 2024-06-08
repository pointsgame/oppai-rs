mod config;

use std::cmp::Ordering;
use std::fmt;
#[cfg(feature = "term-render")]
use std::io::Error;
use std::io::Write;
use std::io::{stdout, Result};
use std::ops::Add;
use std::time::Duration;

use config::cli_parse;
use crossterm::{
  cursor::MoveTo,
  terminal::{Clear, ClearType},
  QueueableCommand,
};
use oppai_client::Client;
use oppai_field::extended_field::ExtendedField;
use oppai_field::field::{NonZeroPos, Pos};
use oppai_field::player::Player;
use oppai_initial::initial::InitialPosition;
#[cfg(feature = "term-render")]
use oppai_term_render::render;
use rand::rngs::SmallRng;
use rand::SeedableRng;

const WIDTH: u32 = 10;
const HEIGHT: u32 = 10;
const INITIAL_POSITION: InitialPosition = InitialPosition::Cross;
const TIME: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Stats {
  wins: u32,
  loses: u32,
  draws: u32,
}

impl Add<Stats> for Stats {
  type Output = Self;

  fn add(self, rhs: Stats) -> Self::Output {
    Stats {
      wins: self.wins + rhs.wins,
      loses: self.loses + rhs.loses,
      draws: self.draws + rhs.draws,
    }
  }
}

impl fmt::Display for Stats {
  fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
    write!(f, "{}/{}/{}", self.wins, self.draws, self.loses)
  }
}

impl Stats {
  const WIN: Self = Stats {
    wins: 1,
    loses: 0,
    draws: 0,
  };

  const LOOSE: Self = Stats {
    wins: 0,
    loses: 1,
    draws: 0,
  };

  const DRAW: Self = Stats {
    wins: 0,
    loses: 0,
    draws: 1,
  };
}

struct Game {
  field: ExtendedField,
  client1: Client,
  client2: Client,
}

impl Game {
  async fn best_move(&mut self, player: Player, swap: bool) -> Result<Option<NonZeroPos>> {
    let moves = if swap {
      self.client2.analyze(player, TIME).await?
    } else {
      self.client1.analyze(player, TIME).await?
    };
    Ok(
      moves
        .into_iter()
        .max_by(|m1, m2| m1.weight.partial_cmp(&m2.weight).unwrap())
        .map(|m| self.field.field.to_pos(m.coords.x, m.coords.y))
        .and_then(NonZeroPos::new),
    )
  }

  async fn put_point(&mut self, pos: Pos, player: Player) -> Result<bool> {
    if self.field.put_players_point(pos, player) {
      let x = self.field.field.to_x(pos);
      let y = self.field.field.to_y(pos);
      Ok(self.client1.put_point(x, y, player).await? && self.client2.put_point(x, y, player).await?)
    } else {
      Ok(false)
    }
  }

  async fn place_initial_position(&mut self, player: Player, initial_position: InitialPosition) -> Result<()> {
    for (pos, player) in initial_position.points(self.field.field.width(), self.field.field.height(), player) {
      self.put_point(pos, player).await?;
    }
    Ok(())
  }

  fn is_game_over(&mut self) -> bool {
    self.field.field.is_game_over()
  }

  fn stats(&self, swap: bool) -> Stats {
    match self.field.field.score(Player::Red).cmp(&0) {
      Ordering::Less => {
        if swap {
          Stats::WIN
        } else {
          Stats::LOOSE
        }
      }
      Ordering::Greater => {
        if swap {
          Stats::LOOSE
        } else {
          Stats::WIN
        }
      }
      Ordering::Equal => Stats::DRAW,
    }
  }

  fn clear_screen() {
    let mut out = stdout();
    out.queue(Clear(ClearType::All)).unwrap();
    out.queue(Clear(ClearType::Purge)).unwrap();
    out.queue(MoveTo(0, 0)).unwrap();
    out.flush().unwrap();
  }

  #[cfg(not(feature = "term-render"))]
  fn draw(&self, stats: &Stats, _ascii: bool) -> Result<()> {
    Game::clear_screen();
    print!("{}", self.field.field);
    println!("{}", stats);
    Ok(())
  }

  #[cfg(feature = "term-render")]
  fn draw(&self, stats: &Stats, ascii: bool) -> Result<()> {
    Game::clear_screen();
    if ascii {
      print!("{}", self.field.field);
    } else {
      let config = oppai_svg::Config {
        width: 256,
        height: 256,
        ..oppai_svg::Config::default()
      };
      render(&self.field, &config).map_err(Error::other)?;
    }
    println!("{}", stats);
    Ok(())
  }

  async fn play(&mut self, mut player: Player, swap: bool, stats: &mut Stats, ascii: bool) -> Result<()> {
    let mut cur_swap = swap;
    self.draw(stats, ascii)?;
    while let Some(pos) = self.best_move(player, cur_swap).await? {
      if !self.put_point(pos.get(), player).await? {
        break;
      }
      self.draw(stats, ascii)?;
      if self.is_game_over() {
        break;
      }
      player = player.next();
      cur_swap = !cur_swap;
    }
    *stats = *stats + self.stats(swap);
    Ok(())
  }

  async fn init(&mut self) -> Result<()> {
    self.field.clear();
    self
      .client1
      .init(self.field.field.width(), self.field.field.height())
      .await?;
    self
      .client2
      .init(self.field.field.width(), self.field.field.height())
      .await?;
    Ok(())
  }
}

fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let config = cli_parse();

  let mut rng = SmallRng::from_entropy();
  let mut game = Game {
    field: ExtendedField::new_from_rng(WIDTH, HEIGHT, &mut rng),
    client1: Client::spawn(config.ai1, config.ai1_args)?,
    client2: Client::spawn(config.ai2, config.ai2_args)?,
  };

  let player = Player::default();
  let mut stats = Stats::default();
  let mut swap = false;

  #[cfg(feature = "term-render")]
  let ascii = config.ascii;
  #[cfg(not(feature = "term-render"))]
  let ascii = true;
  let future = async {
    loop {
      game.init().await?;
      game.place_initial_position(player, INITIAL_POSITION).await?;
      game.play(player, swap, &mut stats, ascii).await?;
      swap = !swap;
    }
  };

  futures::executor::block_on(future)
}
