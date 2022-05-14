use std::cmp::Ordering;
use std::fmt;
use std::ops::Add;
use std::time::Duration;

use oppai_bot_1::bot::Bot as Bot1;
use oppai_bot_1::field::{NonZeroPos, Pos};
use oppai_bot_1::player::Player as Player1;
use oppai_bot_2::bot::Bot as Bot2;
use oppai_bot_2::player::Player as Player2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const WIDTH: u32 = 10;
const HEIGHT: u32 = 10;
const TIME: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Player(Player1, Player2);

impl Player {
  fn next(self) -> Player {
    Player(self.0.next(), self.1.next())
  }
}

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
  bot1: Bot1<SmallRng>,
  bot2: Bot2<SmallRng>,
}

impl Game {
  fn best_move(&mut self, player: Player, swap: bool) -> Option<NonZeroPos> {
    if swap {
      self.bot2.best_move_with_time(player.1, TIME, &Default::default())
    } else {
      self.bot1.best_move_with_time(player.0, TIME, &Default::default())
    }
  }

  fn put_point(&mut self, pos: Pos, player: Player) -> bool {
    self.bot1.field.put_point(pos, player.0) && self.bot2.field.put_point(pos, player.1)
  }

  fn is_game_over(&self) -> bool {
    self.bot1.field.is_game_over()
  }

  fn stats(&self, swap: bool) -> Stats {
    match self.bot1.field.score(Player1::Red).cmp(&0) {
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

  fn play(&mut self, swap: bool, stats: &mut Stats) {
    let mut player = Player::default();
    let mut cur_swap = swap;
    print!("\x1B[2J\x1B[1;1H");
    print!("{}\n{}", stats, self.bot1.field);
    while let Some(pos) = self.best_move(player, cur_swap) {
      if !self.put_point(pos.get(), player) {
        break;
      }
      print!("\x1B[2J\x1B[1;1H");
      print!("{}\n{}", stats, self.bot1.field);
      if self.is_game_over() {
        break;
      }
      player = player.next();
      cur_swap = !cur_swap;
    }
    *stats = *stats + self.stats(swap);
  }

  fn clear(&mut self) {
    self.bot1.clear();
    self.bot2.clear();
  }
}

fn main() {
  let mut rng1 = SmallRng::from_entropy();
  let rng2 = SmallRng::from_seed(rng1.gen());
  let mut game = Game {
    bot1: Bot1::new(WIDTH, HEIGHT, rng1, Default::default(), Default::default()),
    bot2: Bot2::new(WIDTH, HEIGHT, rng2, Default::default(), Default::default()),
  };

  let mut stats = Stats::default();
  let mut swap = false;
  loop {
    game.play(swap, &mut stats);
    game.clear();
    swap = !swap;
  }
}
