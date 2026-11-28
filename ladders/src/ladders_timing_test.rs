use crate::ladders::ladders;
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn timing_field(game: u32, min: u32, max: u32) -> Field {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(777 ^ game as u64);
  let width = rng.random_range(min..max);
  let height = rng.random_range(min..max);
  let mut field = Field::new_from_rng(width, height, &mut rng);
  let density = rng.random_range(0.1..0.8);
  let moves_count = (width * height) as f64 * density;
  for _ in 0..moves_count as u32 {
    let x = rng.random_range(0..width);
    let y = rng.random_range(0..height);
    let player = if rng.random::<bool>() {
      Player::Red
    } else {
      Player::Black
    };
    field.put_point(field.to_pos(x, y), player);
  }
  field
}

// The defender used to try its captures unrelated to the ladder in every
// order, taking seconds on these positions.
#[test]
fn ladders_many_counter_captures() {
  use std::time::{Duration, Instant};
  for (game, player) in [(2686, Player::Red), (11032, Player::Black), (19088, Player::Red)] {
    let mut field = timing_field(game, 10, 40);
    let start = Instant::now();
    let stopped = std::cell::Cell::new(false);
    ladders(&mut field, player, &|| {
      stopped.set(start.elapsed() > Duration::from_secs(2));
      stopped.get()
    });
    assert!(!stopped.get(), "game {} {:?}", game, player);
  }
}

/// Prints the occupied area with SGF coordinates: W/w for live/captured red
/// points, B/b for black ones, `,` for empty cells inside bases.
fn print_area(field: &Field) {
  let occupied = field.moves.iter().map(|&pos| (field.to_x(pos), field.to_y(pos)));
  let (min_x, max_x, min_y, max_y) = occupied.fold((u32::MAX, 0, u32::MAX, 0), |(a, b, c, d), (x, y)| {
    (a.min(x), b.max(x), c.min(y), d.max(y))
  });
  let (min_x, min_y) = (min_x.saturating_sub(2), min_y.saturating_sub(2));
  let (max_x, max_y) = ((max_x + 2).min(field.width() - 1), (max_y + 2).min(field.height() - 1));
  print!("  ");
  for x in min_x..=max_x {
    print!("{}", (b'a' + x as u8) as char);
  }
  println!();
  for y in min_y..=max_y {
    print!("{} ", (b'a' + y as u8) as char);
    for x in min_x..=max_x {
      let cell = field.cell(field.to_pos(x, y));
      let c = match cell.get_players_point() {
        Some(Player::Red) if cell.is_captured() => 'w',
        Some(Player::Red) => 'W',
        Some(Player::Black) if cell.is_captured() => 'b',
        Some(Player::Black) => 'B',
        None if cell.is_empty_base() || cell.is_captured() => ',',
        None => '.',
      };
      print!("{}", c);
    }
    println!();
  }
}

/// Times `ladders` on random positions and prints the slowest ones.
#[test]
#[ignore = "timing fuzzing run, set TIMING_GAMES, TIMING_MIN and TIMING_MAX"]
fn ladders_timing() {
  use std::time::{Duration, Instant};
  let games: u32 = std::env::var("TIMING_GAMES").map_or(2000, |s| s.parse().unwrap());
  let min: u32 = std::env::var("TIMING_MIN").map_or(10, |s| s.parse().unwrap());
  let max: u32 = std::env::var("TIMING_MAX").map_or(40, |s| s.parse().unwrap());
  let limit = Duration::from_secs(2);
  let mut times = Vec::new();
  for game in 0..games {
    let mut field = timing_field(game, min, max);
    for player in [Player::Red, Player::Black] {
      let start = Instant::now();
      let result = ladders(&mut field, player, &|| start.elapsed() > limit);
      times.push((start.elapsed(), game, player, field.width(), field.height(), result));
    }
  }
  times.sort_by_key(|t| std::cmp::Reverse(t.0));
  let total: Duration = times.iter().map(|t| t.0).sum();
  println!("total {:?}, mean {:?}", total, total / times.len() as u32);
  for t in times.iter().take(10) {
    println!("{:?} game {} {:?} {}x{} {:?}", t.0, t.1, t.2, t.3, t.4, t.5);
  }
}

/// Times `ladders` on a single position of `ladders_timing`.
#[test]
#[ignore = "timing helper, set TIMING_GAME, TIMING_PLAYER and TIMING_LIMIT"]
fn ladders_timing_single() {
  use std::time::{Duration, Instant};
  let game: u32 = std::env::var("TIMING_GAME").unwrap().parse().unwrap();
  let player = if std::env::var("TIMING_PLAYER").unwrap() == "Red" {
    Player::Red
  } else {
    Player::Black
  };
  let limit = Duration::from_secs(std::env::var("TIMING_LIMIT").map_or(60, |s| s.parse().unwrap()));
  let mut field = timing_field(game, 10, 40);
  print_area(&field);
  let start = Instant::now();
  let result = ladders(&mut field, player, &|| start.elapsed() > limit);
  println!("{:?}: {:?} in {:?}", player, result, start.elapsed());
}
