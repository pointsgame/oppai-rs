#![cfg_attr(feature="bench", feature(test))]
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

#![allow(dead_code)]
#![cfg_attr(feature="clippy", allow(too_many_arguments))]
#![cfg_attr(feature="clippy", allow(cyclomatic_complexity))]

extern crate rand;

#[macro_use]
extern crate log;

extern crate log4rs;

extern crate num_cpus;

extern crate rustc_serialize;

extern crate toml;

extern crate crossbeam;

extern crate rayon;

extern crate tar;

#[cfg(test)]
extern crate quickcheck;

#[cfg(all(test, feature="bench"))]
extern crate test;

mod config;
mod player;
mod zobrist;
mod cell;
mod field;
mod wave_pruning;
mod trajectories_pruning;
mod common;
mod uct;
mod heuristic;
mod minimax;
mod bot;
mod rotate;
mod dfa;
mod spiral;
mod patterns;

#[cfg(test)]
mod construct_field;
#[cfg(test)]
mod field_test;
#[cfg(test)]
mod patterns_test;

#[cfg(all(test, feature="bench"))]
mod field_benchmark;
#[cfg(all(test, feature="bench"))]
mod uct_benchmark;

use std::io;
use std::io::{Write, BufReader, BufRead};
use std::str::FromStr;
use std::path::Path;
use std::fs::File;
use std::sync::Arc;
use std::default::Default;
use player::Player;
use bot::Bot;
use patterns::Patterns;

const CONFIG_PATH: &'static str = "config/config.toml";

const LOG_CONFIG_PATH: &'static str = "config/log.toml";

const PATTERNS_PATH: &'static str = "patterns.tar";

fn write_author<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} author kurnevsky_evgeny", id).ok();
}

fn write_author_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} author", id).ok();
}

fn write_init<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} init", id).ok();
}

fn write_init_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} init", id).ok();
}

fn write_gen_move<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(output, "= {0} gen_move {1} {2} {3}", id, x, y, player.to_bool() as u32).ok();
}

fn write_gen_move_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move", id).ok();
}

fn write_gen_move_with_complexity<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(output, "= {0} gen_move_with_complexity {1} {2} {3}", id, x, y, player.to_bool() as u32).ok();
}

fn write_gen_move_with_complexity_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move_with_complexity", id).ok();
}

fn write_gen_move_with_time<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(output, "= {0} gen_move_with_time {1} {2} {3}", id, x, y, player.to_bool() as u32).ok();
}

fn write_gen_move_with_time_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move_with_time", id).ok();
}

fn write_gen_move_with_full_time<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(output, "= {0} gen_move_with_full_time {1} {2} {3}", id, x, y, player.to_bool() as u32).ok();
}

fn write_gen_move_with_full_time_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move_with_full_time", id).ok();
}

fn write_license<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} license AGPLv3+", id).ok();
}

fn write_license_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} license", id).ok();
}

fn write_list_commands<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} list_commands gen_move gen_move_with_complexity gen_move_with_time init list_commands name play quit undo version", id).ok();
}

fn write_list_commands_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} list_commands", id).ok();
}

fn write_name<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} name opai-rust", id).ok();
}

fn write_name_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} name", id).ok();
}

fn write_play<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(output, "= {0} play {1} {2} {3}", id, x, y, player.to_bool() as u32).ok();
}

fn write_play_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} play", id).ok();
}

fn write_quit<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} quit", id).ok();
}

fn write_quit_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} quit", id).ok();
}

fn write_undo<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} undo", id).ok();
}

fn write_undo_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} undo", id).ok();
}

fn write_version<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "= {0} version {1}", id, env!("CARGO_PKG_VERSION")).ok();
}

fn write_version_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} version", id).ok();
}

fn write_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} input_error", id).ok();
}

fn main() {
  log4rs::init_file(Path::new(LOG_CONFIG_PATH), Default::default()).ok();
  config::init();
  if let Ok(mut config_file) = File::open(CONFIG_PATH) {
    config::read(&mut config_file);
  } else if let Ok(mut config_file) = File::create(CONFIG_PATH) {
    config::write(&mut config_file);
  }
  let patterns = if let Ok(patterns_file) = File::open(PATTERNS_PATH) {
    Patterns::from_tar(patterns_file)
  } else {
    Patterns::empty()
  };
  let patterns_arc = Arc::new(patterns);
  let mut input = BufReader::new(io::stdin());
  let mut output = io::stdout();
  let mut bot_option = None;
  let mut s = String::new();
  loop {
    s.clear();
    input.read_line(&mut s).ok();
    s.pop();
    let mut split = s.split(' ').fuse();
    if let Some(id) = split.next().and_then(|id_str| u32::from_str(id_str).ok()) {
      match split.next() {
        Some("author") => {
          if split.next().is_some() {
            write_author_error(&mut output, id);
          } else {
            write_author(&mut output, id);
          }
        },
        Some("init") => {
          let x_option = split.next().and_then(|x_str| u32::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| u32::from_str(y_str).ok());
          let seed_option = split.next().and_then(|seed_str| u64::from_str(seed_str).ok());
          if split.next().is_some() {
            write_init_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(seed)) = (x_option, y_option, seed_option) {
            bot_option = Some(Bot::new(x, y, seed, patterns_arc.clone()));
            write_init(&mut output, id);
          } else {
            write_init_error(&mut output, id);
          }
        },
        Some("gen_move") => {
          let player_option = split.next().and_then(|player_str| u32::from_str(player_str).ok()).and_then(|player_u32| match player_u32 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          if split.next().is_some() {
            write_gen_move_error(&mut output, id);
          } else if let (Some(player), Some(bot)) = (player_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move(player) {
              write_gen_move(&mut output, id, x, y, player);
            } else {
              write_gen_move_error(&mut output, id);
            }
          } else {
            write_gen_move_error(&mut output, id);
          }
        },
        Some("gen_move_with_complexity") => {
          let player_option = split.next().and_then(|player_str| u32::from_str(player_str).ok()).and_then(|player_u32| match player_u32 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          let complexity_option = split.next().and_then(|complexity_str| u32::from_str(complexity_str).ok());
          if split.next().is_some() {
            write_gen_move_with_complexity_error(&mut output, id);
          } else if let (Some(player), Some(complexity), Some(bot)) = (player_option, complexity_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move_with_complexity(player, complexity) {
              write_gen_move_with_complexity(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_complexity_error(&mut output, id);
            }
          } else {
            write_gen_move_with_complexity_error(&mut output, id);
          }
        },
        Some("gen_move_with_time") => {
          let player_option = split.next().and_then(|player_str| u32::from_str(player_str).ok()).and_then(|player_u32| match player_u32 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          let time_option = split.next().and_then(|time_str| u32::from_str(time_str).ok());
          if split.next().is_some() {
            write_gen_move_with_time_error(&mut output, id);
          } else if let (Some(player), Some(time), Some(bot)) = (player_option, time_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move_with_time(player, time) {
              write_gen_move_with_time(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_time_error(&mut output, id);
            }
          } else {
            write_gen_move_with_time_error(&mut output, id);
          }
        },
        Some("gen_move_with_full_time") => {
          let player_option = split.next().and_then(|player_str| u32::from_str(player_str).ok()).and_then(|player_u32| match player_u32 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          let remaining_time_option = split.next().and_then(|time_str| u32::from_str(time_str).ok());
          let time_per_move_option = split.next().and_then(|time_str| u32::from_str(time_str).ok());
          if split.next().is_some() {
            write_gen_move_with_full_time_error(&mut output, id);
          } else if let (Some(player), Some(remaining_time), Some(time_per_move), Some(bot)) = (player_option, remaining_time_option, time_per_move_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move_with_full_time(player, remaining_time, time_per_move) {
              write_gen_move_with_full_time(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_full_time_error(&mut output, id);
            }
          } else {
            write_gen_move_with_full_time_error(&mut output, id);
          }
        },
        Some("license") => {
          if split.next().is_some() {
            write_license_error(&mut output, id);
          } else {
            write_license(&mut output, id);
          }
        },
        Some("list_commands") => {
          if split.next().is_some() {
            write_list_commands_error(&mut output, id);
          } else {
            write_list_commands(&mut output, id);
          }
        },
        Some("name") => {
          if split.next().is_some() {
            write_name_error(&mut output, id);
          } else {
            write_name(&mut output, id);
          }
        },
        Some("play") => {
          let x_option = split.next().and_then(|x_str| u32::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| u32::from_str(y_str).ok());
          let player_option = split.next().and_then(|player_str| u32::from_str(player_str).ok()).and_then(|player_u32| match player_u32 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          if split.next().is_some() {
            write_play_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(player), Some(bot)) = (x_option, y_option, player_option, bot_option.as_mut()) {
            if bot.put_point(x, y, player) {
              write_play(&mut output, id, x, y, player);
            } else {
              write_play_error(&mut output, id);
            }
          } else {
            write_play_error(&mut output, id);
          }
        },
        Some("quit") => {
          if split.next().is_some() {
            write_quit_error(&mut output, id);
          } else {
            write_quit(&mut output, id);
            output.flush().ok();
            break;
          }
        },
        Some("undo") => {
          if split.next().is_some() {
            write_undo_error(&mut output, id);
          } else if let Some(bot) = bot_option.as_mut() {
            if bot.undo() {
              write_undo(&mut output, id);
            } else {
              write_undo_error(&mut output, id);
            }
          } else {
            write_undo_error(&mut output, id);
          }
        },
        Some("version") => {
          if split.next().is_some() {
            write_version_error(&mut output, id);
          } else {
            write_version(&mut output, id);
          }
        },
        _ => {
          write_error(&mut output, id);
        }
      }
    } else {
      write_error(&mut output, 0);
    }
    output.flush().ok();
  }
}
