#![allow(dead_code)]
#![feature(unsafe_no_drop_flag)]
#![feature(scoped)]

extern crate rand;

#[macro_use]
extern crate log;

extern crate log4rs;

mod types;
mod config;
mod player;
mod zobrist;
mod cell;
mod field;
mod uct;
mod uct_log;
mod bot;

use std::io;
use std::io::{Write, BufReader, BufRead};
use std::str::FromStr;
use std::string::ToString;
use std::path::Path;
use log4rs::toml::Creator;
use types::{Coord, Time};
use player::Player;
use bot::Bot;
use uct_log::UctLog;

fn write_author<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" author kurnevsky_evgeny\n".as_bytes()).ok();
}

fn write_author_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" author\n".as_bytes()).ok();
}

fn write_init<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" init\n".as_bytes()).ok();
}

fn write_init_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" init\n".as_bytes()).ok();
}

fn write_gen_move<T: Write>(output: &mut T, id: u32, x: Coord, y: Coord, player: Player) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move ".as_bytes()).ok();
  output.write_all(x.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all(y.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all((player.to_bool() as u8).to_string().as_bytes()).ok();
  output.write_all("\n".as_bytes()).ok();
}

fn write_gen_move_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move\n".as_bytes()).ok();
}

fn write_gen_move_with_complexity<T: Write>(output: &mut T, id: u32, x: Coord, y: Coord, player: Player) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move_with_complexity ".as_bytes()).ok();
  output.write_all(x.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all(y.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all((player.to_bool() as u8).to_string().as_bytes()).ok();
  output.write_all("\n".as_bytes()).ok();
}

fn write_gen_move_with_complexity_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move_with_complexity\n".as_bytes()).ok();
}

fn write_gen_move_with_time<T: Write>(output: &mut T, id: u32, x: Coord, y: Coord, player: Player) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move_with_time ".as_bytes()).ok();
  output.write_all(x.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all(y.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all((player.to_bool() as u8).to_string().as_bytes()).ok();
  output.write_all("\n".as_bytes()).ok();
}

fn write_gen_move_with_time_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" gen_move_with_time\n".as_bytes()).ok();
}

fn write_license<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" license GPL3\n".as_bytes()).ok();
}

fn write_license_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" license\n".as_bytes()).ok();
}

fn write_list_commands<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" list_commands gen_move gen_move_with_complexity gen_move_with_time init list_commands name play quit undo version\n".as_bytes()).ok();
}

fn write_list_commands_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" list_commands\n".as_bytes()).ok();
}

fn write_name<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" name opai-rust\n".as_bytes()).ok();
}

fn write_name_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" name\n".as_bytes()).ok();
}

fn write_play<T: Write>(output: &mut T, id: u32, x: Coord, y: Coord, player: Player) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" play ".as_bytes()).ok();
  output.write_all(x.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all(y.to_string().as_bytes()).ok();
  output.write_all(" ".as_bytes()).ok();
  output.write_all((player.to_bool() as u8).to_string().as_bytes()).ok();
  output.write_all("\n".as_bytes()).ok();
}

fn write_play_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" play\n".as_bytes()).ok();
}

fn write_quit<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" quit\n".as_bytes()).ok();
}

fn write_quit_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" quit\n".as_bytes()).ok();
}

fn write_undo<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" undo\n".as_bytes()).ok();
}

fn write_undo_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" undo\n".as_bytes()).ok();
}

fn write_version<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" version 4.0.0\n".as_bytes()).ok();
}

fn write_version_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" version\n".as_bytes()).ok();
}

fn write_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" input_error\n".as_bytes()).ok();
}

fn main() {
  log4rs::init_file(Path::new("config/log.toml"), Creator::default()).ok();
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
          let x_option = split.next().and_then(|x_str| Coord::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| Coord::from_str(y_str).ok());
          let seed_option = split.next();
          if split.next().is_some() {
            write_init_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(_)) = (x_option, y_option, seed_option) {
            bot_option = Some(Bot::new(x, y));
            write_init(&mut output, id);
          } else {
            write_init_error(&mut output, id);
          }
        },
        Some("gen_move") => {
          let player_option = split.next().and_then(|player_str| u8::from_str(player_str).ok()).and_then(|player_u8| match player_u8 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          if split.next().is_some() {
            write_gen_move_error(&mut output, id);
          } else if let (Some(player), Some(bot)) = (player_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move(player, 10000) {
              write_gen_move(&mut output, id, x, y, player);
            } else {
              write_gen_move_error(&mut output, id);
            }
          } else {
            write_gen_move_error(&mut output, id);
          }
        },
        Some("gen_move_with_complexity") => {
          let player_option = split.next().and_then(|player_str| u8::from_str(player_str).ok()).and_then(|player_u8| match player_u8 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          let complexity_option = split.next().and_then(|complexity_str| u8::from_str(complexity_str).ok() );
          if split.next().is_some() {
            write_gen_move_with_complexity_error(&mut output, id);
          } else if let (Some(player), Some(_), Some(bot)) = (player_option, complexity_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move(player, 10000) {
              write_gen_move_with_complexity(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_complexity_error(&mut output, id);
            }
          } else {
            write_gen_move_with_complexity_error(&mut output, id);
          }
        },
        Some("gen_move_with_time") => {
          let player_option = split.next().and_then(|player_str| u8::from_str(player_str).ok()).and_then(|player_u8| match player_u8 { //TODO: from_number method
            0 => Some(Player::Red),
            1 => Some(Player::Black),
            _ => None
          });
          let time_option = split.next().and_then(|time_str| Time::from_str(time_str).ok() );
          if split.next().is_some() {
            write_gen_move_with_time_error(&mut output, id);
          } else if let (Some(player), Some(time), Some(bot)) = (player_option, time_option, bot_option.as_mut()) {
            if let Some((x, y)) = bot.best_move(player, time) {
              write_gen_move_with_time(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_time_error(&mut output, id);
            }
          } else {
            write_gen_move_with_time_error(&mut output, id);
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
          let x_option = split.next().and_then(|x_str| Coord::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| Coord::from_str(y_str).ok());
          let player_option = split.next().and_then(|player_str| u8::from_str(player_str).ok()).and_then(|player_u8| match player_u8 { //TODO: from_number method
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
    let uct_str = "uct";
    if let Some(bot) = bot_option.as_mut() {
      for uct_log in bot.uct_log() {
        match uct_log {
          &UctLog::BestMove(pos, uct) => info!(target: uct_str, "Best move is {0}, uct is {1}.", pos, uct),
          &UctLog::Estimation(pos, uct, wins, draws, visits) => info!(target: uct_str, "Uct for move {0} is {1}, {2} wins, {3} draws, {4} visits.", pos, uct, wins, draws, visits)
        }
      }
      bot.clear_logs();
    }
  }
}
