#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

/*!
Points AI Protocol, version 6
====

The communication between the AI and the GUI interface is done via messages to standart input of the
AI and messages from standart output of the AI. For example, if AI wants to say something, it must
send a message to his standart output. The GUI then must capture this message and analyze it. If the
GUI wants to say something, it sends a message to the standart input of the AI.

More precise, every GUI->AI message must have the form: "id command_name arguments\n", and the AI
must reply to it with one message of the form: "isOk id command_name arguments\n", where:

"isOk" is a single letter. It is "=" in case of a success and "?" in case of an error.

"id" is an int. It must be copied by the AI into his reply.

"\n" means "end of line".

Commands
====

* list_commands

  return arguments - a space separated list of commands that the AI supports (can accept).

* quit - request for the AI to exit.

  return arguments - none.

* init width height random_seed - initialization.

  random_seed - seed for random number generator, useful for reproducing games.

  return arguments - none.

* author

  return arguments - author of the AI.

* name

  return arguments - name of the AI.

* version

  return arguments - version of the AI.

* license

  return arguments - license of the AI.

* play x y color - play a move on the field.

  return arguments - x, y, color  of a played move.

* gen_move color - request to calculate an AI move, but do NOT make it on the field.

  return arguments - x, y, color of the suggested move.

* gen_move_with_complexity color complexity - request to calculate an AI move with the given
  complexity, but NOT to make it on the field.

  complexity - a number from 0 to 100. The interpretation of this number lies on the AI.

  return arguments - x, y, color of the suggested move.

* gen_move_with_time color time - request to calculate an AI move within the given time
  (milliseconds), but NOT to make it on the field.

  return arguments - x, y, color of the suggested move.

* gen_move_with_full_time color remaining_time time_per_move - request to calculate an AI move with
  full time (milliseconds) control at the side of AI, but NOT to make it on the field.

  return arguments - x, y, color of the suggested move.

* undo - undo move.
  return arguments - none.


Explanations
====

The coordinate "x" is a number from 0 to fieldSizeX - 1. Same goes for "y".

"Color" is a boolean value serialized as "0" or "1".

Error messages should not contain return arguments.

If the argument string is returned, it should not contain spaces.


Example
====

Initialize the field:

```
init 3 3 0
= 0 init
```

Place a point in the center, with color "0":

```
1 play 1 1 0
= 1 play 1 1 0
```

Surround the point with 3 opponent points (color "1"):

```
2 play 0 1 1
= 2 play 0 1 1
3 play 1 0 1
= 3 play 1 0 1
4 play 2 1 1
= 4 play 2 1 1
```

Ask the AI to generate a move.

```
5 gen_move_with_time 1 1000
```

The AI should "think" about this command for no more than 1000 milliseconds. If the AI is smart
enough, it will answer with:

```
= 5 gen_move_with_time 1 2 1
```

Thereby asking to surround the central point. If we allow it, we must separately place the point on
the field:

```
6 play 1 2 1
= 6 play 1 2 1
```

That's it, the central point is now surrounded. We initialized the field, placed 4 dots, asked the
computer to generate a move and placed the generated point on the field.
*/

#[macro_use]
extern crate clap;

mod config;

use crate::config::cli_parse;
use oppai_bot::bot::Bot;
use oppai_field::player::Player;
use oppai_patterns::patterns::Patterns;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::{
  default::Default,
  fs::File,
  io::{self, BufRead, BufReader, Write},
  str::FromStr,
  sync::Arc,
};

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
  writeln!(
    output,
    "= {0} gen_move_with_complexity {1} {2} {3}",
    id,
    x,
    y,
    player.to_bool() as u32
  )
  .ok();
}

fn write_gen_move_with_complexity_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move_with_complexity", id).ok();
}

fn write_gen_move_with_time<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(
    output,
    "= {0} gen_move_with_time {1} {2} {3}",
    id,
    x,
    y,
    player.to_bool() as u32
  )
  .ok();
}

fn write_gen_move_with_time_error<T: Write>(output: &mut T, id: u32) {
  writeln!(output, "? {0} gen_move_with_time", id).ok();
}

fn write_gen_move_with_full_time<T: Write>(output: &mut T, id: u32, x: u32, y: u32, player: Player) {
  writeln!(
    output,
    "= {0} gen_move_with_full_time {1} {2} {3}",
    id,
    x,
    y,
    player.to_bool() as u32
  )
  .ok();
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
  let config = cli_parse();
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();
  let patterns = if config.patterns.is_empty() {
    Patterns::empty()
  } else {
    Patterns::from_files(
      config
        .patterns
        .iter()
        .map(|path| File::open(path).expect("Failed to open patterns file.")),
    )
    .expect("Failed to read patterns file.")
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
        }
        Some("init") => {
          let x_option = split.next().and_then(|x_str| u32::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| u32::from_str(y_str).ok());
          let seed_option = split.next().and_then(|seed_str| u64::from_str(seed_str).ok());
          if split.next().is_some() {
            write_init_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(seed)) = (x_option, y_option, seed_option) {
            let rng = SmallRng::seed_from_u64(seed);
            bot_option = Some(Bot::new(x, y, rng, Arc::clone(&patterns_arc), config.bot.clone()));
            write_init(&mut output, id);
          } else {
            write_init_error(&mut output, id);
          }
        }
        Some("gen_move") => {
          let player_option = split
            .next()
            .and_then(|player_str| u32::from_str(player_str).ok())
            .and_then(|player_u32| match player_u32 {
              // TODO: from_number method
              0 => Some(Player::Red),
              1 => Some(Player::Black),
              _ => None,
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
        }
        Some("gen_move_with_complexity") => {
          let player_option = split
            .next()
            .and_then(|player_str| u32::from_str(player_str).ok())
            .and_then(|player_u32| match player_u32 {
              // TODO: from_number method
              0 => Some(Player::Red),
              1 => Some(Player::Black),
              _ => None,
            });
          let complexity_option = split
            .next()
            .and_then(|complexity_str| u32::from_str(complexity_str).ok());
          if split.next().is_some() {
            write_gen_move_with_complexity_error(&mut output, id);
          } else if let (Some(player), Some(complexity), Some(bot)) =
            (player_option, complexity_option, bot_option.as_mut())
          {
            if let Some((x, y)) = bot.best_move_with_complexity(player, complexity) {
              write_gen_move_with_complexity(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_complexity_error(&mut output, id);
            }
          } else {
            write_gen_move_with_complexity_error(&mut output, id);
          }
        }
        Some("gen_move_with_time") => {
          let player_option = split
            .next()
            .and_then(|player_str| u32::from_str(player_str).ok())
            .and_then(|player_u32| match player_u32 {
              // TODO: from_number method
              0 => Some(Player::Red),
              1 => Some(Player::Black),
              _ => None,
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
        }
        Some("gen_move_with_full_time") => {
          let player_option = split
            .next()
            .and_then(|player_str| u32::from_str(player_str).ok())
            .and_then(|player_u32| match player_u32 {
              // TODO: from_number method
              0 => Some(Player::Red),
              1 => Some(Player::Black),
              _ => None,
            });
          let remaining_time_option = split.next().and_then(|time_str| u32::from_str(time_str).ok());
          let time_per_move_option = split.next().and_then(|time_str| u32::from_str(time_str).ok());
          if split.next().is_some() {
            write_gen_move_with_full_time_error(&mut output, id);
          } else if let (Some(player), Some(remaining_time), Some(time_per_move), Some(bot)) = (
            player_option,
            remaining_time_option,
            time_per_move_option,
            bot_option.as_mut(),
          ) {
            if let Some((x, y)) = bot.best_move_with_full_time(player, remaining_time, time_per_move) {
              write_gen_move_with_full_time(&mut output, id, x, y, player);
            } else {
              write_gen_move_with_full_time_error(&mut output, id);
            }
          } else {
            write_gen_move_with_full_time_error(&mut output, id);
          }
        }
        Some("license") => {
          if split.next().is_some() {
            write_license_error(&mut output, id);
          } else {
            write_license(&mut output, id);
          }
        }
        Some("list_commands") => {
          if split.next().is_some() {
            write_list_commands_error(&mut output, id);
          } else {
            write_list_commands(&mut output, id);
          }
        }
        Some("name") => {
          if split.next().is_some() {
            write_name_error(&mut output, id);
          } else {
            write_name(&mut output, id);
          }
        }
        Some("play") => {
          let x_option = split.next().and_then(|x_str| u32::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| u32::from_str(y_str).ok());
          let player_option = split
            .next()
            .and_then(|player_str| u32::from_str(player_str).ok())
            .and_then(|player_u32| match player_u32 {
              // TODO: from_number method
              0 => Some(Player::Red),
              1 => Some(Player::Black),
              _ => None,
            });
          if split.next().is_some() {
            write_play_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(player), Some(bot)) =
            (x_option, y_option, player_option, bot_option.as_mut())
          {
            if bot.put_point(x, y, player) {
              write_play(&mut output, id, x, y, player);
            } else {
              write_play_error(&mut output, id);
            }
          } else {
            write_play_error(&mut output, id);
          }
        }
        Some("quit") => {
          if split.next().is_some() {
            write_quit_error(&mut output, id);
          } else {
            write_quit(&mut output, id);
            output.flush().ok();
            break;
          }
        }
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
        }
        Some("version") => {
          if split.next().is_some() {
            write_version_error(&mut output, id);
          } else {
            write_version(&mut output, id);
          }
        }
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
