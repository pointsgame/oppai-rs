#![allow(dead_code)]
#![feature(unsafe_no_drop_flag)]
#![feature(scoped)]

extern crate rand;

mod types;
mod config;
mod player;
mod zobrist;
mod cell;
mod field;
mod uct;
mod bot;

use std::io;
use std::io::{Read, Write};
use std::str::FromStr;
use std::string::ToString;
use types::Coord;
use bot::Bot;

fn write_author<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" author kurnevsky_evgeny\n".as_bytes()).ok();
}

fn write_author_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? \n".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" author\n".as_bytes()).ok();
}

fn write_init<T: Write>(output: &mut T, id: u32) {
  output.write_all("= ".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" init\n".as_bytes()).ok();
}

fn write_init_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? \n".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" init\n".as_bytes()).ok();
}

fn write_error<T: Write>(output: &mut T, id: u32) {
  output.write_all("? \n".as_bytes()).ok();
  output.write_all(id.to_string().as_bytes()).ok();
  output.write_all(" input_error\n".as_bytes()).ok();
}

fn main() {
  let mut input = io::stdin();
  let mut output = io::stdout();
  let mut bot = None;
  let mut s = String::new();
  loop {
    input.read_to_string(&mut s).ok();
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
          let y_option = split.next().and_then(|y_str| Coord::from_str(y_str).ok());;
          let seed_option = split.next();
          if split.next().is_some() {
            write_init_error(&mut output, id);
          } else if let (Some(x), Some(y), Some(seed)) = (x_option, y_option, seed_option) {
            bot = Some(Bot::new(x, y));
            write_init(&mut output, id);
          } else {
            write_init_error(&mut output, id);
          }
        },
        _ => {
          write_error(&mut output, id);
        }
      }
    } else {
      write_error(&mut output, 0);
    }
  }
}
