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
use std::io::Read;
use std::str::FromStr;
use types::Coord;
use bot::Bot;

fn main() {
  let mut input = io::stdin();
  let mut bot = None;
  let mut s = String::new();
  loop {
    input.read_to_string(&mut s);
    let mut split = s.split(' ').fuse();
    if let Some(id) = split.next() {
      match split.next() {
        Some("init") => {
          let x_option = split.next().and_then(|x_str| Coord::from_str(x_str).ok());
          let y_option = split.next().and_then(|y_str| Coord::from_str(y_str).ok());;
          let seed_option = split.next();
          if split.next().is_some() {
            
          } else if let (Some(x), Some(y), Some(seed)) = (x_option, y_option, seed_option) {
            bot = Some(Bot::new(x, y));
          } else {
            
          }
        },
        _ => {}
      }
    } else {
      
    }
  }
}
