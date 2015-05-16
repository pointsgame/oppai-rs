#![allow(dead_code)]
#![feature(unsafe_no_drop_flag)]

extern crate rand;

mod types;
mod config;
mod player;
mod zobrist;
mod cell;
mod field;
mod uct;

fn main() {
  println!("Hello, world!");
}
