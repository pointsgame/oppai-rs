#![allow(clippy::too_many_arguments)]

#[macro_use]
extern crate log;

pub mod uct;
#[cfg(test)]
mod uct_test;
pub mod wave_pruning;
