#[macro_use]
extern crate log;

pub mod bot;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
pub mod heuristic;
