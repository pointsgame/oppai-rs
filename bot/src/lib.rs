#[macro_use]
extern crate log;

pub mod bot;
pub mod cli;
pub mod config;
pub mod heuristic;

pub use oppai_field::{extended_field, field, player, zobrist};
pub use oppai_patterns::patterns;
