#[macro_use]
extern crate log;

pub mod bot;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
pub mod config;
pub mod heuristic;

pub use oppai_field::{field, player, zobrist};
pub use oppai_patterns::patterns;
