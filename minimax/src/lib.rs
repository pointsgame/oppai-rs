#![allow(clippy::too_many_arguments)]

#[macro_use]
extern crate log;

pub mod hash_table;
#[cfg(test)]
mod hash_table_test;
pub mod minimax;
#[cfg(test)]
mod minimax_test;
pub mod trajectories_pruning;
