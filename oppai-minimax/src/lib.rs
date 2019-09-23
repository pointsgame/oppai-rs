#[macro_use]
extern crate log;

pub mod minimax;
pub mod trajectories_pruning;
pub mod hash_table;
#[cfg(test)]
mod hash_table_test;
#[cfg(test)]
mod minimax_test;
