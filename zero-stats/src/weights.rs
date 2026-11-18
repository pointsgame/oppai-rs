//! The size of the weights each input of a network feeds, read straight from
//! its file: an input whose weights stay at or near zero is one the network
//! ignores. Inputs added to a trained network start at zero, so this shows how
//! far it has picked them up.
//!
//! The weights are found by shape rather than by the names of layers, so that
//! any architecture is covered: every weight of a layer whose input dimension
//! is the number of board planes, of global features or of the score belief's
//! per-bin inputs.

use crate::common::{Table, fmt, label, mean};
use anyhow::{Context, Result, bail};
use burn::tensor::{bf16, f16};
use burn_pack::{DType, Reader};
use oppai_zero::{
  field_features::{CHANNELS, GLOBAL_FEATURES, HISTORY_CHANNELS},
  tactics::TACTICAL_CHANNELS,
};
use oppai_zero_burn::model::SCORE_BELIEF_OFFSETS;
use std::{collections::BTreeMap, path::PathBuf};

/// The names of the global features, see `global_to_vec`.
const GLOBAL_NAMES: [&str; GLOBAL_FEATURES] = ["lead", "whole komi", "komi", "shortfall", "opponent shortfall"];

/// The names of the score belief's per-bin inputs, see `score_belief_offsets`.
const OFFSET_NAMES: [&str; SCORE_BELIEF_OFFSETS] = ["bin score", "level"];

/// The board planes in groups, as ranges of channels, see
/// `field_features_to_vec`.
fn plane_groups() -> Vec<(String, std::ops::Range<usize>)> {
  let board = CHANNELS - HISTORY_CHANNELS - TACTICAL_CHANNELS;
  let mut groups = vec![
    (format!("board 0-{}", board - 1), 0..board),
    (
      format!("history {}-{}", board, board + HISTORY_CHANNELS - 1),
      board..board + HISTORY_CHANNELS,
    ),
  ];
  for i in 0..TACTICAL_CHANNELS {
    let c = board + HISTORY_CHANNELS + i;
    groups.push((format!("tactics {c}"), c..c + 1));
  }
  groups
}

fn values(tensor: &burn_pack::Tensor) -> Result<Vec<f32>> {
  let bytes = tensor.to_bytes()?;
  Ok(match tensor.dtype {
    DType::F32 => bytes
      .as_chunks::<4>()
      .0
      .iter()
      .map(|&c| f32::from_le_bytes(c))
      .collect(),
    DType::F16 => bytes
      .as_chunks::<2>()
      .0
      .iter()
      .map(|&c| f16::from_le_bytes(c).to_f32())
      .collect(),
    DType::BF16 => bytes
      .as_chunks::<2>()
      .0
      .iter()
      .map(|&c| bf16::from_le_bytes(c).to_f32())
      .collect(),
    dtype => bail!("tensor {} has unsupported type {dtype:?}", tensor.name),
  })
}

/// The L2 norm of the weights of every input of a tensor, the input dimension
/// being `input_dim` of `shape`.
fn input_norms(values: &[f32], shape: &[usize], input_dim: usize) -> Vec<f64> {
  let inputs = shape[input_dim];
  let inner: usize = shape[input_dim + 1..].iter().product();
  let mut sums = vec![0.0f64; inputs];
  for (i, &v) in values.iter().enumerate() {
    sums[(i / inner) % inputs] += (v as f64) * (v as f64);
  }
  sums.into_iter().map(f64::sqrt).collect()
}

pub fn run(models: &[PathBuf], per_channel: bool) -> Result<()> {
  // Per tensor name, the rows of its table.
  let mut tables: BTreeMap<String, Table> = BTreeMap::new();
  for path in models {
    let tensors = Reader::from_file(path)
      .and_then(|reader| reader.into_tensors())
      .with_context(|| format!("reading {}", path.display()))?;
    let mut found = false;
    for tensor in &tensors {
      let shape: Vec<usize> = tensor.shape.iter().copied().collect();
      // Only weights of layers: other parameters, like those of attention, can
      // have the same shapes. Convolutions are [out, in, kh, kw]; linear
      // layers [in, out].
      if !tensor.name.ends_with("weight") {
        continue;
      }
      let (input_dim, names): (usize, Vec<String>) = match shape.len() {
        4 if shape[1] == CHANNELS => (1, Vec::new()),
        2 if shape[0] == GLOBAL_FEATURES => (0, GLOBAL_NAMES.iter().map(|n| n.to_string()).collect()),
        2 if shape[0] == SCORE_BELIEF_OFFSETS => (0, OFFSET_NAMES.iter().map(|n| n.to_string()).collect()),
        _ => continue,
      };
      found = true;
      let norms = input_norms(&values(tensor)?, &shape, input_dim);
      let (header, row): (Vec<String>, Vec<String>) = if input_dim == 1 {
        let mut header = Vec::new();
        let mut row = Vec::new();
        for (name, range) in plane_groups() {
          header.push(name);
          row.push(fmt(mean(&norms[range]), 3));
        }
        if per_channel {
          for (c, norm) in norms.iter().enumerate() {
            header.push(c.to_string());
            row.push(format!("{norm:.2}"));
          }
        }
        (header, row)
      } else {
        (names, norms.iter().map(|n| format!("{n:.3}")).collect())
      };
      let title = if input_dim == 1 {
        format!("{} (L2 norm per input plane, mean over each group)", tensor.name)
      } else {
        format!("{} (L2 norm per input)", tensor.name)
      };
      let table = tables.entry(tensor.name.clone()).or_insert_with(|| {
        let mut columns = vec!["model".to_string()];
        columns.extend(header);
        Table::new(&title, &columns.iter().map(String::as_str).collect::<Vec<_>>())
      });
      let mut cells = vec![label(path)];
      cells.extend(row);
      table.row(cells);
    }
    if !found {
      eprintln!("{}: no tensor takes the inputs of the current features", path.display());
    }
  }
  for table in tables.values() {
    table.print();
  }
  Ok(())
}
