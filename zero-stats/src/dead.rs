//! Output channels of the networks' layers that have died or are about to,
//! read straight from the model files.
//!
//! Muon gives a channel whose gradient is zero no update at all, so plain
//! weight decay can shrink it to nothing; once the weights reading it have
//! followed it there, neither side gets a gradient again and it is lost for
//! good. This report shows how many channels are dead, and how many are weak
//! - small enough to be on their way there, or just little used.
//!
//! A channel counts as dead when its norm is at most [`DEAD_FRACTION`] of the
//! tensor's reference norm, the median of its live channels' norms, see
//! [`reference_norm`]. A dead channel of the trunk can still feed the next
//! layer a constant through the offset of the normalization after it, so it
//! is dead in the sense that matters for Muon - its own weights get no
//! gradient - rather than necessarily useless.

use crate::common::{Table, label, percent};
use crate::weights::values;
use anyhow::{Context, Result};
use burn_pack::Reader;
use std::path::PathBuf;

/// The fraction of the reference norm of a tensor at or below which a channel
/// is dead.
const DEAD_FRACTION: f64 = 0.01;

/// The tensors with dead channels listed per model, the most dead first.
const LISTED_TENSORS: usize = 10;

/// The norm of every output channel of a layer's weights: the first dimension
/// of a convolution's `[out, in, kh, kw]`, and the second of a linear layer's
/// `[in, out]`. `None` for any other tensor.
fn channel_norms(values: &[f32], shape: &[usize]) -> Option<Vec<f64>> {
  let (channels, inner, stride) = match shape.len() {
    4 => (shape[0], shape[1..].iter().product::<usize>(), 1),
    2 => (shape[1], shape[0], shape[1]),
    _ => return None,
  };
  let mut sums = vec![0.0f64; channels];
  for (i, &v) in values.iter().enumerate() {
    let channel = if stride == 1 { i / inner } else { i % stride };
    sums[channel] += f64::from(v) * f64::from(v);
  }
  Some(sums.into_iter().map(f64::sqrt).collect())
}

/// The typical norm of the live channels of a tensor: the median over the
/// channels above a thousandth of the largest, so that a tensor most of whose
/// channels are dead still gets a reference from the live ones. 0 when every
/// channel is about 0.
fn reference_norm(norms: &[f64]) -> f64 {
  let largest = norms.iter().copied().fold(0.0, f64::max);
  let mut live: Vec<f64> = norms.iter().copied().filter(|&n| n > 1e-3 * largest).collect();
  live.sort_by(f64::total_cmp);
  live.get(live.len() / 2).copied().unwrap_or(0.0)
}

/// The norm at or below which a channel of a tensor is weak: `weak` times the
/// root mean square of the channel norms.
fn weak_norm(norms: &[f64], weak: f64) -> f64 {
  weak * (norms.iter().map(|n| n * n).sum::<f64>() / norms.len() as f64).sqrt()
}

#[derive(Default)]
struct Counts {
  channels: usize,
  dead: usize,
  weak: usize,
  /// Tensors whose every weight is 0, see [`run`].
  zero_tensors: usize,
}

pub fn run(models: &[PathBuf], weak: f64) -> Result<()> {
  let mut summary = Table::new(
    &format!(
      "Output channels at most {DEAD_FRACTION} of their tensor's median live norm (dead), and at or below {weak} of \
       its root mean square norm (weak); the trunk is what Muon trains. Tensors all of zeros are counted \
       apart: Fixup starts the last layer of every residual branch at zero"
    ),
    &[
      "model",
      "trunk channels",
      "trunk dead",
      "trunk weak",
      "trunk zero tensors",
      "other channels",
      "other dead",
    ],
  );
  let mut listed = Vec::new();
  for path in models {
    let tensors = Reader::from_file(path)
      .and_then(|reader| reader.into_tensors())
      .with_context(|| format!("reading {}", path.display()))?;
    let mut trunk = Counts::default();
    let mut other = Counts::default();
    let mut with_dead = Vec::new();
    for tensor in &tensors {
      // Only weights of layers: other parameters can have the same shapes.
      if !tensor.name.ends_with("weight") {
        continue;
      }
      let shape: Vec<usize> = tensor.shape.iter().copied().collect();
      let Some(norms) = channel_norms(&values(tensor)?, &shape) else {
        continue;
      };
      if norms.len() < 2 {
        continue;
      }
      let counts = if tensor.name.starts_with("residuals.") {
        &mut trunk
      } else {
        &mut other
      };
      let reference = reference_norm(&norms);
      // A tensor all of zeros has no live channels to tell dead ones from: it
      // is a layer fresh from Fixup's initialization, or a broken one.
      if reference == 0.0 {
        counts.zero_tensors += 1;
        continue;
      }
      let dead = norms.iter().filter(|&&n| n <= DEAD_FRACTION * reference).count();
      let weak_norm = weak_norm(&norms, weak);
      let weak = norms.iter().filter(|&&n| n <= weak_norm).count();
      counts.channels += norms.len();
      counts.dead += dead;
      counts.weak += weak;
      if dead > 0 {
        with_dead.push((dead, norms.len(), tensor.name.clone()));
      }
    }
    summary.row(vec![
      label(path),
      trunk.channels.to_string(),
      percent(trunk.dead, trunk.channels),
      percent(trunk.weak, trunk.channels),
      trunk.zero_tensors.to_string(),
      other.channels.to_string(),
      percent(other.dead, other.channels),
    ]);
    with_dead.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    with_dead.truncate(LISTED_TENSORS);
    listed.push((label(path), with_dead));
  }
  summary.print();
  for (model, with_dead) in listed {
    if with_dead.is_empty() {
      continue;
    }
    let mut table = Table::new(
      &format!("{model}: tensors with the most dead channels"),
      &["tensor", "dead", "channels"],
    );
    for (dead, channels, name) in with_dead {
      table.row(vec![name, dead.to_string(), channels.to_string()]);
    }
    table.print();
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::{channel_norms, reference_norm, weak_norm};

  #[test]
  fn channels_are_read_along_the_output_dimension() {
    // A convolution [2, 1, 1, 2]: the channels are the first dimension.
    let norms = channel_norms(&[3.0, 4.0, 0.0, 1.0], &[2, 1, 1, 2]).unwrap();
    assert_eq!(norms, vec![5.0, 1.0]);
    // A linear layer [2, 3]: the channels are the second dimension.
    let norms = channel_norms(&[3.0, 0.0, 1.0, 4.0, 0.0, 0.0], &[2, 3]).unwrap();
    assert_eq!(norms, vec![5.0, 0.0, 1.0]);
    assert!(channel_norms(&[1.0], &[1]).is_none());
  }

  #[test]
  fn the_reference_ignores_dead_channels() {
    assert_eq!(reference_norm(&[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 2.0);
    assert_eq!(reference_norm(&[0.0, 0.0]), 0.0);
    assert_eq!(weak_norm(&[2.0, 2.0], 0.1), 0.2);
  }
}
