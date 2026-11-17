//! The losses of training runs, averaged over each run, from the log of the
//! training.
//!
//! A run starts at the line logging its optimizer, or after a pause in the log
//! longer than a given gap, so that logs of several runs appended to one file,
//! with or without other output in between, split the same way.

use crate::common::{Table, fmt};
use anyhow::{Context, Result};
use std::{
  fs::File,
  io::{BufRead, BufReader},
  path::Path,
};

/// The components shown when none are asked for.
pub const DEFAULT_COMPONENTS: &str = "value,td value,policy,opponent policy,pdf,cdf,captured,q";

/// Seconds since the epoch of a log line's `[YYYY-MM-DDTHH:MM:SSZ` prefix.
fn timestamp(line: &str) -> Option<i64> {
  let s = line.strip_prefix('[')?;
  let num = |range: std::ops::Range<usize>| s.get(range)?.parse::<i64>().ok();
  let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
  let (hh, mm, ss) = (num(11..13)?, num(14..16)?, num(17..19)?);
  // Days from the civil date, see Howard Hinnant's `days_from_civil`.
  let y = if m <= 2 { y - 1 } else { y };
  let era = y.div_euclid(400);
  let yoe = y - era * 400;
  let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
  let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  let days = era * 146097 + doe - 719468;
  Some(days * 86400 + hh * 3600 + mm * 60 + ss)
}

/// The named numbers of a `Loss:` line: every run of words followed by a
/// number.
fn parse_components(text: &str) -> Vec<(String, f64)> {
  let mut result = Vec::new();
  let mut name: Vec<&str> = Vec::new();
  for token in text.split_whitespace() {
    match token.parse::<f64>() {
      Ok(value) if !name.is_empty() => {
        result.push((name.join(" "), value));
        name.clear();
      }
      _ => name.push(token),
    }
  }
  result
}

#[derive(Default)]
struct Run {
  start: String,
  batches: usize,
  /// The sums of the components, in the order they first appear.
  sums: Vec<(String, f64, usize)>,
  gradient_norm: (f64, usize),
}

impl Run {
  fn add(&mut self, name: &str, value: f64) {
    match self.sums.iter_mut().find(|(n, _, _)| n == name) {
      Some((_, sum, count)) => {
        *sum += value;
        *count += 1;
      }
      None => self.sums.push((name.to_string(), value, 1)),
    }
  }

  fn mean(&self, name: &str) -> Option<f64> {
    self
      .sums
      .iter()
      .find(|(n, _, _)| n == name)
      .map(|(_, sum, count)| sum / *count as f64)
  }
}

pub fn run(log: &Path, components: Option<&str>, last: usize, gap_minutes: f64) -> Result<()> {
  let file = File::open(log).with_context(|| format!("opening {}", log.display()))?;
  let mut runs: Vec<Run> = Vec::new();
  let mut last_time: Option<i64> = None;
  let mut new_run = true;
  for line in BufReader::new(file).lines() {
    let line = line?;
    let time = timestamp(&line);
    if let (Some(time), Some(last)) = (time, last_time)
      && (time - last) as f64 > gap_minutes * 60.0
    {
      new_run = true;
    }
    if time.is_some() {
      last_time = time;
    }
    if line.contains("] Optimizer ") {
      new_run = true;
      continue;
    }
    let loss = line.split_once("Loss: ").map(|(_, rest)| rest);
    let gradient = line.split_once("Gradient norm: ").map(|(_, rest)| rest);
    if loss.is_none() && gradient.is_none() {
      continue;
    }
    if new_run {
      runs.push(Run {
        start: line.get(1..20).unwrap_or("").replace('T', " "),
        ..Run::default()
      });
      new_run = false;
    }
    let run = runs.last_mut().unwrap();
    if let Some(loss) = loss {
      run.batches += 1;
      for (name, value) in parse_components(loss) {
        run.add(&name, value);
      }
    }
    if let Some(norm) = gradient.and_then(|g| g.split_whitespace().next()?.parse::<f64>().ok()) {
      run.gradient_norm.0 += norm;
      run.gradient_norm.1 += 1;
    }
  }

  let names: Vec<String> = match components {
    Some("all") => {
      runs
        .iter()
        .flat_map(|r| r.sums.iter().map(|(n, _, _)| n.clone()))
        .fold(Vec::new(), |mut names, n| {
          if !names.contains(&n) {
            names.push(n);
          }
          names
        })
    }
    Some(list) => list.split(',').map(|n| n.trim().to_string()).collect(),
    None => DEFAULT_COMPONENTS.split(',').map(str::to_string).collect(),
  };

  let mut header = vec!["run", "start", "batches"];
  header.extend(names.iter().map(String::as_str));
  header.push("gradient norm");
  let mut table = Table::new(&format!("Mean losses per training run, {}", log.display()), &header);
  let skip = runs.len().saturating_sub(last);
  for (i, run) in runs.iter().enumerate().skip(skip) {
    let mut row = vec![(i + 1).to_string(), run.start.clone(), run.batches.to_string()];
    row.extend(names.iter().map(|n| fmt(run.mean(n), 4)));
    let (sum, count) = run.gradient_norm;
    row.push(fmt((count > 0).then(|| sum / count as f64), 2));
    table.row(row);
  }
  table.print();
  Ok(())
}
