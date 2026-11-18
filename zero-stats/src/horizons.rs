//! How far ahead play stays connected to a position, by board size: the
//! numbers to pick the horizons of the training targets from.
//!
//! Two measures, both as functions of the lookahead `k` in moves:
//!
//! - value: the root mean square change of the search value over `k` moves,
//!   with the game's result standing in for the values past its end, the way
//!   the TD targets blend it. Where it rises steeply and then flattens is how
//!   long the fights take to resolve;
//! - locality: how much more often than a random empty cell the `k`-th move
//!   after a position lands near the last moves before it. Where that excess
//!   fades is where play leaves the current fight.
//!
//! Each is summed up by the lookahead at which it gets to a given part of the
//! way, and that lookahead divided by the side and by the area of the board:
//! whichever of the three stays the same across board sizes says how the
//! horizon should scale.

use crate::common::{Game, Table, fmt, label, mean, read_games};
use anyhow::Result;
use oppai_field::player::Player;
use std::{collections::BTreeMap, path::PathBuf};

/// The lookaheads shown, in moves.
const LAGS: [usize; 16] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64];

pub struct HorizonsParams {
  pub files: Vec<PathBuf>,
  pub count: Option<usize>,
  /// The width of the board size buckets, in the square root of the area.
  pub bucket: u32,
  /// How many moves before a position make its recent play.
  pub recent: usize,
  /// The Chebyshev distance from the recent play within which a move is near.
  pub radius: u32,
}

/// The sums of one board size bucket.
#[derive(Default)]
struct Bucket {
  games: usize,
  sides: Vec<f64>,
  lengths: Vec<f64>,
  /// For each of [`LAGS`]: the sum of the squared value changes and their
  /// count.
  value: [(f64, usize); LAGS.len()],
  /// The squared change of the value to the result: the change at an
  /// infinite lookahead.
  value_to_result: (f64, usize),
  /// For each of [`LAGS`]: the moves near the recent play, the expected
  /// number of them for random empty cells, and the count of moves.
  locality: [(f64, f64, usize); LAGS.len()],
}

impl Bucket {
  fn add(&mut self, game: &Game, params: &HorizonsParams) {
    let field = game.field();
    let area = (game.width * game.height) as usize;
    self.games += 1;
    self.sides.push((area as f64).sqrt());
    self.lengths.push(game.moves.len() as f64);

    // Value changes. The values belong to the last moves of the game, the
    // ones before them being the initial position.
    let initial = game.moves.len() - game.red_values.len();
    let result = (game.final_field().score(Player::Red) * 2 + game.komi_x_2).signum() as f64;
    let values = &game.red_values;
    let value_at = |i: usize| values.get(i).copied().unwrap_or(result);
    for (i, &value) in values.iter().enumerate() {
      for (lag, sums) in LAGS.iter().zip(self.value.iter_mut()) {
        let change = value_at(i + lag) - value;
        sums.0 += change * change;
        sums.1 += 1;
      }
      let change = result - value;
      self.value_to_result.0 += change * change;
      self.value_to_result.1 += 1;
    }

    // Locality. Dots never leave the board, so a cell is empty at a move if
    // its dot comes later or never; cells captured empty are counted as empty,
    // a small overcount of where a random move could go.
    let coords: Vec<(i64, i64)> = game
      .moves
      .iter()
      .map(|&(pos, _)| (field.to_x(pos) as i64, field.to_y(pos) as i64))
      .collect();
    let mut fill = vec![usize::MAX; field.length()];
    for (i, &(pos, _)) in game.moves.iter().enumerate() {
      fill[pos] = i;
    }
    let radius = params.radius as i64;
    let (width, height) = (game.width as i64, game.height as i64);
    for t in initial.max(params.recent)..game.moves.len() {
      let recent = &coords[t - params.recent..t];
      let near = |(x, y): (i64, i64)| {
        recent
          .iter()
          .any(|&(rx, ry)| (x - rx).abs() <= radius && (y - ry).abs() <= radius)
      };
      // The cells near the recent play and when each of them fills.
      let mut near_fills = Vec::new();
      for y in 0..height {
        for x in 0..width {
          if near((x, y)) {
            near_fills.push(fill[field.to_pos(x as u32, y as u32)]);
          }
        }
      }
      for (lag, sums) in LAGS.iter().zip(self.locality.iter_mut()) {
        let m = t + lag - 1;
        if m >= game.moves.len() {
          continue;
        }
        let empty = area - m;
        let near_empty = near_fills.iter().filter(|&&f| f >= m).count();
        sums.0 += f64::from(near(coords[m]) as u8);
        sums.1 += near_empty as f64 / empty as f64;
        sums.2 += 1;
      }
    }
  }

  fn value_rms(&self) -> Vec<f64> {
    self.value.iter().map(|&(sum, n)| (sum / n as f64).sqrt()).collect()
  }

  fn value_to_result_rms(&self) -> f64 {
    (self.value_to_result.0 / self.value_to_result.1 as f64).sqrt()
  }

  /// How much more often than a random empty cell the move at each lag is
  /// near the recent play, or `None` for a lag with no moves.
  fn locality_excess(&self) -> Vec<Option<f64>> {
    self
      .locality
      .iter()
      .map(|&(near, expected, n)| (n > 0).then(|| (near - expected) / n as f64))
      .collect()
  }
}

/// The lookahead, interpolated linearly between [`LAGS`], at which `curve`
/// first rises to `target`, or `None` if it never does.
fn rises_to(curve: &[f64], target: f64) -> Option<f64> {
  let mut previous: Option<(f64, f64)> = None;
  for (&lag, &value) in LAGS.iter().zip(curve) {
    let lag = lag as f64;
    if value >= target {
      return Some(match previous {
        Some((p_lag, p_value)) if value > p_value => p_lag + (target - p_value) / (value - p_value) * (lag - p_lag),
        _ => lag,
      });
    }
    previous = Some((lag, value));
  }
  None
}

/// The lookahead at which `curve` first falls to `target`.
fn falls_to(curve: &[f64], target: f64) -> Option<f64> {
  rises_to(&curve.iter().map(|v| -v).collect::<Vec<_>>(), -target)
}

pub fn run(params: HorizonsParams) -> Result<()> {
  let mut buckets: BTreeMap<u32, Bucket> = BTreeMap::new();
  let mut labels = Vec::new();
  for path in &params.files {
    labels.push(label(path));
    for game in read_games(path, params.count)? {
      let side = f64::from(game.width * game.height).sqrt();
      let key = side as u32 / params.bucket * params.bucket;
      buckets.entry(key).or_default().add(&game, &params);
    }
  }
  let name = |key: u32| format!("side {}-{}", key, key + params.bucket - 1);
  let lag_names: Vec<String> = LAGS.iter().map(|lag| lag.to_string()).collect();
  let header = |first: &'static str| {
    let mut header = vec![first];
    header.extend(lag_names.iter().map(String::as_str));
    header
  };
  println!("Games: {}\n", labels.join(", "));

  let mut table = Table::new(
    "Root mean square change of the search value over k moves (Red's view; the result past the end)",
    &[header("k"), vec!["to result"]].concat(),
  );
  for (&key, bucket) in &buckets {
    let mut row = vec![name(key)];
    row.extend(bucket.value_rms().into_iter().map(|v| fmt(Some(v), 3)));
    row.push(fmt(Some(bucket.value_to_result_rms()), 3));
    table.row(row);
  }
  table.print();

  let mut table = Table::new(
    &format!(
      "Share of the k-th move landing within {} of the last {} moves, above a random empty cell",
      params.radius, params.recent
    ),
    &header("k"),
  );
  for (&key, bucket) in &buckets {
    let mut row = vec![name(key)];
    row.extend(
      bucket
        .locality_excess()
        .into_iter()
        .map(|v| fmt(v.map(|v| 100.0 * v), 1)),
    );
    table.row(row);
  }
  table.print();

  let mut table = Table::new(
    "Lookaheads in moves, and divided by the side and by the area (per cent): the one that stays the same scales",
    &[
      "bucket",
      "games",
      "mean side",
      "mean length",
      "value 25%",
      "/side",
      "/area%",
      "value 50%",
      "/side",
      "/area%",
      "locality half",
      "/side",
      "/area%",
    ],
  );
  for (&key, bucket) in &buckets {
    let side = mean(&bucket.sides).unwrap();
    let to_result = bucket.value_to_result_rms();
    let value = bucket.value_rms();
    let excess = bucket.locality_excess();
    let excess: Vec<f64> = excess.iter().map_while(|v| *v).collect();
    let locality = excess.first().and_then(|&first| falls_to(&excess, first / 2.0));
    let mut row = vec![
      name(key),
      bucket.games.to_string(),
      fmt(Some(side), 1),
      fmt(mean(&bucket.lengths), 0),
    ];
    for k in [
      rises_to(&value, to_result * 0.25),
      rises_to(&value, to_result * 0.5),
      locality,
    ] {
      row.push(fmt(k, 1));
      row.push(fmt(k.map(|k| k / side), 3));
      row.push(fmt(k.map(|k| 100.0 * k / (side * side)), 2));
    }
    table.row(row);
  }
  table.print();
  Ok(())
}
