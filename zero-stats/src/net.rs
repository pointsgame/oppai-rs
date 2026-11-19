//! What networks predict for the positions of recorded games: how well the
//! value, score and policy match what happened, the probability of a draw the
//! draw term of the search reads, and, with inputs blanked, how much the
//! networks rely on each group of inputs.

use crate::common::{Game, Table, fmt, label, mean, median, percent, quantile};
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array4, s};
use oppai_field::player::Player;
use oppai_zero::{
  field_features::{
    CHANNELS, GLOBAL_FEATURES, GLOBAL_KOMI, GLOBAL_SHORTFALL, HISTORY_CHANNELS, field_features_to_vec, global_to_vec,
  },
  mcgs::{draw_phase, settled},
  model::Model,
  tactics::{TACTICAL_CHANNELS, Tactics},
};
use oppai_zero_burn::model::{Model as BurnModel, ModelConfig, Predictor};
use std::{ops::Range, path::PathBuf};

/// An input group to blank, see [`Blank::parse`].
#[derive(Clone, Copy, PartialEq)]
pub enum Blank {
  Tactics,
  History,
  Komi,
  Shortfall,
}

impl Blank {
  pub const NAMES: &str = "tactics, history, komi, shortfall";

  pub fn parse(name: &str) -> Result<Self> {
    match name.trim() {
      "tactics" => Ok(Blank::Tactics),
      "history" => Ok(Blank::History),
      "komi" => Ok(Blank::Komi),
      "shortfall" => Ok(Blank::Shortfall),
      other => Err(anyhow!("unknown input group {other}, expected one of {}", Self::NAMES)),
    }
  }

  fn name(self) -> &'static str {
    match self {
      Blank::Tactics => "tactics",
      Blank::History => "history",
      Blank::Komi => "komi",
      Blank::Shortfall => "shortfall",
    }
  }

  /// The board planes the group covers, see `field_features_to_vec`.
  fn planes(self) -> Range<usize> {
    match self {
      Blank::Tactics => CHANNELS - TACTICAL_CHANNELS..CHANNELS,
      Blank::History => CHANNELS - TACTICAL_CHANNELS - HISTORY_CHANNELS..CHANNELS - TACTICAL_CHANNELS,
      Blank::Komi | Blank::Shortfall => 0..0,
    }
  }

  /// The global features the group covers: the komi features, those after
  /// the lead up to the komi itself, or the two shortfalls, see
  /// `global_to_vec`.
  fn globals(self) -> Range<usize> {
    match self {
      Blank::Komi => 1..GLOBAL_KOMI + 1,
      Blank::Shortfall => GLOBAL_SHORTFALL..GLOBAL_SHORTFALL + 2,
      _ => 0..0,
    }
  }

  /// Blanks the inputs of the group: zero is what a network saw for an input
  /// before it existed.
  fn apply(self, inputs: &mut Array4<f32>, global: &mut Array2<f32>) {
    let planes = self.planes();
    if !planes.is_empty() {
      inputs.slice_mut(s![.., planes, .., ..]).fill(0.0);
    }
    let globals = self.globals();
    if !globals.is_empty() {
      global.slice_mut(s![.., globals]).fill(0.0);
    }
  }
}

pub struct NetParams {
  pub device: String,
  pub config: PathBuf,
  pub models: Vec<PathBuf>,
  pub games: Vec<Game>,
  pub games_label: String,
  pub from: usize,
  pub stride: usize,
  pub blanks: Vec<Vec<Blank>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
  Draw,
  DecidedKomi0,
  DecidedOtherWhole,
  HalfKomi,
}

/// A position of a game the networks are asked about.
struct Position {
  /// The quarter of the game it is in, from 0 to 3.
  quarter: usize,
  /// The game's result from the view of the player to move: 1, 0 or -1.
  outcome: f64,
  /// The final score of the field from the view of the player to move, komi
  /// not included, as the score head predicts it.
  final_score: f64,
  /// The board level, komi included: where the draw term applies.
  level: bool,
  /// How far into the part of the game where the draw term applies, see
  /// `draw_phase`.
  phase: f64,
  last: bool,
  /// The index into the policy of the move played, if any.
  played: Option<usize>,
  /// The policy indices of the cells where the player to move and the
  /// opponent capture with a single dot.
  captures: [Vec<usize>; 2],
  /// For every cell, whether it is captured by the player to move and by the
  /// opponent now and at the end of the game: what the captured head
  /// predicts the latter of.
  captured_now: [Vec<bool>; 2],
  captured_end: [Vec<bool>; 2],
}

/// The inputs for the positions of one game.
struct Batch {
  kind: Kind,
  whole_komi: bool,
  inputs: Array4<f32>,
  global: Array2<f32>,
  positions: Vec<Position>,
}

fn batch(game: &Game, from: usize, stride: usize) -> Option<Batch> {
  let (width, height) = (game.width, game.height);
  let final_field = game.final_field();
  let result_x_2 = final_field.score(Player::Red) * 2 + game.komi_x_2;
  let kind = match (result_x_2 == 0, game.whole_komi(), game.komi_x_2 == 0) {
    (true, _, _) => Kind::Draw,
    (false, false, _) => Kind::HalfKomi,
    (false, true, true) => Kind::DecidedKomi0,
    (false, true, false) => Kind::DecidedOtherWhole,
  };
  let length = game.moves.len();
  let mut field = game.field();
  let (mut features, mut global, mut positions) = (Vec::new(), Vec::new(), Vec::new());
  for i in 0..=length {
    let sampled = i >= from && ((i - from).is_multiple_of(stride.max(1)) || i == length);
    if sampled && length > 0 {
      let player = if i < length {
        game.moves[i].1
      } else {
        game.moves[length - 1].1.next()
      };
      let komi_x_2 = if player == Player::Red {
        game.komi_x_2
      } else {
        -game.komi_x_2
      };
      let tactics = Tactics::new(&mut field, player);
      field_features_to_vec::<f32>(
        &field,
        player,
        &tactics,
        width,
        height,
        0,
        HISTORY_CHANNELS,
        &mut features,
      );
      global_to_vec::<f32>(&field, player, komi_x_2, &mut global);
      let index = |pos| {
        let (x, y) = field.to_xy(pos);
        (y * width + x) as usize
      };
      let mut captures = [Vec::new(), Vec::new()];
      for y in 0..height {
        for x in 0..width {
          let pos = field.to_pos(x, y);
          for (plane, cells) in captures.iter_mut().enumerate() {
            if tactics.plane(plane, pos) {
              cells.push((y * width + x) as usize);
            }
          }
        }
      }
      let captured = |field: &oppai_field::field::Field, owner: Player| -> Vec<bool> {
        (0..height)
          .flat_map(|y| (0..width).map(move |x| (x, y)))
          .map(|(x, y)| {
            let cell = field.cell(field.to_pos(x, y));
            cell.is_captured() && cell.is_owner(owner)
          })
          .collect()
      };
      let captured_now = [captured(&field, player), captured(&field, player.next())];
      let captured_end = [captured(&final_field, player), captured(&final_field, player.next())];
      let red_outcome = result_x_2.signum() as f64;
      positions.push(Position {
        quarter: (4 * i / (length + 1)).min(3),
        outcome: if player == Player::Red {
          red_outcome
        } else {
          -red_outcome
        },
        final_score: final_field.score(player) as f64,
        level: field.score(player) * 2 + komi_x_2 == 0,
        phase: draw_phase(&field),
        last: i == length,
        played: (i < length).then(|| index(game.moves[i].0)),
        captures,
        captured_now,
        captured_end,
      });
    }
    if i < length {
      let (pos, player) = game.moves[i];
      assert!(field.put_point(pos, player));
      field.update_grounded();
    }
  }
  let n = positions.len();
  if n == 0 {
    return None;
  }
  Some(Batch {
    kind,
    whole_komi: game.whole_komi(),
    inputs: Array4::from_shape_vec((n, CHANNELS, height as usize, width as usize), features).ok()?,
    global: Array2::from_shape_vec((n, GLOBAL_FEATURES), global).ok()?,
    positions,
  })
}

#[derive(Default)]
struct Calibration {
  positions: usize,
  value_sq: [Vec<f64>; 4],
  sign_right: usize,
  decided: usize,
  score_abs: Vec<f64>,
  played: usize,
  top_played: usize,
  played_nll: Vec<f64>,
}

#[derive(Default)]
struct DrawStats {
  /// P(draw) at the final position, per kind.
  last: [Vec<f64>; 4],
  /// The weight of the draw term, `settled` times the phase, at level
  /// positions, per kind; a half komi never leaves the
  /// board level.
  level: [Vec<f64>; 4],
  /// The weight of the draw term at level positions of drawn games, per
  /// quarter.
  draw_quarters: [Vec<f64>; 4],
  /// P(draw) at level positions of games under a whole komi, per quarter: of
  /// drawn games and of decided ones.
  level_quarters: [[Vec<f64>; 4]; 2],
}

#[derive(Default)]
struct Ablation {
  positions: usize,
  kl: f64,
  top_kept: usize,
  value: f64,
  score: f64,
  draw_whole: Vec<f64>,
  draw_half: Vec<f64>,
  /// Per own and opponent's capture cells: positions with any, and the policy
  /// on them before and after blanking.
  captures: [(usize, f64, f64); 2],
}

/// The predictions of the captured head for one of its two channels, against
/// what got captured by the end of the game.
#[derive(Default)]
struct CapturedStats {
  cells: usize,
  /// The cross-entropy of the predictions, summed over the cells.
  bce: f64,
  /// Cells captured at the end.
  captured_end: usize,
  /// The cells captured already, and of them those still captured at the end.
  now: usize,
  now_kept: usize,
  /// The summed predictions on the cells captured already.
  p_now: f64,
  /// The predictions and the outcomes on the cells not captured yet.
  open: Vec<(f32, bool)>,
}

impl CapturedStats {
  fn add(&mut self, p: f32, now: bool, end: bool) {
    let q = (p as f64).clamp(1e-7, 1.0 - 1e-7);
    self.cells += 1;
    self.bce -= if end { q.ln() } else { (1.0 - q).ln() };
    self.captured_end += end as usize;
    if now {
      self.now += 1;
      self.now_kept += end as usize;
      self.p_now += p as f64;
    } else {
      self.open.push((p, end));
    }
  }

  /// The cross-entropy of always predicting the share of the cells captured
  /// at the end: what a head that ignores the board gets at best.
  fn base_bce(&self) -> f64 {
    let p = (self.captured_end as f64 / self.cells as f64).clamp(1e-7, 1.0 - 1e-7);
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
  }

  /// The cross-entropy of telling the cells captured now from the rest and
  /// predicting for each of the two groups the share of it captured at the
  /// end: what a head that only reads the current captures gets at best.
  fn copy_bce(&self) -> f64 {
    let group = |positives: usize, total: usize| {
      let p = (positives as f64 / total.max(1) as f64).clamp(1e-7, 1.0 - 1e-7);
      -(positives as f64 * p.ln() + (total - positives) as f64 * (1.0 - p).ln())
    };
    let later = self.open.iter().filter(|(_, end)| *end).count();
    (group(later, self.open.len()) + group(self.now_kept, self.now)) / self.cells as f64
  }

  /// The probability that a cell not captured yet that gets captured is
  /// predicted higher than one that does not (the area under the ROC curve).
  fn open_auc(&mut self) -> Option<f64> {
    self.open.sort_by(|a, b| a.0.total_cmp(&b.0));
    let positives = self.open.iter().filter(|(_, end)| *end).count();
    let negatives = self.open.len() - positives;
    if positives == 0 || negatives == 0 {
      return None;
    }
    // Ranks with ties averaged.
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < self.open.len() {
      let mut j = i;
      while j < self.open.len() && self.open[j].0 == self.open[i].0 {
        j += 1;
      }
      let rank = (i + j + 1) as f64 / 2.0;
      rank_sum += rank * self.open[i..j].iter().filter(|(_, end)| *end).count() as f64;
      i = j;
    }
    Some((rank_sum - (positives * (positives + 1)) as f64 / 2.0) / (positives * negatives) as f64)
  }

  /// The mean prediction on the cells not captured yet that get captured and
  /// on those that do not.
  fn open_means(&self) -> (Option<f64>, Option<f64>) {
    let (mut later, mut never) = (Vec::new(), Vec::new());
    for &(p, end) in &self.open {
      if end {
        later.push(p as f64)
      } else {
        never.push(p as f64)
      }
    }
    (mean(&later), mean(&never))
  }
}

fn argmax(values: &[f32]) -> usize {
  values
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.total_cmp(b.1))
    .map_or(0, |(i, _)| i)
}

fn kind_index(kind: Kind) -> usize {
  kind as usize
}

pub fn run(params: NetParams) -> Result<()> {
  let device = oppai_zero_burn::device::parse_device(&params.device)?;
  let config = ModelConfig::from_file(&params.config)?;
  let batches: Vec<Batch> = params
    .games
    .iter()
    .filter_map(|game| batch(game, params.from, params.stride))
    .collect();
  let positions: usize = batches.iter().map(|b| b.positions.len()).sum();

  let mut calibration_table = Table::new(
    &format!(
      "Predictions on {} positions of {} games of {}, against what happened",
      positions,
      batches.len(),
      params.games_label
    ),
    &[
      "model",
      "value MSE",
      "by quarter of the game",
      "value sign right (decided)",
      "score MAE",
      "played move top",
      "played move NLL",
    ],
  );
  let mut draw_table = Table::new(
    "P(draw) at the final position (median), and the draw term's weight at level positions (share active / mean)",
    &[
      "model",
      "final: draws",
      "decided komi 0",
      "half komi",
      "level: draws",
      "decided whole komi",
      "draws by quarter (active)",
    ],
  );
  let mut level_table = Table::new(
    "P(draw) at level positions under a whole komi, by quarter of the game (p10 / median / p90), for the bounds of the draw term",
    &["model", "games", "positions", "1st quarter", "2nd", "3rd", "4th"],
  );
  let mut ablation_table = Table::new(
    "Blanked inputs: change of the outputs, and the policy on single-dot capture cells (own / opponent's)",
    &[
      "model",
      "blanked",
      "policy KL",
      "top move kept",
      "|d value|",
      "|d score|",
      "|d P(draw)| whole / half",
      "own captures",
      "opponent's captures",
    ],
  );
  let mut captured_table = Table::new(
    "Captured head: cells captured at the end of the game by the player to move (own) and the opponent, cross-entropy per cell against a constant and against copying the current captures, and the mean predicted probability by what happened",
    &[
      "model",
      "captures",
      "captured at end",
      "of them already",
      "BCE",
      "constant",
      "copy current",
      "p already",
      "p captured later",
      "p never",
      "AUC of later",
    ],
  );

  for path in &params.models {
    let model = BurnModel::load(&device, &config, path).map_err(|e| anyhow!("loading {}: {e:?}", path.display()))?;
    let predictor = Predictor {
      model,
      device: device.clone(),
    };
    let predict = |inputs: Array4<f32>, global: Array2<f32>| {
      let n = inputs.len_of(ndarray::Axis(0));
      futures::executor::block_on(predictor.predict(inputs, global, Array1::zeros(n)))
        .map_err(|e| anyhow!("prediction failed: {e:?}"))
    };
    let mut calibration = Calibration::default();
    let mut draws = DrawStats::default();
    let mut ablations: Vec<Ablation> = params.blanks.iter().map(|_| Ablation::default()).collect();
    let mut captured_stats = [CapturedStats::default(), CapturedStats::default()];

    for batch in &batches {
      let (policies, values) = predict(batch.inputs.clone(), batch.global.clone())?;
      let k = kind_index(batch.kind);
      let ahead = predictor
        .predict_board_ahead(batch.inputs.clone(), batch.global.clone())
        .map_err(|e| anyhow!("prediction failed: {e:?}"))?;
      for (j, position) in batch.positions.iter().enumerate() {
        for (channel, stats) in captured_stats.iter_mut().enumerate() {
          let p = ahead.slice(s![j, channel, .., ..]);
          for ((&p, &now), &end) in p
            .iter()
            .zip(&position.captured_now[channel])
            .zip(&position.captured_end[channel])
          {
            stats.add(p, now, end);
          }
        }
      }
      let policy = |p: &ndarray::Array3<f32>, j: usize| p.slice(s![j, .., ..]).iter().copied().collect::<Vec<f32>>();
      for (j, position) in batch.positions.iter().enumerate() {
        let value = (values[(j, 0)] - values[(j, 1)]) as f64;
        let draw = values[(j, 4)] as f64;
        calibration.positions += 1;
        calibration.value_sq[position.quarter].push((value - position.outcome).powi(2));
        if position.outcome != 0.0 {
          calibration.decided += 1;
          if value * position.outcome > 0.0 {
            calibration.sign_right += 1;
          }
        }
        calibration
          .score_abs
          .push((values[(j, 3)] as f64 - position.final_score).abs());
        if let Some(played) = position.played {
          let p = policy(&policies, j);
          calibration.played += 1;
          if argmax(&p) == played {
            calibration.top_played += 1;
          }
          calibration.played_nll.push(-(p[played].max(1e-9) as f64).ln());
        }
        if position.last {
          draws.last[k].push(draw);
        }
        if position.level {
          let weight = settled(draw) * position.phase;
          draws.level[k].push(weight);
          if batch.kind == Kind::Draw {
            draws.draw_quarters[position.quarter].push(weight);
          }
          if batch.whole_komi {
            draws.level_quarters[usize::from(batch.kind != Kind::Draw)][position.quarter].push(draw);
          }
        }
      }

      for (blanks, ablation) in params.blanks.iter().zip(&mut ablations) {
        let (mut inputs, mut global) = (batch.inputs.clone(), batch.global.clone());
        for blank in blanks {
          blank.apply(&mut inputs, &mut global);
        }
        let (blank_policies, blank_values) = predict(inputs, global)?;
        for (j, position) in batch.positions.iter().enumerate() {
          let (p, q) = (policy(&policies, j), policy(&blank_policies, j));
          ablation.positions += 1;
          ablation.kl += p
            .iter()
            .zip(&q)
            .filter(|(p, _)| **p > 1e-8)
            .map(|(&p, &q)| p as f64 * (p as f64 / (q as f64).max(1e-9)).ln())
            .sum::<f64>();
          if argmax(&p) == argmax(&q) {
            ablation.top_kept += 1;
          }
          let value = |v: &Array2<f32>| (v[(j, 0)] - v[(j, 1)]) as f64;
          ablation.value += (value(&values) - value(&blank_values)).abs();
          ablation.score += (values[(j, 3)] - blank_values[(j, 3)]).abs() as f64;
          let d_draw = (values[(j, 4)] - blank_values[(j, 4)]).abs() as f64;
          if batch.whole_komi {
            ablation.draw_whole.push(d_draw);
          } else {
            ablation.draw_half.push(d_draw);
          }
          for (cells, (count, before, after)) in position.captures.iter().zip(&mut ablation.captures) {
            if !cells.is_empty() {
              *count += 1;
              *before += cells.iter().map(|&c| p[c] as f64).sum::<f64>();
              *after += cells.iter().map(|&c| q[c] as f64).sum::<f64>();
            }
          }
        }
      }
    }

    let name = label(path);
    let all_sq: Vec<f64> = calibration.value_sq.iter().flatten().copied().collect();
    calibration_table.row(vec![
      name.clone(),
      fmt(mean(&all_sq), 4),
      calibration
        .value_sq
        .iter()
        .map(|q| fmt(mean(q), 3))
        .collect::<Vec<_>>()
        .join(" "),
      percent(calibration.sign_right, calibration.decided),
      fmt(mean(&calibration.score_abs), 2),
      percent(calibration.top_played, calibration.played),
      fmt(mean(&calibration.played_nll), 3),
    ]);

    let active = |values: &[f64]| {
      format!(
        "{} / {}",
        percent(values.iter().filter(|&&s| s > 0.0).count(), values.len()),
        fmt(mean(values), 2)
      )
    };
    let decided_whole: Vec<f64> = draws.level[kind_index(Kind::DecidedKomi0)]
      .iter()
      .chain(&draws.level[kind_index(Kind::DecidedOtherWhole)])
      .copied()
      .collect();
    draw_table.row(vec![
      name.clone(),
      fmt(median(&draws.last[kind_index(Kind::Draw)]), 3),
      fmt(median(&draws.last[kind_index(Kind::DecidedKomi0)]), 3),
      fmt(median(&draws.last[kind_index(Kind::HalfKomi)]), 3),
      active(&draws.level[kind_index(Kind::Draw)]),
      active(&decided_whole),
      draws
        .draw_quarters
        .iter()
        .map(|q| percent(q.iter().filter(|&&s| s > 0.0).count(), q.len()))
        .collect::<Vec<_>>()
        .join(" "),
    ]);

    for (outcome, quarters) in ["drawn", "decided"].into_iter().zip(&draws.level_quarters) {
      let mut row = vec![
        name.clone(),
        outcome.to_string(),
        quarters.iter().map(Vec::len).sum::<usize>().to_string(),
      ];
      row.extend(quarters.iter().map(|q| {
        format!(
          "{} / {} / {}",
          fmt(quantile(q, 0.1), 2),
          fmt(median(q), 2),
          fmt(quantile(q, 0.9), 2)
        )
      }));
      level_table.row(row);
    }

    for (channel, stats) in ["own", "opponent's"].into_iter().zip(&mut captured_stats) {
      let (later, never) = stats.open_means();
      captured_table.row(vec![
        name.clone(),
        channel.to_string(),
        percent(stats.captured_end, stats.cells),
        percent(stats.now_kept, stats.captured_end),
        fmt(Some(stats.bce / stats.cells as f64), 4),
        fmt(Some(stats.base_bce()), 4),
        fmt(Some(stats.copy_bce()), 4),
        fmt(Some(stats.p_now / stats.now.max(1) as f64), 3),
        fmt(later, 3),
        fmt(never, 4),
        fmt(stats.open_auc(), 3),
      ]);
    }

    for (blanks, ablation) in params.blanks.iter().zip(&ablations) {
      let n = ablation.positions.max(1) as f64;
      let captures = |(count, before, after): (usize, f64, f64)| {
        if count == 0 {
          "-".to_string()
        } else {
          format!(
            "{:.3} -> {:.3} ({})",
            before / count as f64,
            after / count as f64,
            percent(count, ablation.positions)
          )
        }
      };
      ablation_table.row(vec![
        name.clone(),
        blanks.iter().map(|b| b.name()).collect::<Vec<_>>().join("+"),
        format!("{:.4}", ablation.kl / n),
        percent(ablation.top_kept, ablation.positions),
        format!("{:.4}", ablation.value / n),
        format!("{:.3}", ablation.score / n),
        format!(
          "{} / {}",
          fmt(mean(&ablation.draw_whole), 4),
          fmt(mean(&ablation.draw_half), 4)
        ),
        captures(ablation.captures[0]),
        captures(ablation.captures[1]),
      ]);
    }
  }

  calibration_table.print();
  draw_table.print();
  level_table.print();
  captured_table.print();
  if !params.blanks.is_empty() {
    ablation_table.print();
  }
  Ok(())
}
