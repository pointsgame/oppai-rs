//! What networks predict for the positions of recorded games: how well the
//! value, score and policy match what happened, the probability of a draw the
//! draw term of the search reads, and, with inputs blanked, how much the
//! networks rely on each group of inputs.

use crate::common::{Game, Table, fmt, label, mean, median, percent};
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array4, s};
use oppai_field::player::Player;
use oppai_zero::{
  field_features::{
    CHANNELS, GLOBAL_FEATURES, GLOBAL_KOMI, GLOBAL_SHORTFALL, HISTORY_CHANNELS, field_features_to_vec, global_to_vec,
  },
  mcgs::settled,
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
  last: bool,
  /// The index into the policy of the move played, if any.
  played: Option<usize>,
  /// The policy indices of the cells where the player to move and the
  /// opponent capture with a single dot.
  captures: [Vec<usize>; 2],
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
        last: i == length,
        played: (i < length).then(|| index(game.moves[i].0)),
        captures,
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
  /// `settled` at level positions, per kind; a half komi never leaves the
  /// board level.
  level: [Vec<f64>; 4],
  /// `settled` at level positions of drawn games, per quarter.
  draw_quarters: [Vec<f64>; 4],
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

    for batch in &batches {
      let (policies, values) = predict(batch.inputs.clone(), batch.global.clone())?;
      let k = kind_index(batch.kind);
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
          draws.level[k].push(settled(draw));
          if batch.kind == Kind::Draw {
            draws.draw_quarters[position.quarter].push(settled(draw));
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
  if !params.blanks.is_empty() {
    ablation_table.print();
  }
  Ok(())
}
