//! The training losses of networks on the positions of games they weren't
//! trained on.
//!
//! The games a model plays are the first it hasn't been trained on: model
//! `N + 1` is trained on games up to `games_N`, and plays `games_{N+1}`. So
//! the losses of `model_{N+1}` and `model_swa_{N+1}` on `games_{N+1}` say how
//! well each predicts new positions - and how much averaging the weights gains
//! over the weights a training ends on. A large gain means the steps of the
//! training carry a lot of noise, which a lower learning rate would cut.
//!
//! The examples are made from the games as training makes them, without the
//! surprise weighting, so that every searched position counts once, and
//! without the side positions; the loss terms are those training logs.

use crate::common::{Table, for_each_game, komi_x_2, label};
use anyhow::{Result, anyhow};
use burn::tensor::FloatDType;
use oppai_field::{
  field::{Field, length},
  zobrist::Zobrist,
};
use oppai_sgf::from_sgf;
use oppai_zero::examples::Examples;
use oppai_zero_burn::model::{Model as BurnModel, ModelConfig, Predictor};
use oppai_zero_sgf::sgf_to_visits;
use rand::{SeedableRng, rngs::SmallRng};
use std::{path::PathBuf, sync::Arc};

pub struct HoldoutParams {
  pub device: String,
  pub config: PathBuf,
  pub models: Vec<PathBuf>,
  pub games: PathBuf,
  pub count: Option<usize>,
  /// Stop reading games once they hold this many positions.
  pub positions: Option<usize>,
  pub batch_size: usize,
  /// Whether the convolutions compute in brain floating point, as in training
  /// with mixed precision.
  pub half: bool,
}

pub fn run(params: HoldoutParams) -> Result<()> {
  let device = oppai_zero_burn::device::parse_device(&params.device)?;
  let config = ModelConfig::from_file(&params.config)?;
  // Seeded, so that every run makes the same examples - the same rotations
  // and history dropout - and every model is measured on the same rows.
  let mut rng = SmallRng::seed_from_u64(1);

  let mut examples = Examples::default();
  let (mut width, mut height) = (0, 0);
  let mut games = 0;
  let mut invalid = 0;
  for_each_game(&params.games, |node| {
    if params.count.is_some_and(|count| games >= count)
      || params.positions.is_some_and(|positions| examples.len() >= positions)
    {
      return false;
    }
    let Some(field) = from_sgf::<Field, _>(&node, &mut rng) else {
      invalid += 1;
      return true;
    };
    let visits = sgf_to_visits(&node, field.stride);
    width = width.max(field.width());
    height = height.max(field.height());
    examples.add(komi_x_2(&node), visits, &field, true, false, &mut rng);
    games += 1;
    true
  })?;
  if invalid > 0 {
    eprintln!("{}: skipped {invalid} invalid games", params.games.display());
  }
  if examples.is_empty() {
    return Err(anyhow!("{} holds no searched positions", params.games.display()));
  }
  // Shuffled rather than grouped by board size as training groups them: a
  // batch is padded to its biggest board, and every new shape of a batch has
  // its kernels compiled and tuned first, which in a single run over a few
  // thousand positions costs far more than the padding of mixed batches does.
  examples.shuffle(&mut rng);
  let zobrist = Arc::new(Zobrist::new(length(width, height) * 3, &mut rng));

  // Per model, the loss terms in the order training logs them.
  let mut columns: Vec<(String, Vec<(&'static str, f64)>)> = Vec::new();
  for path in &params.models {
    let model = BurnModel::load(&device, &config, path).map_err(|e| anyhow!("loading {}: {e:?}", path.display()))?;
    let predictor = Predictor {
      model,
      device: device.clone(),
    };
    let mut sums: Vec<(&'static str, f64)> = Vec::new();
    let mut rows = 0;
    let mut built = std::time::Instant::now();
    for batch in examples.batches::<f32>(zobrist.clone(), params.batch_size) {
      let started = std::time::Instant::now();
      // The terms are means over the batch, so they are weighed by its rows.
      let batch_rows = batch.inputs.dim().0;
      let batch_shape = batch.inputs.dim();
      let losses = predictor
        .losses(
          params.half.then_some(FloatDType::BF16),
          batch.inputs,
          batch.global,
          batch.policies,
          batch.opponent_policies,
          batch.values,
          batch.td_values,
          batch.td_scores,
          batch.scores,
          batch.captured,
          batch.future,
          batch.q_values,
          batch.outcome_weights,
        )
        .values();
      if sums.is_empty() {
        sums = losses.iter().map(|&(name, _)| (name, 0.0)).collect();
      }
      for ((_, sum), (_, loss)) in sums.iter_mut().zip(losses) {
        *sum += f64::from(loss) * batch_rows as f64;
      }
      rows += batch_rows;
      log::debug!(
        "batch of {:?}: built in {:?}, losses in {:?}",
        batch_shape,
        started - built,
        started.elapsed()
      );
      built = std::time::Instant::now();
    }
    for (_, sum) in &mut sums {
      *sum /= rows as f64;
    }
    columns.push((label(path), sums));
  }

  let mut header = vec!["loss".to_string()];
  header.extend(columns.iter().map(|(model, _)| model.clone()));
  let mut table = Table::new(
    &format!(
      "Training losses on {} positions of {games} games of {}, every searched position once",
      examples.len(),
      label(&params.games)
    ),
    &header.iter().map(String::as_str).collect::<Vec<_>>(),
  );
  let terms = columns.first().map_or(0, |(_, sums)| sums.len());
  for term in 0..terms {
    let mut row = vec![columns[0].1[term].0.to_string()];
    row.extend(columns.iter().map(|(_, sums)| format!("{:.5}", sums[term].1)));
    table.row(row);
  }
  let mut total = vec!["total".to_string()];
  total.extend(
    columns
      .iter()
      .map(|(_, sums)| format!("{:.5}", sums.iter().map(|(_, loss)| loss).sum::<f64>())),
  );
  table.row(total);
  table.print();
  Ok(())
}
