use std::cmp::Ordering;

use crate::episode::{episode, mcts};
use crate::mcts::MctsNode;
use crate::model::{Model, TrainableModel};
use ndarray::{Array, Axis};
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::seq::SliceRandom;
use rand::Rng;

const ITERATIONS_NUMBER: u32 = 10000;
const PIT_GAMES: u64 = 50;
const WIN_RATE_THRESHOLD: f64 = 0.55;
const BATCH_SIZE: usize = 128;
const EPOCHS: u32 = 200;
const EPISODES: u32 = 10;
const MCTS_SIMS: u32 = 32;

fn play_single_move<E, M, R>(field: &mut Field, player: Player, model: &M, rng: &mut R) -> Result<bool, E>
where
  M: Model<E = E>,
  R: Rng,
{
  let mut node = MctsNode::new(0, 0f64, 0f64);
  for _ in 0..MCTS_SIMS {
    mcts(field, player, &mut node, model)?;
  }

  if let Some(pos) = node.best_move(rng) {
    field.put_point(pos, player);
    Ok(true)
  } else {
    Ok(false)
  }
}

fn play<E, M, R>(field: &mut Field, player: Player, model1: &M, model2: &M, rng: &mut R) -> Result<i32, E>
where
  M: Model<E = E>,
  R: Rng,
{
  loop {
    // TODO: persistent tree?
    if !play_single_move(field, player, model1, rng)? || field.is_game_over() {
      break;
    }

    if !play_single_move(field, player.next(), model2, rng)? || field.is_game_over() {
      break;
    }
  }

  Ok(field.score(player))
}

fn win_rate(wins: u64, losses: u64, games: u64) -> f64 {
  if games == 0 {
    0.0
  } else {
    let draws = games - wins - losses;
    (wins as f64 + draws as f64 / 2.0) / games as f64
  }
}

fn pit<E, M, R>(field: &Field, player: Player, new_model: &M, old_model: &M, rng: &mut R) -> Result<bool, E>
where
  M: Model<E = E>,
  R: Rng,
{
  let mut wins = 0;
  let mut losses = 0;

  for i in 0..PIT_GAMES {
    log::info!("Game {}, win rate {}", i * 2, win_rate(wins, losses, i * 2));

    match play(&mut field.clone(), player, new_model, old_model, rng)?.cmp(&0) {
      Ordering::Less => losses += 1,
      Ordering::Greater => wins += 1,
      Ordering::Equal => {}
    };

    log::info!("Game {}, win rate {}", i * 2 + 1, win_rate(wins, losses, i * 2 + 1));

    match play(&mut field.clone(), player, old_model, new_model, rng)?.cmp(&0) {
      Ordering::Less => wins += 1,
      Ordering::Greater => losses += 1,
      Ordering::Equal => {}
    };
  }

  let win_rate = win_rate(wins, losses, PIT_GAMES * 2);
  log::info!("Win rate is {}", win_rate);

  Ok(win_rate > WIN_RATE_THRESHOLD)
}

pub fn self_play<E, M, R>(field: &Field, player: Player, mut model: M, rng: &mut R) -> Result<(), E>
where
  M: TrainableModel<E = E> + Clone,
  R: Rng,
{
  for i in 0..ITERATIONS_NUMBER {
    log::info!("Iteration {}", i);

    let mut inputs = Vec::new();
    let mut policies = Vec::new();
    let mut values = Vec::new();
    for j in 0..EPISODES {
      log::info!("Episode {}", j);
      episode(
        &mut field.clone(),
        player,
        &model,
        rng,
        &mut inputs,
        &mut policies,
        &mut values,
      )?;
    }

    log::info!("Train the model");
    let copy = model.clone();

    if inputs.len() > BATCH_SIZE * EPOCHS as usize {
      log::warn!("Learning doesn't use all {} inputs", inputs.len());
    }

    let mut indices = (0..inputs.len()).collect::<Vec<_>>();
    let mut shift = inputs.len();
    for j in 0..EPOCHS {
      log::info!("Epoch {}", j);

      if indices.len() - shift < BATCH_SIZE {
        indices.shuffle(rng);
        shift = 0;
      }

      let inputs = ndarray::stack(
        Axis(0),
        indices[shift..shift + BATCH_SIZE]
          .iter()
          .map(|&i| inputs[i].view())
          .collect::<Vec<_>>()
          .as_slice(),
      )
      .unwrap();
      let policies = ndarray::stack(
        Axis(0),
        indices[shift..shift + BATCH_SIZE]
          .iter()
          .map(|&i| policies[i].view())
          .collect::<Vec<_>>()
          .as_slice(),
      )
      .unwrap();
      let values = Array::from_iter(indices[shift..shift + BATCH_SIZE].iter().map(|&i| values[i]));

      model.train(inputs, policies, values)?;

      shift += BATCH_SIZE;
    }

    log::info!("Pit the new model");
    if pit(field, player, &model, &copy, rng)? {
      model.save()?;
    } else {
      log::warn!("Rejecting new model");
      model = copy;
    }
  }

  Ok(())
}
