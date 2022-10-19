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
const PIT_GAMES: u32 = 50;
const WIN_RATE_THRESHOLD: f64 = 0.55;
const BATCH_SIZE: usize = 128;
const EPOCHS: u32 = 200;
const EPISODES: u32 = 10;
const MCTS_SIMS: u32 = 10;

fn play_single_move<E, M>(field: &mut Field, player: Player, model: &M) -> Result<(), E>
where
  M: Model<E = E>,
{
  let mut node = MctsNode::new(0, 0f64, 0f64);
  for _ in 0..MCTS_SIMS {
    mcts(field, player, &mut node, model)?;
  }

  let pos = node.best_move().unwrap();
  field.put_point(pos, player);

  Ok(())
}

fn play<E, M>(field: &mut Field, player: Player, model1: &M, model2: &M) -> Result<i32, E>
where
  M: Model<E = E>,
{
  loop {
    // TODO: persistent tree?
    play_single_move(field, player, model1)?;
    if field.is_game_over() {
      break;
    }

    play_single_move(field, player.next(), model2)?;
    if field.is_game_over() {
      break;
    }
  }

  Ok(field.score(player))
}

fn pit<E, M>(field: &Field, player: Player, new_model: &M, old_model: &M) -> Result<bool, E>
where
  M: Model<E = E>,
{
  let mut wins = 0;
  let mut losses = 0;

  for i in 0..PIT_GAMES {
    log::info!("Game {}", i * 2);

    match play(&mut field.clone(), player, new_model, old_model)?.cmp(&0) {
      Ordering::Less => losses += 1,
      Ordering::Greater => wins += 1,
      Ordering::Equal => {}
    };

    log::info!("Game {}", i * 2 + 1);

    match play(&mut field.clone(), player, old_model, new_model)?.cmp(&0) {
      Ordering::Less => wins += 1,
      Ordering::Greater => losses += 1,
      Ordering::Equal => {}
    };
  }

  let draws = PIT_GAMES * 2 - wins - losses;
  let win_rate = (wins as f64 + draws as f64 / 2.0) / (PIT_GAMES * 2) as f64;
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
    if pit(field, player, &model, &copy)? {
      model.save()?;
    } else {
      log::warn!("Rejecting new model");
      model = copy;
    }
  }

  Ok(())
}
