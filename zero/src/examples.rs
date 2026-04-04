use crate::{
  episode::Visits,
  field_features::{
    CHANNELS, GLOBAL_FEATURES, SCORE_ONE_HOT_SIZE, field_features_to_vec, global_to_vec, score_one_hot_to_vec,
  },
};
use ndarray::{Array, Array2, Array3, Array4};
use num_traits::{Float, One, Zero};
use oppai_field::{
  field::{Field, Pos},
  player::Player,
  zobrist::Zobrist,
};
use oppai_rotate::rotate::{MIRRORS, ROTATIONS};
use rand::{Rng, seq::SliceRandom};
use std::{cmp::Ordering, ops::Range, sync::Arc};

#[derive(Clone, Debug)]
pub struct Batch<N> {
  pub inputs: Array4<N>,
  pub global: Array2<N>,
  pub policies: Array3<N>,
  pub opponent_policies: Array3<N>,
  pub values: Array2<N>,
  pub scores: Array2<N>,
}

#[derive(Clone, Debug)]
pub struct ExampleGame {
  /// Field width
  pub width: u32,
  /// Field height
  pub height: u32,
  /// Moves played before this training position
  pub moves: Vec<(Pos, Player)>,
  /// Komi, multiplied by 2
  pub komi_x_2: i32,
  /// Score at the terminal game state
  pub score: i32,
  pub visits: Vec<Visits>,
}

#[derive(Clone, Debug)]
pub struct Example {
  pub game: usize,
  pub position: usize,
  pub rotation: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Examples {
  pub games: Vec<ExampleGame>,
  pub examples: Vec<Example>,
}

impl Examples {
  pub fn add(&mut self, komi_x_2: i32, visits: Vec<Visits>, field: &Field, rotations: bool) {
    let initial_moves = field.moves_count() - visits.len();
    let rotations = if rotations { ROTATIONS } else { MIRRORS };

    let game = ExampleGame {
      width: field.width(),
      height: field.height(),
      moves: field.colored_moves().collect(),
      komi_x_2,
      score: field.score(Player::Red),
      visits,
    };
    let game_index = self.games.len();
    self.games.push(game);

    for (i, visits) in self.games[game_index].visits.iter().enumerate() {
      if visits.1 {
        for rotation in 0..rotations {
          self.examples.push(Example {
            game: game_index,
            position: initial_moves + i,
            rotation,
          });
        }
      }
    }
  }

  #[inline]
  pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
    self.examples.shuffle(rng);
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.examples.len()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.examples.is_empty()
  }

  #[inline]
  pub fn batches_count(&self, size: usize) -> usize {
    (self.len() / size).max(1)
  }

  fn values_to_vec<N: Float + Zero + One + Copy>(score: i32, komi_x_2: i32, values: &mut Vec<N>) {
    let score = score * 2 + komi_x_2;
    let scores = match score.cmp(&0) {
      Ordering::Less => [N::zero(), N::one()],
      Ordering::Equal => [N::one() / (N::one() + N::one()), N::one() / (N::one() + N::one())],
      Ordering::Greater => [N::one(), N::zero()],
    };
    values.extend_from_slice(&scores);
  }

  fn batch<N: Float + Zero + One + Copy>(
    &self,
    range: Range<usize>,
    width: u32,
    height: u32,
    zobrist: Arc<Zobrist<u64>>,
  ) -> Batch<N> {
    let mut inputs = Vec::<N>::with_capacity(range.len() * CHANNELS * height as usize * width as usize);
    let mut global = Vec::<N>::with_capacity(range.len() * GLOBAL_FEATURES);
    let mut policies = Vec::<N>::with_capacity(range.len() * height as usize * width as usize);
    let mut opponent_policies = Vec::<N>::with_capacity(range.len() * height as usize * width as usize);
    let mut values = Vec::<N>::with_capacity(range.len() * 2);
    let mut scores = Vec::<N>::with_capacity(range.len() * SCORE_ONE_HOT_SIZE);
    for example in self.examples.get(range.clone()).unwrap() {
      let game = &self.games[example.game];
      let mut field = Field::new(game.width, game.height, zobrist.clone());
      for &(pos, player) in game.moves.iter().take(example.position) {
        assert!(field.put_point(pos, player));
        field.update_grounded();
      }
      let player = game.moves[example.position - 1].1.next();
      let (score, komi_x_2) = if player == Player::Red {
        (game.score, game.komi_x_2)
      } else {
        (-game.score, -game.komi_x_2)
      };
      field_features_to_vec(&field, player, width, height, example.rotation, &mut inputs);
      global_to_vec(&field, player, komi_x_2, &mut global);
      let initial_moves = game.moves.len() - game.visits.len();
      game.visits[example.position - initial_moves].policies_to_vec(
        width,
        height,
        game.width,
        game.height,
        example.rotation,
        &mut policies,
      );
      let default_vists = Visits::default();
      game
        .visits
        .get(example.position - initial_moves + 1)
        .unwrap_or(&default_vists)
        .policies_to_vec(
          width,
          height,
          game.width,
          game.height,
          example.rotation,
          &mut opponent_policies,
        );
      Self::values_to_vec::<N>(score, komi_x_2, &mut values);
      score_one_hot_to_vec(score, komi_x_2, &mut scores);
    }
    Batch {
      inputs: Array::from(inputs)
        .into_shape_with_order((range.len(), CHANNELS, height as usize, width as usize))
        .unwrap(),
      global: Array::from(global)
        .into_shape_with_order((range.len(), GLOBAL_FEATURES))
        .unwrap(),
      policies: Array::from(policies)
        .into_shape_with_order((range.len(), height as usize, width as usize))
        .unwrap(),
      opponent_policies: Array::from(opponent_policies)
        .into_shape_with_order((range.len(), height as usize, width as usize))
        .unwrap(),
      values: Array::from(values).into_shape_with_order((range.len(), 2)).unwrap(),
      scores: Array::from(scores)
        .into_shape_with_order((range.len(), SCORE_ONE_HOT_SIZE))
        .unwrap(),
    }
  }

  pub fn batches<N: Float + Zero + One + Copy>(
    &self,
    width: u32,
    height: u32,
    zobrist: Arc<Zobrist<u64>>,
    size: usize,
  ) -> impl Iterator<Item = Batch<N>> + '_ {
    (0..self.batches_count(size))
      .map(move |i| self.batch::<N>(i * size..(i + 1) * size, width, height, zobrist.clone()))
  }
}
