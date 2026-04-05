use crate::field_features::{field_features, global};
use crate::mcgs::Search;
use crate::model::Model;
use log::info;
use ndarray::{Array2, Array3, Axis};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use oppai_rotate::rotate::rotate;
use rand::distr::uniform::SampleUniform;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp, Exp1, Open01, StandardNormal};
use std::fmt::{Debug, Display};
use std::iter::{self, Sum};

const MCTS_SIMS: u32 = 200;
const MCTS_FULL_SIMS: u32 = 1000;

#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct Visits(pub Vec<(Pos, u64)>, pub bool);

impl Visits {
  pub fn total(&self) -> u64 {
    self.0.iter().map(|&(_, v)| v).sum()
  }

  pub fn max(&self) -> u64 {
    self.0.iter().map(|&(_, v)| v).max().unwrap_or_default()
  }

  /// Improved stochastic policy values, pushed into an existing vector.
  pub fn policies_to_vec<N: Float + Copy>(
    &self,
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    rotation: u8,
    policies: &mut Vec<N>,
  ) {
    let total = self.total();
    let start_idx = policies.len();

    policies.extend(iter::repeat_n(N::zero(), (width * height) as usize));

    if total > 0 {
      for &(pos, visits) in &self.0 {
        let x = to_x(field_width + 1, pos);
        let y = to_y(field_width + 1, pos);
        let (x, y) = rotate(field_width, field_height, x, y, rotation);

        let idx = start_idx + (y as usize) * (width as usize) + (x as usize);
        policies[idx] = N::from(visits).unwrap() / N::from(total).unwrap();
      }
    } else {
      let uniform_prob = N::one() / N::from(field_width * field_height).unwrap();
      for y in 0..field_height as usize {
        for x in 0..field_width as usize {
          let idx = start_idx + y * (width as usize) + x;
          policies[idx] = uniform_prob;
        }
      }
    }
  }

  /// Improved stochastic policy values.
  pub fn policies<N: Float>(
    &self,
    width: u32,
    height: u32,
    field_width: u32,
    field_height: u32,
    rotation: u8,
  ) -> Array2<N> {
    let mut vec = Vec::with_capacity((width * height) as usize);

    self.policies_to_vec(width, height, field_width, field_height, rotation, &mut vec);

    Array2::from_shape_vec((height as usize, width as usize), vec).unwrap()
  }
}

fn interpolate_early<N: Float>(field: &Field, early_value: N, value: N) -> N {
  let halflives = N::from(field.moves_count()).unwrap() / N::from(field.width() * field.height()).unwrap().sqrt();
  value + (early_value - value) * N::from(0.5).unwrap().powf(halflives)
}

fn select_policy_move<N, R>(field: &Field, policy: Array3<N>, rng: &mut R) -> Option<NonZeroPos>
where
  N: Float + Sum + SampleUniform,
  R: Rng,
{
  // TODO: KataGo also makes random moves with small probability, see PlayUtils::getGameInitializationMove
  let mut sum = N::zero();
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      sum = sum + policy[(0, y as usize, x as usize)];
    }
  }
  let mut sample = rng.random_range(N::zero()..sum);
  for pos in field.min_pos()..=field.max_pos() {
    if field.is_putting_allowed(pos) {
      let (x, y) = field.to_xy(pos);
      let policy = policy[(0, y as usize, x as usize)];
      if policy >= sample {
        return NonZeroPos::new(pos);
      } else {
        sample = sample - policy;
      }
    }
  }
  None
}

pub fn episode<N, M, R>(
  field: &mut Field,
  mut player: Player,
  model: &mut M,
  mut komi_x_2: i32,
  rng: &mut R,
) -> Result<Vec<Visits>, M::E>
where
  M: Model<N>,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  // TODO: does it scale to big sizes?
  let exp = Exp::new(N::from(25).unwrap() / N::from(field.width() * field.height()).unwrap()).unwrap();
  let raw_policy_moves = exp.sample(rng).floor().to_u32().unwrap();

  info!("Playing {} raw policy moves", raw_policy_moves);

  for _ in 0..raw_policy_moves {
    let features = field_features(field, player, field.width(), field.height(), 0);
    let global = global(field, player, komi_x_2);
    let (policy, _) = model.predict(features.insert_axis(Axis(0)), global.insert_axis(Axis(0)))?;
    if let Some(pos) = select_policy_move(field, policy, rng) {
      assert!(field.put_point(pos.get(), player));
      field.update_grounded();
      player = player.next();
      komi_x_2 = -komi_x_2;
    } else {
      break;
    }
  }

  let mut search = Search::new();
  let mut visits = Vec::new();

  while !field.is_game_over(komi_x_2) {
    let full_search = rng.random::<f64>() <= 0.25;

    let sims = if full_search {
      // TODO: does it scale to big sizes?
      let shape = N::from(0.03 * 19.0.powi(2)).unwrap() / N::from(field.width() * field.height()).unwrap();
      let temperature = interpolate_early(field, N::from(1.25).unwrap(), N::from(1.1).unwrap());
      search.add_dirichlet_noise(rng, N::from(0.25).unwrap(), shape, temperature);
      MCTS_FULL_SIMS
    } else {
      MCTS_SIMS
    };

    for _ in 0..sims {
      search.mcgs(field, player, model, komi_x_2, rng)?;
    }

    visits.push(Visits(
      if full_search {
        // Use pruned visits for full searches with Dirichlet noise.
        // This removes the extra forced playouts from the policy target,
        // producing a cleaner training signal.
        search.pruned_visits().collect()
      } else {
        search.visits().collect()
      },
      full_search,
    ));

    let pos = if let Some(pos) = search.next_root_with_temperature(
      interpolate_early(field, N::from(0.75).unwrap(), N::from(0.15).unwrap()),
      rng,
    ) {
      pos
    } else {
      break;
    };

    search.compact();
    assert!(field.put_point(pos.get(), player));
    field.update_grounded();
    player = player.next();
    komi_x_2 = -komi_x_2;
  }

  Ok(visits)
}
