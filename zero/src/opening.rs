use rand::{Rng, RngExt};
use rand_distr::{Distribution, weighted::WeightedIndex};

/// Picks a random offset for a pattern of the given size placed on a field of
/// the given size.
///
/// The pattern is kept around the center of the field: the offset deviates from
/// it by at most a sixth of the field size. The deviation is symmetrical, so
/// there is no skew towards either side regardless of the parity of the sizes -
/// when the free space can't be split evenly the two offsets closest to the
/// center are both allowed instead of preferring one of them.
fn offset<R: Rng>(field_size: u32, size: u32, rng: &mut R) -> u32 {
  let free = field_size.saturating_sub(size);
  let deviation = field_size / 6;
  // Both bounds are clamped by the same condition (`deviation > free / 2`),
  // hence clamping can't make the range asymmetrical.
  let min = (free / 2).saturating_sub(deviation);
  let max = (free / 2 + free % 2 + deviation).min(free);

  rng.random_range(min..=max)
}

fn cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let rotation = rng.random();
  let x_points;
  let o_points;

  if rotation {
    // XO
    // OX
    x_points = [(0, 0), (1, 1)];
    o_points = [(0, 1), (1, 0)];
  } else {
    // OX
    // XO
    x_points = [(0, 1), (1, 0)];
    o_points = [(0, 0), (1, 1)];
  }

  let x_offset = offset(width, 2, rng);
  let y_offset = offset(height, 2, rng);

  let mut result = Vec::new();
  for i in 0..2 {
    result.push((x_offset + x_points[i].0, y_offset + x_points[i].1));
    result.push((x_offset + o_points[i].0, y_offset + o_points[i].1));
  }

  result
}

fn triple_cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let rotation = rng.random_range(0..4);
  let x_points;
  let o_points;

  match rotation {
    0 => {
      // .X.
      // OXO
      // XOX
      // .O.
      x_points = [(1, 0), (1, 1), (0, 2), (2, 2)];
      o_points = [(1, 3), (1, 2), (0, 1), (2, 1)];
    }
    1 => {
      // .O.
      // XOX
      // OXO
      // .X.
      x_points = [(1, 3), (1, 2), (0, 1), (2, 1)];
      o_points = [(1, 0), (1, 1), (0, 2), (2, 2)];
    }
    2 => {
      // .OX.
      // XXOO
      // .OX.
      x_points = [(0, 1), (1, 1), (2, 0), (2, 2)];
      o_points = [(3, 1), (2, 1), (1, 0), (1, 2)];
    }
    3 => {
      // .XO.
      // OOXX
      // .XO.
      x_points = [(3, 1), (2, 1), (1, 0), (1, 2)];
      o_points = [(0, 1), (1, 1), (2, 0), (2, 2)];
    }
    _ => unreachable!(),
  }

  let (w, h) = if rotation < 2 { (3, 4) } else { (4, 3) };

  let x_offset = offset(width, w, rng);
  let y_offset = offset(height, h, rng);

  let mut result = Vec::new();
  for i in 0..4 {
    result.push((x_offset + x_points[i].0, y_offset + x_points[i].1));
    result.push((x_offset + o_points[i].0, y_offset + o_points[i].1));
  }

  result
}

pub fn opening<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let weigths = WeightedIndex::new([8, 1]).unwrap();

  match weigths.sample(rng) {
    0 => cross(width, height, rng),
    1 => triple_cross(width, height, rng),
    _ => unreachable!(),
  }
}
