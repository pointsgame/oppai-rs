use rand::Rng;
use rand_distr::{Distribution, weighted::WeightedIndex};

const PADDING: u32 = 4;

fn random<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let weigths = WeightedIndex::new([1, 2]).unwrap();
  let counts = [2, 4];

  let count = counts[weigths.sample(rng)];

  let mut result = Vec::new();

  for _ in 0..count {
    let (x, y) = loop {
      let x = rng.random_range(PADDING..width - PADDING);
      let y = rng.random_range(PADDING..height - PADDING);
      if !result.contains(&(x, y)) {
        break (x, y);
      }
    };
    result.push((x, y));
  }

  result
}

fn crosses<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let weigths = WeightedIndex::new([1, 1, 1, 1]).unwrap();
  let counts = [1, 2, 3, 4];

  let count = counts[weigths.sample(rng)];

  let mut result = Vec::new();

  for _ in 0..count {
    let (x, y) = loop {
      let x = rng.random_range(PADDING..width - PADDING - 1);
      let y = rng.random_range(PADDING..height - PADDING - 1);
      if !result
        .iter()
        .any(|&(x_, y_)| x_ == x || y_ == y || x_ == x + 1 || y_ == y + 1)
      {
        break (x, y);
      }
    };
    if rng.random() {
      // XO
      // OX
      result.push((x, y));
      result.push((x + 1, y));
      result.push((x + 1, y + 1));
      result.push((x, y + 1));
    } else {
      // OX
      // XO
      result.push((x + 1, y));
      result.push((x + 1, y + 1));
      result.push((x, y + 1));
      result.push((x, y));
    }
  }

  result
}

fn double_cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let rotation = rng.random_range(0..4);
  let x_points;
  let o_points;

  match rotation {
    0 => {
      // XOOX
      // OXXO
      x_points = [(0, 0), (1, 1), (2, 1), (3, 0)];
      o_points = [(0, 1), (1, 0), (2, 0), (3, 1)];
    }
    1 => {
      // OXXO
      // XOOX
      x_points = [(0, 1), (1, 0), (2, 0), (3, 1)];
      o_points = [(0, 0), (1, 1), (2, 1), (3, 0)];
    }
    2 => {
      // XO
      // OX
      // OX
      // XO
      x_points = [(0, 0), (1, 1), (1, 2), (0, 3)];
      o_points = [(1, 0), (0, 1), (0, 2), (1, 3)];
    }
    3 => {
      // OX
      // XO
      // XO
      // OX
      x_points = [(1, 0), (0, 1), (0, 2), (1, 3)];
      o_points = [(0, 0), (1, 1), (1, 2), (0, 3)];
    }
    _ => unreachable!(),
  }

  let (w, h) = if rotation < 2 { (4, 2) } else { (2, 4) };

  let x_offset = rng.random_range(PADDING..width - PADDING - w + 1);
  let y_offset = rng.random_range(PADDING..height - PADDING - h + 1);

  let mut result = Vec::new();

  for i in 0..4 {
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

  let x_offset = rng.random_range(PADDING..width - PADDING - w + 1);
  let y_offset = rng.random_range(PADDING..height - PADDING - h + 1);

  let mut result = Vec::new();
  for i in 0..4 {
    result.push((x_offset + x_points[i].0, y_offset + x_points[i].1));
    result.push((x_offset + o_points[i].0, y_offset + o_points[i].1));
  }

  result
}

pub fn opening<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let weigths = WeightedIndex::new([1, 8, 4, 2]).unwrap();

  match weigths.sample(rng) {
    0 => random(width, height, rng),
    1 => crosses(width, height, rng),
    2 => double_cross(width, height, rng),
    3 => triple_cross(width, height, rng),
    _ => unreachable!(),
  }
}
