use crate::opening::{
  QUADRUPLE_CROSS_MIN_SIDE, cross, opening, quadruple_cross, quadruple_cross_fits, triple_cross, try_quadruple_cross,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashSet;

type Pattern = fn(u32, u32, &mut Xoshiro256PlusPlus) -> Vec<(u32, u32)>;

/// All the cells the pattern can occupy on a field of the given size.
fn occupied(width: u32, height: u32, pattern: Pattern) -> HashSet<(u32, u32)> {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let mut result = HashSet::new();

  for _ in 0..100_000 {
    let moves = pattern(width, height, &mut rng);
    assert!(!moves.is_empty());
    for (x, y) in moves {
      assert!(x < width && y < height);
      result.insert((x, y));
    }
  }

  result
}

#[test]
fn symmetrical() {
  for (width, height) in [(10, 10), (11, 11), (10, 11), (39, 32), (30, 40)] {
    let cells = occupied(width, height, opening);
    for &(x, y) in &cells {
      assert!(
        cells.contains(&(width - 1 - x, y)),
        "{width}x{height}: ({x}, {y}) has no horizontal mirror"
      );
      assert!(
        cells.contains(&(x, height - 1 - y)),
        "{width}x{height}: ({x}, {y}) has no vertical mirror"
      );
    }
  }
}

#[test]
fn deviation() {
  for ((width, height), pattern) in [(10, 10), (11, 11), (10, 11), (39, 32), (30, 40)]
    .into_iter()
    .flat_map(|size| [(size, cross as Pattern), (size, triple_cross as Pattern)])
  {
    let cells = occupied(width, height, pattern);
    // A sixth of the field size in either direction from the center, the size
    // of the biggest pattern (4) and one more cell for the odd free space.
    let x_spread = 2 * (width / 6) + 4 + 1;
    let y_spread = 2 * (height / 6) + 4 + 1;
    let min_x = cells.iter().map(|&(x, _)| x).min().unwrap();
    let max_x = cells.iter().map(|&(x, _)| x).max().unwrap();
    let min_y = cells.iter().map(|&(_, y)| y).min().unwrap();
    let max_y = cells.iter().map(|&(_, y)| y).max().unwrap();
    assert!(max_x - min_x + 1 <= x_spread, "{width}x{height}: {min_x}..={max_x}");
    assert!(max_y - min_y + 1 <= y_spread, "{width}x{height}: {min_y}..={max_y}");
  }
}

#[test]
fn quadruple_cross_symmetrical() {
  for (width, height) in [(18, 18), (19, 18), (39, 32), (30, 40)] {
    let cells = occupied(width, height, quadruple_cross);
    for &(x, y) in &cells {
      assert!(
        cells.contains(&(width - 1 - x, y)),
        "{width}x{height}: ({x}, {y}) has no horizontal mirror"
      );
      assert!(
        cells.contains(&(x, height - 1 - y)),
        "{width}x{height}: ({x}, {y}) has no vertical mirror"
      );
    }
  }
}

#[test]
fn quadruple_cross_size_limits() {
  assert!(!quadruple_cross_fits(
    QUADRUPLE_CROSS_MIN_SIDE - 1,
    QUADRUPLE_CROSS_MIN_SIDE - 1
  ));
  assert!(!quadruple_cross_fits(QUADRUPLE_CROSS_MIN_SIDE - 1, 30));
  assert!(quadruple_cross_fits(QUADRUPLE_CROSS_MIN_SIDE, QUADRUPLE_CROSS_MIN_SIDE));
  assert!(quadruple_cross_fits(18, 36));
  assert!(!quadruple_cross_fits(18, 37));
  assert!(quadruple_cross_fits(39, 32));

  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  for _ in 0..10_000 {
    assert!(opening(17, 17, &mut rng).len() < 16);
  }
}

/// Checks the rules from the `quadruple_cross` documentation directly, in
/// floating point and on cross centers, independently of the integer
/// arithmetic used for sampling.
fn check_quadruple_cross(width: u32, height: u32, moves: &[(u32, u32)]) {
  assert_eq!(moves.len(), 16);
  assert_eq!(moves.iter().collect::<HashSet<_>>().len(), 16);

  let (w, h) = (width as f64, height as f64);
  let scale = (w * h).sqrt();
  let mut centers = Vec::new();
  for cross in moves.chunks(4) {
    let min_x = cross.iter().map(|&(x, _)| x).min().unwrap();
    let min_y = cross.iter().map(|&(_, y)| y).min().unwrap();
    // A 2x2 square with the first player on one diagonal and the second on the
    // other one. Moves alternate between the players.
    for (i, &(x, y)) in cross.iter().enumerate() {
      assert!(x - min_x <= 1 && y - min_y <= 1, "{cross:?}");
      let diagonal = (x - min_x) == (y - min_y);
      let first_diagonal = (cross[0].0 - min_x) == (cross[0].1 - min_y);
      assert_eq!(diagonal == first_diagonal, i % 2 == 0, "{cross:?}");
      // Edge margin.
      assert!(
        x >= width / 4 && x <= width - 1 - width / 4,
        "{width}x{height}: {cross:?}"
      );
      assert!(
        y >= height / 4 && y <= height - 1 - height / 4,
        "{width}x{height}: {cross:?}"
      );
    }
    centers.push((min_x as f64 + 0.5, min_y as f64 + 0.5));
  }

  // Center.
  let mean_x = centers.iter().map(|c| c.0).sum::<f64>() / 4.0;
  let mean_y = centers.iter().map(|c| c.1).sum::<f64>() / 4.0;
  assert!(
    (mean_x - (w - 1.0) / 2.0).abs() <= 0.08 * w + 1e-9,
    "{width}x{height}: {centers:?}"
  );
  assert!(
    (mean_y - (h - 1.0) / 2.0).abs() <= 0.08 * h + 1e-9,
    "{width}x{height}: {centers:?}"
  );

  // Separation and spread.
  let mut max_distance: f64 = 0.0;
  for i in 0..4 {
    for j in i + 1..4 {
      let distance = (centers[i].0 - centers[j].0)
        .abs()
        .max((centers[i].1 - centers[j].1).abs());
      assert!(distance >= 0.225 * scale - 1e-9, "{width}x{height}: {centers:?}");
      max_distance = max_distance.max(distance);
    }
  }
  assert!(max_distance >= 0.36 * scale - 1e-9, "{width}x{height}: {centers:?}");
  assert!(max_distance <= 0.54 * scale + 1e-9, "{width}x{height}: {centers:?}");
}

#[test]
fn quadruple_cross_rules() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  for (width, height) in [
    (18, 18),
    (19, 23),
    (20, 20),
    (24, 18),
    (36, 18),
    (39, 32),
    (32, 39),
    (50, 50),
    (60, 30),
  ] {
    for _ in 0..1000 {
      check_quadruple_cross(width, height, &quadruple_cross(width, height, &mut rng));
    }
  }
}

/// The closest crosses on 39x32 are exactly 8 king steps apart, and that
/// happens both orthogonally and diagonally.
#[test]
fn quadruple_cross_separation() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let mut closest = u32::MAX;
  let mut closest_orthogonal = false;
  let mut closest_diagonal = false;
  for _ in 0..100_000 {
    let moves = quadruple_cross(39, 32, &mut rng);
    let corners = moves
      .chunks(4)
      .map(|cross| {
        (
          cross.iter().map(|&(x, _)| x).min().unwrap(),
          cross.iter().map(|&(_, y)| y).min().unwrap(),
        )
      })
      .collect::<Vec<_>>();
    for i in 0..4 {
      for j in i + 1..4 {
        let dx = corners[i].0.abs_diff(corners[j].0);
        let dy = corners[i].1.abs_diff(corners[j].1);
        let distance = dx.max(dy);
        closest = closest.min(distance);
        if distance == 8 {
          closest_orthogonal |= dx == 0 || dy == 0;
          closest_diagonal |= dx == dy;
        }
      }
    }
  }
  assert_eq!(closest, 8);
  assert!(closest_orthogonal && closest_diagonal);
}

/// Valid placements exist on every field the quadruple cross is played on, so
/// the sampling always finishes, and they are found quickly.
#[test]
fn quadruple_cross_feasible() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  for width in QUADRUPLE_CROSS_MIN_SIDE..=64 {
    for height in QUADRUPLE_CROSS_MIN_SIDE..=64 {
      if !quadruple_cross_fits(width, height) {
        continue;
      }
      let attempts = (1..=1_000_000)
        .find(|_| try_quadruple_cross(width, height, &mut rng).is_some())
        .unwrap_or_else(|| panic!("{width}x{height}: no placement found"));
      // Not a hard bound, but with the acceptance rate of at least 0.08% even
      // the worst size out of ~1700 shouldn't take more than ~20000 attempts.
      assert!(attempts <= 30_000, "{width}x{height}: {attempts} attempts");
    }
  }
}

/// The example from the `cross` documentation.
#[test]
fn cross_range() {
  let cells = occupied(39, 32, cross);
  assert_eq!(cells.iter().map(|&(x, _)| x).min(), Some(12));
  assert_eq!(cells.iter().map(|&(x, _)| x).max(), Some(26));
  assert_eq!(cells.iter().map(|&(_, y)| y).min(), Some(10));
  assert_eq!(cells.iter().map(|&(_, y)| y).max(), Some(21));
}
