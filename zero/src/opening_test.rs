use crate::opening::opening;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashSet;

/// All the cells the openings can occupy on a field of the given size.
fn occupied(width: u32, height: u32) -> HashSet<(u32, u32)> {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let mut result = HashSet::new();

  for _ in 0..100_000 {
    let moves = opening(width, height, &mut rng);
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
    let cells = occupied(width, height);
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
  for (width, height) in [(10, 10), (11, 11), (10, 11), (39, 32), (30, 40)] {
    let cells = occupied(width, height);
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
