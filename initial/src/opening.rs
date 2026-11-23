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

/// Places a single cross: a 2x2 square with each player's points on one of its
/// diagonals.
///
/// Coordinates are counted from 0, so a point has `0 <= x < width` and
/// `0 <= y < height`. All placements satisfying the rules below are equally
/// likely:
///
/// 1. Direction. The cross takes one of its two directions with equal
///    probability.
/// 2. Placement. The cross's position is chosen independently along each axis,
///    as described in [`offset`]: its center deviates from the field center by
///    at most `width / 6` lines horizontally and `height / 6` lines vertically,
///    with division rounded down, plus half a line on the sides where it can't
///    be centered exactly.
///
/// On 39x32 that puts the cross's points in columns 12 to 26 and rows 10 to 21.
///
/// The cross turns into itself when the field is mirrored and the colors are
/// swapped, so neither player is favored apart from the right to move. Off the
/// center lines the position is mirrored to a different one, but both are
/// equally likely.
pub fn cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
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

/// Places a triple cross: three crosses sharing points, which makes a 3x4 (or
/// 4x3) pattern of 8 points:
///
/// ```text
/// .X.
/// OXO
/// XOX
/// .O.
/// ```
///
/// Coordinates are counted from 0, so a point has `0 <= x < width` and
/// `0 <= y < height`. All placements satisfying the rules below are equally
/// likely:
///
/// 1. Orientation. The pattern takes one of its four distinct orientations
///    with equal probability: the one above, its vertical mirror, and both of
///    them transposed. Mirroring the pattern across its long axis keeps it the
///    same, while mirroring it across its short axis swaps the colors, so these
///    four cover all its rotations and reflections.
/// 2. Placement. The same as for [`cross`], applied to the pattern's bounding
///    box: its center deviates from the field center by at most `width / 6`
///    lines horizontally and `height / 6` lines vertically, plus half a line on
///    the sides where it can't be centered exactly.
///
/// The pattern turns into itself when it's mirrored across its short axis and
/// the colors are swapped, so neither player is favored apart from the right to
/// move.
pub fn triple_cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
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

/// The smallest field side the quadruple cross is played on. Placements exist
/// on smaller fields too, but there the crosses end up only a few king steps
/// apart, so they make a single cluster instead of four separate fights.
pub const QUADRUPLE_CROSS_MIN_SIDE: u32 = 18;

/// Whether the quadruple cross can be played on a field of the given size.
///
/// The longer side is limited to twice the shorter one: valid placements are
/// verified to exist for all such sizes, while on more elongated fields they
/// might not, and then the sampling would never finish.
pub fn quadruple_cross_fits(width: u32, height: u32) -> bool {
  let min = width.min(height);
  let max = width.max(height);
  min >= QUADRUPLE_CROSS_MIN_SIDE && max <= 2 * min
}

/// Makes one attempt to place a quadruple cross, see [`quadruple_cross`] for
/// the rules. Returns `None` if the attempt violates them.
///
/// The rules are checked with integer arithmetic: every inequality is
/// multiplied by a common denominator so that no rounding is involved and the
/// mirrored placements are accepted or rejected exactly alike.
pub(crate) fn try_quadruple_cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Option<Vec<(u32, u32)>> {
  // Top left points of the crosses. Each cross spans two lines in both
  // directions, so the edge margin rule leaves
  // `[side / 4, side - 2 - side / 4]`.
  let x_margin = width / 4;
  let y_margin = height / 4;
  let mut crosses = [(0i64, 0i64); 4];
  for cross in &mut crosses {
    *cross = (
      rng.random_range(x_margin..=width - 2 - x_margin) as i64,
      rng.random_range(y_margin..=height - 2 - y_margin) as i64,
    );
  }

  let w = width as i64;
  let h = height as i64;

  // Center rule: `|sum / 4 - (side - 2) / 2| <= 2 / 25 * side`, multiplied by
  // 100. Top left points are used instead of the centers, both sides of the
  // inequality are shifted by the same half a line.
  let sum_x = crosses.iter().map(|&(x, _)| x).sum::<i64>();
  let sum_y = crosses.iter().map(|&(_, y)| y).sum::<i64>();
  if (25 * sum_x - 50 * (w - 2)).abs() > 8 * w || (25 * sum_y - 50 * (h - 2)).abs() > 8 * h {
    return None;
  }

  // Distance rules compare `distance^2` with `k^2 * width * height`, multiplied
  // by the denominator of `k^2`: `0.225^2 = 81 / 1600`, `0.36^2 = 81 / 625`,
  // `0.54^2 = 729 / 2500`.
  let area = w * h;
  let mut max_distance_2 = 0;
  for i in 0..4 {
    for j in i + 1..4 {
      let distance = (crosses[i].0 - crosses[j].0)
        .abs()
        .max((crosses[i].1 - crosses[j].1).abs());
      let distance_2 = distance * distance;
      if 1600 * distance_2 < 81 * area {
        return None;
      }
      max_distance_2 = max_distance_2.max(distance_2);
    }
  }
  if 625 * max_distance_2 < 81 * area || 2500 * max_distance_2 > 729 * area {
    return None;
  }

  let mut result = Vec::with_capacity(16);
  for (x, y) in crosses {
    let (x, y) = (x as u32, y as u32);
    if rng.random() {
      // XO
      // OX
      result.extend([(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]);
    } else {
      // OX
      // XO
      result.extend([(x, y + 1), (x, y), (x + 1, y), (x + 1, y + 1)]);
    }
  }

  Some(result)
}

/// Places four crosses spread over the central part of the field.
///
/// Coordinates are counted from 0, so a point has `0 <= x < width` and
/// `0 <= y < height`. The position of a cross is its center, which lies
/// between the points. The distance between two crosses is the Chebyshev
/// distance `max(|dx|, |dy|)` between their centers, i.e. the number of king
/// steps. `L = sqrt(width * height)` is the field's scale. All placements
/// satisfying the rules below are equally likely:
///
/// 1. Direction. Each cross takes one of its two directions with equal
///    probability, independently of the other crosses.
/// 2. Edge margin. Every point of every cross satisfies
///    `width / 4 <= x <= width - 1 - width / 4` and
///    `height / 4 <= y <= height - 1 - height / 4`, with division rounded
///    down.
/// 3. Center. The average of the four cross centers deviates from the field
///    center `((width - 1) / 2, (height - 1) / 2)` by at most `0.08 * width`
///    horizontally and `0.08 * height` vertically.
/// 4. Separation. The distance between any two crosses is at least
///    `0.225 * L`.
/// 5. Spread. The largest distance between two crosses lies within
///    `[0.36 * L, 0.54 * L]`.
///
/// On 39x32 that's an edge margin of 9 columns and 8 rows, a center tolerance
/// of 3.12 horizontally and 2.56 vertically, a separation of 8 and a spread of
/// 13 to 19.
///
/// The rules generalize the tournament rules for the 39x32 field. Those used
/// absolute numbers fitted to that size, which leave no valid placements on
/// square fields below 30x30. So here:
///
/// - The edge margin is a fraction of its own side, so the allowed area keeps
///   the field's proportions and the crosses cover the same share of the field
///   on every size. Rounding it down keeps the allowed area from shrinking when
///   the field grows, which with rounding up leaves some sizes (13x13, 14x14,
///   17x17) without valid placements.
/// - The other distances scale with `L`.
/// - Distances are Chebyshev rather than Manhattan or Euclidean. Points connect
///   diagonally, so both connecting two crosses and walling them off from each
///   other take as many moves as their Chebyshev distance. With Chebyshev
///   distance the closest crosses are equally far apart in moves in any
///   direction, while the other metrics let diagonal neighbours come closer
///   in moves than orthogonal ones:
///   - Manhattan distance by up to a factor of 2: with a Manhattan separation
///     of 10, diagonal crosses could be just 5 king steps apart, while
///     orthogonal ones had to be 10 apart.
///   - Euclidean distance by up to a factor of `sqrt(2)`: with a Euclidean
///     separation fitted to 39x32 (8.5), the closest crosses are 9 king steps
///     apart orthogonally but 7 diagonally.
///
///   The cost is that the two closest crosses are rarely diagonal neighbours:
///   they are roughly diagonal in about 0-15% of the placements, depending on
///   the field size, against 15-45% with Euclidean distance. The overall
///   layouts are almost the same: on 39x32 the directions between all pairs of
///   crosses are distributed nearly identically with both metrics. Neither
///   player is favored either way.
/// - The center box is symmetric around the field center: the vertical range
///   in the original rules was shifted by one line.
///
/// The placement is sampled by drawing the crosses uniformly within the edge
/// margin until all the rules hold, which keeps all valid placements equally
/// likely. On fields up to 64x64, 0.08-2% of the draws are accepted, the
/// fewest on elongated fields.
///
/// Must only be called when [`quadruple_cross_fits`] holds.
pub fn quadruple_cross<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  debug_assert!(quadruple_cross_fits(width, height));
  loop {
    if let Some(result) = try_quadruple_cross(width, height, rng) {
      return result;
    }
  }
}

/// Picks a random opening for a field of the given size. Moves alternate
/// between the players starting with the first one, and every opening has an
/// even number of moves, so the first player moves next.
///
/// The opening is a `cross`, a `triple_cross` or a `quadruple_cross`
/// with weights 8 : 1 : 4. The quadruple cross is only played on fields where
/// [`quadruple_cross_fits`] holds; elsewhere the weights are 8 : 1.
pub fn opening<R: Rng>(width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
  let quadruple_weight = if quadruple_cross_fits(width, height) { 4 } else { 0 };
  let weigths = WeightedIndex::new([8, 1, quadruple_weight]).unwrap();

  match weigths.sample(rng) {
    0 => cross(width, height, rng),
    1 => triple_cross(width, height, rng),
    2 => quadruple_cross(width, height, rng),
    _ => unreachable!(),
  }
}
