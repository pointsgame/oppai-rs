use std::iter;

use oppai_field::field::{to_pos, to_x, to_y};
use oppai_sgf::{from_coordinate, to_coordinate};
use oppai_zero::episode::Visits;
use sgf_parse::{SgfNode, unknown_game::Prop};

pub fn visits_to_sgf(mut node: &mut SgfNode<Prop>, visits: &[Visits], stride: u32, moves_count: usize) {
  for _ in 0..moves_count - visits.len() {
    node = &mut node.children[0];
  }

  for Visits(weights, target_weight, surprise, value, raw_value, q_values) in visits {
    node = &mut node.children[0];

    node.properties.push(Prop::Unknown(
      "ZR".into(),
      iter::once(target_weight.to_string())
        .chain([surprise, value, raw_value].map(|value| value.to_string()))
        .chain(weights.iter().map(|&(pos, weight)| {
          format!(
            "{}{}{}",
            from_coordinate(to_x(stride, pos) as u8) as char,
            from_coordinate(to_y(stride, pos) as u8) as char,
            weight,
          )
        }))
        .collect(),
    ));

    // The per-move q values go into a property of their own rather than the ZR
    // entries, so data recorded without them keeps parsing and the absence
    // stays distinguishable from a search with no explored children.
    if !q_values.is_empty() {
      node.properties.push(Prop::Unknown(
        "ZQ".into(),
        q_values
          .iter()
          .map(|&(pos, weight, q, score)| {
            format!(
              "{}{}{}:{}:{}",
              from_coordinate(to_x(stride, pos) as u8) as char,
              from_coordinate(to_y(stride, pos) as u8) as char,
              weight,
              q,
              score,
            )
          })
          .collect(),
      ));
    }
  }
}

pub fn sgf_to_visits(node: &SgfNode<Prop>, stride: u32) -> Vec<Visits> {
  node
    .main_variation()
    .filter_map(|node| {
      let visits = match node.get_property("ZR") {
        Some(Prop::Unknown(_, visits)) => visits,
        _ => return None,
      };
      // The training weight of the row. Data recorded when this was a boolean
      // full-search flag loads as weight 1 or 0, which is exactly what those
      // searches weighed.
      let target_weight = match visits[0].as_str() {
        "true" => 1.0,
        "false" => 0.0,
        weight => weight.parse().unwrap(),
      };
      // The policy surprise, search value and raw network value are stored
      // after the training weight. Older self-play data predates some or all
      // of them, so parse greedily and fall back to 0 - weight entries always
      // start with a coordinate letter and so never parse as a float.
      let mut numbers = [0.0f64; 3];
      let mut rest = &visits[1..];
      for number in &mut numbers {
        if let Some(Ok(value)) = rest.first().map(|s| s.parse::<f64>()) {
          *number = value;
          rest = &rest[1..];
        } else {
          break;
        }
      }
      let [surprise, value, raw_value] = numbers;
      // Data recorded before per-move q values were stored has no ZQ property
      // and loads with an empty list, which the training loss masks out.
      let q_values = match node.get_property("ZQ") {
        Some(Prop::Unknown(_, entries)) => entries
          .iter()
          .filter_map(|s| {
            let x = to_coordinate(s.as_bytes()[0]) as u32;
            let y = to_coordinate(s.as_bytes()[1]) as u32;
            let (weight, rest) = s[2..].split_once(':')?;
            // Entries recorded before the per-move score was stored have no
            // second separator. They are dropped rather than given a made-up
            // score: the loss has a single weight for both targets, so a zero
            // score would be trained towards as if the search had settled on
            // it.
            let (q, score) = rest.split_once(':')?;
            Some((
              to_pos(stride, x, y),
              weight.parse().unwrap(),
              q.parse().unwrap(),
              score.parse().unwrap(),
            ))
          })
          .collect(),
        _ => Vec::new(),
      };
      Some(Visits(
        rest
          .iter()
          .map(|s| {
            let x = to_coordinate(s.as_bytes()[0]) as u32;
            let y = to_coordinate(s.as_bytes()[1]) as u32;
            // Search weights are fractional. Data predating uncertainty
            // weighting stores integer visit counts, which parse as the
            // weights they are - one per playout.
            let weight = s[2..].parse().unwrap();
            (to_pos(stride, x, y), weight)
          })
          .collect(),
        target_weight,
        surprise,
        value,
        raw_value,
        q_values,
      ))
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use oppai_field::{any_field::AnyField, construct_field::construct_field, extended_field::ExtendedField};
  use oppai_sgf::to_sgf;
  use oppai_zero::episode::Visits;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256PlusPlus;
  use sgf_parse::unknown_game::Prop;

  use super::{sgf_to_visits, visits_to_sgf};

  const SEED: u64 = 7;

  #[test]
  fn save_load_visits() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let visits = vec![Visits(
      vec![
        (field.field().to_pos(0, 0), 1.5),
        (field.field().to_pos(0, 1), 2.25),
        (field.field().to_pos(2, 0), 24.0),
      ],
      1.0,
      0.625,
      0.25,
      -0.125,
      vec![
        (field.field().to_pos(0, 0), 1.5, 0.5, 3.5),
        (field.field().to_pos(2, 0), 24.0, -0.75, -1.5),
      ],
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, field.field().stride, field.field().moves_count());
    let sgf_visits = sgf_to_visits(&node, field.field().stride);
    assert_eq!(sgf_visits, visits);
  }

  /// Data recorded when the q entries carried only the value has no per-move
  /// score to load. Such entries are dropped rather than given a made-up score,
  /// since one weight covers both targets.
  #[test]
  fn load_q_values_recorded_without_scores() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    let visits = vec![Visits(
      vec![(field.field().to_pos(0, 0), 1.5)],
      1.0,
      0.625,
      0.25,
      -0.125,
      vec![(field.field().to_pos(0, 0), 1.5, 0.5, 3.5)],
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, stride, field.field().moves_count());
    let mut sgf_node = &mut node;
    loop {
      for property in &mut sgf_node.properties {
        if let Prop::Unknown(name, values) = property
          && name == "ZQ"
        {
          // The old encoding: coordinates, weight and q value, no score.
          *values = vec!["aa1.5:0.5".to_string()];
        }
      }
      match sgf_node.children.first_mut() {
        Some(child) => sgf_node = child,
        None => break,
      }
    }
    let mut old = visits;
    old[0].5.clear();
    assert_eq!(sgf_to_visits(&node, stride), old);
  }

  /// Self-play data recorded before uncertainty weighting stores integer visit
  /// counts where the weights now go. Those are weights with every playout
  /// weighing one, so they must still load - and normalize to the same policy
  /// target they always did.
  #[test]
  fn load_visits_recorded_as_integers() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    // Legacy data also predates the per-move q values, so they stay empty -
    // and an empty list must not write the property at all.
    let integer = vec![Visits(
      vec![
        (field.field().to_pos(0, 0), 1.0),
        (field.field().to_pos(0, 1), 2.0),
        (field.field().to_pos(2, 0), 3.0),
      ],
      1.0,
      0.625,
      0.25,
      -0.125,
      Vec::new(),
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &integer, stride, field.field().moves_count());
    // Whole weights serialize without a decimal point, exactly as the old
    // integer format did, so this is byte-for-byte the legacy encoding.
    let property = node
      .main_variation()
      .flat_map(|node| node.get_property("ZR"))
      .next()
      .unwrap();
    let Prop::Unknown(_, values) = property else {
      panic!("expected an unknown ZR property");
    };
    assert_eq!(&values[4..], &["aa1", "ab2", "ca3"]);
    assert!(node.main_variation().all(|node| node.get_property("ZQ").is_none()));
    assert_eq!(sgf_to_visits(&node, stride), integer);
  }

  /// Data recorded when the training weight was a boolean full-search flag
  /// stores `true`/`false` where the weight now goes. A full search weighed 1
  /// and a cheap one 0, so that is what the flags load as.
  #[test]
  fn load_visits_recorded_with_full_flag() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    let visits = vec![Visits(
      vec![(field.field().to_pos(0, 0), 1.5)],
      1.0,
      0.625,
      0.25,
      -0.125,
      Vec::new(),
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, stride, field.field().moves_count());
    let mut sgf_node = &mut node;
    loop {
      for property in &mut sgf_node.properties {
        if let Prop::Unknown(name, values) = property
          && name == "ZR"
        {
          // The old encoding: a boolean flag where the weight now goes.
          values[0] = "true".to_string();
        }
      }
      match sgf_node.children.first_mut() {
        Some(child) => sgf_node = child,
        None => break,
      }
    }
    assert_eq!(sgf_to_visits(&node, stride), visits);
  }
}
