use std::iter;

use oppai_field::field::{to_pos, to_x, to_y};
use oppai_sgf::{from_coordinate, to_coordinate};
use oppai_zero::episode::Visits;
use sgf_parse::{SgfNode, unknown_game::Prop};

pub fn visits_to_sgf(mut node: &mut SgfNode<Prop>, visits: &[Visits], stride: u32, moves_count: usize) {
  for _ in 0..moves_count - visits.len() {
    node = &mut node.children[0];
  }

  for Visits(visits, full, surprise, value, raw_value) in visits {
    node = &mut node.children[0];

    node.properties.push(Prop::Unknown(
      "ZR".into(),
      iter::once(full.to_string())
        .chain([surprise, value, raw_value].map(|value| value.to_string()))
        .chain(visits.iter().map(|&(pos, visits)| {
          format!(
            "{}{}{}",
            from_coordinate(to_x(stride, pos) as u8) as char,
            from_coordinate(to_y(stride, pos) as u8) as char,
            visits,
          )
        }))
        .collect(),
    ));
  }
}

pub fn sgf_to_visits(node: &SgfNode<Prop>, stride: u32) -> Vec<Visits> {
  node
    .main_variation()
    .flat_map(|node| node.get_property("ZR"))
    .flat_map(|prop| match prop {
      Prop::Unknown(_, visits) => {
        let full = visits[0].parse().unwrap();
        // The policy surprise, search value and raw network value are stored
        // after the full flag. Older self-play data predates some or all of
        // them, so parse greedily and fall back to 0 - visit entries always
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
        Some(Visits(
          rest
            .iter()
            .map(|s| {
              let x = to_coordinate(s.as_bytes()[0]) as u32;
              let y = to_coordinate(s.as_bytes()[1]) as u32;
              let visits = s[2..].parse().unwrap();
              (to_pos(stride, x, y), visits)
            })
            .collect(),
          full,
          surprise,
          value,
          raw_value,
        ))
      }
      _ => None,
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
        (field.field().to_pos(0, 0), 1),
        (field.field().to_pos(0, 1), 2),
        (field.field().to_pos(2, 0), 3),
      ],
      true,
      0.625,
      0.25,
      -0.125,
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, field.field().stride, field.field().moves_count());
    let sgf_visits = sgf_to_visits(&node, field.field().stride);
    assert_eq!(sgf_visits, visits);
  }
}
