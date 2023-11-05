use crate::{from_coordinate, to_coordinate};
use oppai_field::field::{to_pos, to_x, to_y};
use oppai_zero::episode::Visits;
use sgf_parse::{unknown_game::Prop, SgfNode};

pub fn visits_to_sgf(mut node: &mut SgfNode<Prop>, visits: &[Visits], width: u32, moves_count: usize) {
  for _ in 0..moves_count - visits.len() {
    node = &mut node.children[0];
  }

  for Visits(visits) in visits {
    node = &mut node.children[0];

    node.properties.push(Prop::Unknown(
      "ZR".into(),
      visits
        .iter()
        .map(|&(pos, visits)| {
          format!(
            "{}{}{}",
            from_coordinate(to_x(width, pos) as u8) as char,
            from_coordinate(to_y(width, pos) as u8) as char,
            visits,
          )
        })
        .collect(),
    ));
  }
}

pub fn sgf_to_visits(node: &SgfNode<Prop>, width: u32) -> Vec<Visits> {
  node
    .main_variation()
    .flat_map(|node| node.get_property("ZR"))
    .flat_map(|prop| match prop {
      Prop::Unknown(_, visits) => Some(Visits(
        visits
          .iter()
          .map(|s| {
            let x = to_coordinate(s.as_bytes()[0]) as u32;
            let y = to_coordinate(s.as_bytes()[1]) as u32;
            let visits = s[2..].parse().unwrap();
            (to_pos(width, x, y), visits)
          })
          .collect(),
      )),
      _ => None,
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use crate::to_sgf;
  use oppai_field::{any_field::AnyField, construct_field::construct_field, extended_field::ExtendedField};
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
    let visits = vec![Visits(vec![
      (field.field().to_pos(0, 0), 1),
      (field.field().to_pos(0, 1), 2),
      (field.field().to_pos(2, 0), 3),
    ])];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, field.field().width(), field.field().moves_count());
    let sgf_visits = sgf_to_visits(&node, field.field().width());
    assert_eq!(sgf_visits, visits);
  }
}
