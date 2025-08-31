use anyhow::Result;
use oppai_field::{field::Field, player::Player};
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> Result<()> {
  let mut rng = SmallRng::from_os_rng();
  let mut field = Field::new_from_rng(20, 20, &mut rng);
  let mut player = Player::Red;
  let mut s = String::new();
  loop {
    s.clear();
    std::io::stdin().read_line(&mut s)?;
    let mut iter = s.trim().split(" ").filter(|s| !s.is_empty()).peekable();
    if iter.peek().is_none() {
      if (field.min_pos() ..= field.max_pos()).any(|pos| field.is_putting_allowed(pos)) {
        anyhow::bail!("field is not fully occupied");
      }
      field.clear();
      player = Player::Red;
      println!();
      continue;
    }
    let x = iter.next().unwrap().parse()?;
    let y = iter.next().unwrap().parse()?;
    let pos = field.to_pos(x, y);
    if !field.put_point(pos, player) {
      anyhow::bail!("invalid position");
    }
    player = player.next();
    println!("{} {}", field.score_red, field.score_black);
  }
}
