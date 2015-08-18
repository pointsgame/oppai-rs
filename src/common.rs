use player::Player;
use field::{Pos, Field};

pub fn is_last_move_stupid(field: &Field, pos: Pos, player: Player) -> bool {
  let delta_score = field.get_delta_score(player);
  delta_score < 0 || delta_score == 0 && {
    let enemy = player.next();
    let mut enemies_around = 0u32;
    if field.is_players_point(field.n(pos), enemy) {
      enemies_around += 1;
    }
    if field.is_players_point(field.s(pos), enemy) {
      enemies_around += 1;
    }
    if field.is_players_point(field.w(pos), enemy) {
      enemies_around += 1;
    }
    if field.is_players_point(field.e(pos), enemy) {
      enemies_around += 1;
    }
    enemies_around == 3
  } && {
    field.is_putting_allowed(field.n(pos)) || field.is_putting_allowed(field.s(pos)) || field.is_putting_allowed(field.w(pos)) || field.is_putting_allowed(field.e(pos))
  }
}

pub fn is_penult_move_stuped(field: &Field) -> bool {
  let moves_count = field.moves_count();
  moves_count > 1 && field.is_captured(field.points_seq()[moves_count - 2])
}
