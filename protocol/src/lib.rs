use oppai_field::player::Player;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DurationMilliSeconds};
use std::time::Duration;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Coords {
  pub x: u32,
  pub y: u32,
}

#[serde_as]
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Constraint {
  Time(#[serde_as(as = "DurationMilliSeconds")] Duration),
  Complexity(f64),
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum Request {
  Init { width: u32, height: u32 },
  PutPoint { coords: Coords, player: Player },
  Undo,
  Analyze { player: Player, constraint: Constraint },
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Move {
  pub coords: Coords,
  pub weight: f64,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum Response {
  Init,
  PutPoint { put: bool },
  Undo { undone: bool },
  Analyze { moves: Vec<Move> },
}

#[cfg(test)]
mod tests {
  use super::*;

  macro_rules! from_to_json_test {
    ($name:ident, $type:ty, $value:expr, $string:expr) => {
      #[test]
      fn $name() {
        let json = serde_json::to_string(&$value).unwrap();
        assert_eq!(json, $string);
        let parsed: $type = serde_json::from_str($string).unwrap();
        assert_eq!(parsed, $value);
      }
    };
  }

  from_to_json_test!(
    init_request,
    Request,
    Request::Init { width: 39, height: 32 },
    r#"{"command":"Init","width":39,"height":32}"#
  );

  from_to_json_test!(
    put_point_request,
    Request,
    Request::PutPoint {
      coords: Coords { x: 1, y: 2 },
      player: Player::Red
    },
    r#"{"command":"PutPoint","coords":{"x":1,"y":2},"player":"Red"}"#
  );

  from_to_json_test!(undo_request, Request, Request::Undo, r#"{"command":"Undo"}"#);

  from_to_json_test!(
    analyze_with_time_request,
    Request,
    Request::Analyze {
      player: Player::Red,
      constraint: Constraint::Time(Duration::from_secs(7)),
    },
    r#"{"command":"Analyze","player":"Red","constraint":{"type":"Time","value":7000}}"#
  );

  from_to_json_test!(
    analyze_with_complexity_request,
    Request,
    Request::Analyze {
      player: Player::Red,
      constraint: Constraint::Complexity(1.0),
    },
    r#"{"command":"Analyze","player":"Red","constraint":{"type":"Complexity","value":1.0}}"#
  );

  from_to_json_test!(init_response, Response, Response::Init, r#"{"command":"Init"}"#);

  from_to_json_test!(
    put_point_response,
    Response,
    Response::PutPoint { put: true },
    r#"{"command":"PutPoint","put":true}"#
  );

  from_to_json_test!(
    undo_response,
    Response,
    Response::Undo { undone: true },
    r#"{"command":"Undo","undone":true}"#
  );

  from_to_json_test!(
    analyze_response,
    Response,
    Response::Analyze {
      moves: vec![Move {
        coords: Coords { x: 1, y: 2 },
        weight: 1.0
      }]
    },
    r#"{"command":"Analyze","moves":[{"coords":{"x":1,"y":2},"weight":1.0}]}"#
  );
}
