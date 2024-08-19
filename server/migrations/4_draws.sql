CREATE TYPE color AS ENUM ('red', 'black');

ALTER TABLE moves
RENAME putting_time TO "timestamp";

ALTER TABLE moves
ADD COLUMN player color NOT NULL DEFAULT 'red';

UPDATE moves
SET player = 'black'
FROM games
WHERE moves.game_id = games.id
AND moves.player_id = games.black_player_id;

ALTER TABLE moves
ALTER COLUMN player DROP DEFAULT;

ALTER TABLE moves
DROP COLUMN player_id;

CREATE TABLE IF NOT EXISTS draw_offers (
  game_id uuid NOT NULL REFERENCES games (id),
  player color NOT NULL,
  offer boolean,
  "timestamp" timestamp NOT NULL
);
