CREATE TABLE IF NOT EXISTS players (
  id UUID PRIMARY KEY,
  oidc_subject TEXT,
  oidc_email TEXT,
  oidc_nickname TEXT,
  oidc_name TEXT,
  oidc_preferred_username TEXT,
  registration_time TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
  id UUID PRIMARY KEY,
  red_player_id UUID NOT NULL REFERENCES players (id),
  black_player_id UUID NOT NULL REFERENCES players (id),
  start_time TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS moves (
  game_id UUID NOT NULL REFERENCES games (id),
  player_id NOT NULL REFERENCES players (id),
  "number" SMALLINT NOT NULL,
  x SMALLINT NOT NULL,
  y SMALLINT NOT NULL,
  putting_time TIMESTAMP NOT NULL,
  PRIMARY KEY (game_id, "number")
);
