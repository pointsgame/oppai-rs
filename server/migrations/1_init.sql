CREATE TABLE IF NOT EXISTS players (
  id uuid PRIMARY KEY,
  registration_time timestamp NOT NULL
);

CREATE TYPE provider AS ENUM ('google');

CREATE TABLE IF NOT EXISTS oidc_players (
  player_id uuid NOT NULL REFERENCES players (id),
  provider provider,
  subject text,
  email text,
  "name" text,
  nickname text,
  preferred_username text,
  PRIMARY KEY (subject, provider)
);

CREATE INDEX oidc_email ON oidc_players USING HASH (email);

CREATE TABLE IF NOT EXISTS games (
  id uuid PRIMARY KEY,
  red_player_id uuid NOT NULL REFERENCES players (id),
  black_player_id uuid NOT NULL REFERENCES players (id),
  start_time timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS moves (
  game_id uuid NOT NULL REFERENCES games (id),
  player_id uuid NOT NULL REFERENCES players (id),
  "number" smallint NOT NULL,
  x smallint NOT NULL,
  y smallint NOT NULL,
  putting_time timestamp NOT NULL,
  PRIMARY KEY (game_id, "number")
);
