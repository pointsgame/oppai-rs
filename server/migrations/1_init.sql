CREATE TABLE IF NOT EXISTS players (
  id uuid PRIMARY KEY,
  nickname text UNIQUE NOT NULL,
  registration_time timestamp NOT NULL
);

CREATE TYPE provider AS ENUM ('google', 'gitlab');

CREATE TABLE IF NOT EXISTS oidc_players (
  player_id uuid NOT NULL REFERENCES players (id),
  provider provider,
  subject text,
  email text,
  email_verified boolean,
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

CREATE OR REPLACE FUNCTION unique_nickname(p_nickname text)
  RETURNS text AS
$$
DECLARE
  new_name text;
  counter integer;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM players WHERE nickname = p_nickname) THEN
    RETURN p_nickname;
  END IF;
  counter := 2;
  LOOP
    new_name := p_nickname || '_' || counter;
    IF NOT EXISTS (SELECT 1 FROM players WHERE nickname = new_name) THEN
      RETURN new_name;
    END IF;
    counter := counter + 1;
  END LOOP;
END;
$$ LANGUAGE plpgsql;
