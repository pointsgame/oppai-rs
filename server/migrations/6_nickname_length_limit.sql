-- Limit nickname length to 32 characters

-- Change nickname column type from TEXT to VARCHAR(32)
ALTER TABLE players ALTER COLUMN nickname TYPE VARCHAR(32);

-- Update the unique_nickname function to handle the new length constraint
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
    IF LENGTH(new_name) > 32 THEN
      base_name := SUBSTRING(base_name FROM 1 FOR 32 - LENGTH('_' || counter));
      new_name := base_name || '_' || counter;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM players WHERE nickname = new_name) THEN
      RETURN new_name;
    END IF;
    counter := counter + 1;
  END LOOP;
END;
$$ LANGUAGE plpgsql;
