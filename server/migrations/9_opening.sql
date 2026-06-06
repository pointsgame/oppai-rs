CREATE TYPE opening AS ENUM ('cross', 'two_crosses', 'triple_cross');
ALTER TABLE games ADD COLUMN opening opening NOT NULL DEFAULT 'cross';