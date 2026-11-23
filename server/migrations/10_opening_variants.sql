-- Two crosses is no longer offered. The cast fails if any game still uses it.
ALTER TYPE opening RENAME TO opening_old;
CREATE TYPE opening AS ENUM ('cross', 'triple_cross', 'quadruple_cross');
ALTER TABLE games ALTER COLUMN opening DROP DEFAULT;
ALTER TABLE games ALTER COLUMN opening TYPE opening USING opening::text::opening;
ALTER TABLE games ALTER COLUMN opening SET DEFAULT 'cross';
DROP TYPE opening_old;
