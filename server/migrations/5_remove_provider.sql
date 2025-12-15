-- Remove provider column from oidc_players table

-- First, drop the primary key constraint that includes provider
ALTER TABLE oidc_players DROP CONSTRAINT oidc_players_pkey;

-- Remove the provider column
ALTER TABLE oidc_players DROP COLUMN provider;

-- Add a new primary key constraint without provider
ALTER TABLE oidc_players ADD PRIMARY KEY (subject);

-- Since we're not using provider anymore, we can drop the type
DROP TYPE provider;
