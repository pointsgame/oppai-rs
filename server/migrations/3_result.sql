CREATE TYPE gameresult AS ENUM ('resignedred', 'resignedblack', 'groundedred', 'groundedblack', 'timeoutred', 'timeoutblack', 'drawagreement', 'drawgrounded');

ALTER TABLE games
ADD COLUMN "result" gameresult,
ADD COLUMN finish_time timestamp;
