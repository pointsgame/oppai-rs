#!/usr/bin/env bash

set -Euo pipefail

export RUSTFLAGS="-C target-cpu=native"
BIN=(cargo run --release --quiet --)

MODELS=./models
GAMES=./games
SGFS=./sgfs
# How many games are played in parallel
PARALLEL_GAMES=8
# How many games are played before training
PLAY_GAMES=16
# How many games are used for training
TRAIN_GAMES=32

mkdir -p "$MODELS"

CHECKPOINT=$(
  fd 'model_[\d]+.pt' $MODELS |
    rg -o -r '$1' 'model_([\d]+).pt$' |
    awk 'BEGIN { max = -1 } { if ($0 > max) { max = $0; } } END { print max }'
)

if [ "$CHECKPOINT" -eq -1 ]; then
  CHECKPOINT=0
  "${BIN[@]}" init --model "$MODELS/model_$CHECKPOINT.pt"
else
  echo "Resuming from the checkpoint $CHECKPOINT"
fi

while true; do

  for ((i = 1; i <= PLAY_GAMES; i++)); do
    TIMESTAMP=$(date +%s%N)

    mkdir -p "$GAMES/$CHECKPOINT"
    mkdir -p "$SGFS/$CHECKPOINT"

    parallel --semaphore -u -j "$PARALLEL_GAMES" "
    echo \"Playing game $i with timestamp $TIMESTAMP\"
    ${BIN[*]} play --model $MODELS/model_$CHECKPOINT.pt --game $GAMES/$CHECKPOINT/$TIMESTAMP.cbor --sgf $SGFS/$CHECKPOINT/$TIMESTAMP.sgf
  "
  done

  parallel --wait

  echo "Training checkpoint $((CHECKPOINT + 1))"
  fd '\d+\.cbor' "$GAMES" |
    sort -rn |
    head -n "$TRAIN_GAMES" |
    xargs "${BIN[@]}" train --model "$MODELS/model_$CHECKPOINT.pt" --model-new "$MODELS/model_$((CHECKPOINT + 1)).pt" --games

  echo "Pit checkpoint $((CHECKPOINT + 1))"

  ret=0
  "${BIN[@]}" pit --model "$MODELS/model_$CHECKPOINT.pt" --model-new "$MODELS/model_$((CHECKPOINT + 1)).pt" || ret=$?
  if [ "$ret" -eq 0 ]; then
    echo "Accepting the new model"
    ((CHECKPOINT++))
  elif [ "$ret" -eq 2 ]; then
    echo "Rejecting the new model"
    rm "$MODELS/model_$((CHECKPOINT + 1)).pt"
  else
    exit "$ret"
  fi

done
