#!/usr/bin/env bash

set -Euo pipefail

export RUSTFLAGS="-C target-cpu=native"
export RUST_LOG="oppai_zero=info,oppai_zero_burn=info,oppai_zero_train=info"
export BURN_WGPU_MAX_TASKS=4
BIN=(cargo run --release --quiet --)

EXTENSION=.mpk
MODELS=./models
GAMES=./games
# How many games are played in parallel
PARALLEL_GAMES=2
# How many games are played before training
PLAY_GAMES=64
# How many games are used for training
TRAIN_GAMES=128

mkdir -p "$MODELS"

CHECKPOINT=$(
  fd "model_[\d]+$EXTENSION" $MODELS |
    rg -o -r '$1' "model_([\d]+)$EXTENSION$" |
    awk 'BEGIN { max = -1 } { if ($0 > max) { max = $0; } } END { print max }'
)

if [ "$CHECKPOINT" -eq -1 ]; then
  CHECKPOINT=0
  "${BIN[@]}" init --model "$MODELS/model_$CHECKPOINT" --optimizer "$MODELS/optimizer_$CHECKPOINT"
else
  echo "Resuming from the checkpoint $CHECKPOINT"
fi

while true; do

  for ((i = 1; i <= PLAY_GAMES; i++)); do
    TIMESTAMP=$(date +%s%N)

    mkdir -p "$GAMES/$CHECKPOINT"

    parallel --semaphore -u -j "$PARALLEL_GAMES" "
      echo \"Playing game $i with timestamp $TIMESTAMP\"
      if [ \"$CHECKPOINT\" -eq 0 ]; then
        ${BIN[*]} play --game $GAMES/$CHECKPOINT/$TIMESTAMP.sgf
      else
        ${BIN[*]} play --model $MODELS/model_$CHECKPOINT --game $GAMES/$CHECKPOINT/$TIMESTAMP.sgf
      fi
    "
  done

  parallel --wait

  echo "Training checkpoint $((CHECKPOINT + 1))"
  fd '\d+\.sgf' "$GAMES" |
    sort -rn |
    head -n "$TRAIN_GAMES" |
    xargs "${BIN[@]}" train \
      --model "$MODELS/model_$CHECKPOINT" \
      --optimizer "$MODELS/optimizer_$CHECKPOINT" \
      --model-new "$MODELS/model_$((CHECKPOINT + 1))" \
      --optimizer-new "$MODELS/optimizer_$((CHECKPOINT + 1))" \
      --games

  echo "Pit checkpoint $((CHECKPOINT + 1))"

  ret=0
  "${BIN[@]}" pit --model "$MODELS/model_$CHECKPOINT" --model-new "$MODELS/model_$((CHECKPOINT + 1))" || ret=$?
  if [ "$ret" -eq 0 ]; then
    echo "Accepting the new model"
    ((CHECKPOINT++))
  elif [ "$ret" -eq 2 ]; then
    echo "Rejecting the new model"
    rm "$MODELS/model_$((CHECKPOINT + 1))$EXTENSION"
    rm "$MODELS/optimizer_$((CHECKPOINT + 1))$EXTENSION"
  else
    exit "$ret"
  fi

done
