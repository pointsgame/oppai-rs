#!/usr/bin/env bash

set -euo pipefail

# Seeds a new run of synchronous_loop.sh with the most recent games of another
# run, so that a new network - a bigger one, say - catches up on them instead of
# starting from random games.
#
# The games are copied as the new run's first cycles, oldest first, and the loop
# then goes through them as if it had played them: a cycle that already has its
# games only trains, with the learning rate warming up and the weights averaged
# from the new run's start. Once the copied cycles are used up, the network
# plays its own games. The first cycle has to hold MIN_POSITIONS positions, or
# the loop would top it up with random games, so the oldest of the copied files
# are merged into it as needed. Every other file must hold at least the loop's
# NUM_GAMES_PER_CYCLE games, or the loop would top it up with games of the new
# network: the old run's latest file is left out while it is still being
# played, the earlier ones are complete.
#
# Only the most recent games are worth carrying over: the oldest ones were
# played by much weaker networks, and their targets lead away from good play.
# But a fresh network needs its whole learning rate warmup - 2M trained
# positions - and more, and training that much on too few positions repeats
# them often enough to memorize them. The plan printed tells how much the loop
# will train on what is carried over; a few passes over each position is about
# right.

usage() {
	echo "Usage: $0 [-n] OLD_BASEDIR NEW_BASEDIR POSITIONS"
	echo "Copies the games of OLD_BASEDIR's most recent cycles, back until they hold POSITIONS positions, as the first cycles of NEW_BASEDIR"
	echo "-n only prints the plan"
	echo "The loop's NUM_GAMES_PER_CYCLE, MIN_POSITIONS, TRAININGS_PER_POSITION, MAX_TRAINING_POSITIONS, WINDOW_EXPAND, WINDOW_EXPONENT and WINDOW_SCALE can be overridden from the environment, and should be when the loop's differ"
	exit 1
}

DRY_RUN=0
if [[ "${1:-}" == -n ]]; then
	DRY_RUN=1
	shift
fi
(($# == 3)) || usage
OLD_GAMESDIR="$(realpath "$1")"/games
NEW_BASEDIR="$2"
NEW_GAMESDIR="$NEW_BASEDIR"/games
WANTED="$3"

# As in synchronous_loop.sh, which the plan follows.
NUM_GAMES_PER_CYCLE="${NUM_GAMES_PER_CYCLE:-1024}"
MIN_POSITIONS="${MIN_POSITIONS:-100000}"
TRAININGS_PER_POSITION="${TRAININGS_PER_POSITION:-8}"
MAX_TRAINING_POSITIONS="${MAX_TRAINING_POSITIONS:-500000}"
WINDOW_EXPAND="${WINDOW_EXPAND:-0.4}"
WINDOW_EXPONENT="${WINDOW_EXPONENT:-0.65}"
WINDOW_SCALE="${WINDOW_SCALE:-50000}"

GITROOTDIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BIN="${BIN:-"$GITROOTDIR"/target/release/oppai-zero-train}"

# Positions of a games file, from the cache the loop keeps next to it when it is
# up to date.
positions() {
	local file="$1" cache="${1%.sgf.gz}.positions" size
	size=$(stat -c %s "$file")
	if [[ -e "$cache" ]] && [[ "$(cut -d' ' -f1 "$cache")" == "$size" ]]; then
		cut -d' ' -f2 "$cache"
	else
		"$BIN" count --games "$file" | sed -n 's/^Games: [0-9]*; examples: \([0-9]*\).*/\1/p'
	fi
}

# The same as in synchronous_loop.sh.
window_positions() {
	awk -v n="$1" -v m="$MIN_POSITIONS" -v e="$WINDOW_EXPAND" -v p="$WINDOW_EXPONENT" -v s="$WINDOW_SCALE" 'BEGIN {
		if (n <= m) { printf "%d", m; exit }
		x = n - m + s
		printf "%d", m + e * (x ^ p - s ^ p) / (p * s ^ (p - 1))
	}'
}

if [[ "$DRY_RUN" == 0 ]] && [[ -e "$NEW_GAMESDIR" ]] && [[ -n "$(ls -A "$NEW_GAMESDIR")" ]]; then
	echo "$NEW_GAMESDIR is not empty" >&2
	exit 1
fi

# The old run's cycles, newest first, back until they hold the positions wanted.
LAST=-1
for f in "$OLD_GAMESDIR"/games_*.sgf.gz; do
	n="${f##*/games_}"
	n="${n%.sgf.gz}"
	[[ "$n" =~ ^[0-9]+$ ]] && ((n > LAST)) && LAST=$n
done
((LAST >= 0)) || {
	echo "No games in $OLD_GAMESDIR" >&2
	exit 1
}
GAMES=$("$BIN" count --games "$OLD_GAMESDIR"/games_"$LAST".sgf.gz | sed -n 's/^Games: \([0-9]*\);.*/\1/p')
if ((GAMES < NUM_GAMES_PER_CYCLE)); then
	echo "Leaving out games_$LAST, which has only $GAMES games so far"
	LAST=$((LAST - 1))
fi
FILES=()
COUNTS=()
TOTAL=0
for ((i = LAST; i >= 0 && TOTAL < WANTED; i--)); do
	f="$OLD_GAMESDIR"/games_"$i".sgf.gz
	[[ -e "$f" ]] || continue
	count=$(positions "$f")
	FILES=("$f" "${FILES[@]}")
	COUNTS=("$count" "${COUNTS[@]}")
	TOTAL=$((TOTAL + count))
done

# The new run's cycles: the oldest files merged into the first one until it has
# MIN_POSITIONS positions, then one file each.
CYCLE_FILES=("${FILES[0]}")
CYCLE_COUNTS=("${COUNTS[0]}")
for ((i = 1; i < ${#FILES[@]}; i++)); do
	if ((${#CYCLE_FILES[@]} == 1 && CYCLE_COUNTS[0] < MIN_POSITIONS)); then
		CYCLE_FILES[0]+=" ${FILES[i]}"
		CYCLE_COUNTS[0]=$((CYCLE_COUNTS[0] + COUNTS[i]))
	else
		CYCLE_FILES+=("${FILES[i]}")
		CYCLE_COUNTS+=("${COUNTS[i]}")
	fi
done
if ((CYCLE_COUNTS[0] < MIN_POSITIONS)); then
	echo "Only $TOTAL positions in $OLD_GAMESDIR, fewer than MIN_POSITIONS" >&2
	exit 1
fi

# What the loop will train on each of them, the way it decides it.
printf '%-6s %-40s %10s %10s %10s %12s\n' cycle from positions window trained "trained sum"
TRAINED=0
for ((n = 0; n < ${#CYCLE_FILES[@]}; n++)); do
	PLAYED=0
	for ((i = 0; i <= n; i++)); do
		PLAYED=$((PLAYED + CYCLE_COUNTS[i]))
	done
	WINDOW=$(window_positions "$PLAYED")
	IN_WINDOW=${CYCLE_COUNTS[n]}
	for ((i = n - 1; i >= 0 && IN_WINDOW < WINDOW; i--)); do
		IN_WINDOW=$((IN_WINDOW + CYCLE_COUNTS[i]))
	done
	TRAIN=$((CYCLE_COUNTS[n] * TRAININGS_PER_POSITION))
	((TRAIN <= MAX_TRAINING_POSITIONS)) || TRAIN=$MAX_TRAINING_POSITIONS
	((TRAIN <= IN_WINDOW)) || TRAIN=$IN_WINDOW
	TRAINED=$((TRAINED + TRAIN))
	FROM=""
	for f in ${CYCLE_FILES[n]}; do
		FROM+="${f##*/games_}"
		FROM="${FROM%.sgf.gz}+"
	done
	printf '%-6s %-40s %10d %10d %10d %12d\n' "$n" "games_${FROM%+}" "${CYCLE_COUNTS[n]}" "$IN_WINDOW" "$TRAIN" "$TRAINED"
done
echo "$TOTAL positions carried over, each trained on $(awk -v t="$TRAINED" -v p="$TOTAL" 'BEGIN { printf "%.1f", t / p }') times on average"

[[ "$DRY_RUN" == 0 ]] || exit 0

mkdir -p "$NEW_GAMESDIR"
for ((n = 0; n < ${#CYCLE_FILES[@]}; n++)); do
	dst="$NEW_GAMESDIR"/games_"$n".sgf.gz
	read -ra parts <<<"${CYCLE_FILES[n]}"
	if ((${#parts[@]} == 1)); then
		cp --reflink=auto "${parts[0]}" "$dst"
	else
		# The files are concatenations of gzip members, so they concatenate.
		cat "${parts[@]}" >"$dst"
	fi
	echo "$(stat -c %s "$dst") ${CYCLE_COUNTS[n]}" >"${dst%.sgf.gz}.positions"
done
echo "Copied into $NEW_GAMESDIR; start synchronous_loop.sh on $NEW_BASEDIR with the new network's MODEL_CONFIG"
