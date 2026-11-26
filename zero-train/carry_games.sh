#!/usr/bin/env bash

set -euo pipefail

# Seeds a new run of synchronous_loop.sh with the most recent games of another
# run, so that a new network - a bigger one, say - catches up on them instead of
# starting from random games.
#
# The games files are copied as they are, oldest first, as the new run's first
# cycles. Started on them, the loop first pretrains a fresh network on all of
# them at once, PRETRAIN_PASSES times over, and then plays its own games; the
# carried ones stay in its window until newer games push them out. The old
# run's latest file is carried over as it is when this runs, even while it is
# still being played.
#
# Only the most recent games are worth carrying over: the oldest ones were
# played by much weaker networks, and their targets lead away from good play.
# But a fresh network needs its whole learning rate warmup - 1M trained
# positions - and more, and training that much on too few positions repeats
# them often enough to memorize them. The plan printed tells how much the
# pretraining will train on what is carried over.

usage() {
	echo "Usage: $0 [-n] OLD_BASEDIR NEW_BASEDIR POSITIONS"
	echo "Copies the games of OLD_BASEDIR's most recent cycles, back until they hold POSITIONS positions, as the first cycles of NEW_BASEDIR"
	echo "-n only prints the plan"
	echo "PRETRAIN_PASSES can be overridden from the environment, and should be when the loop's differs"
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

# As in synchronous_loop.sh.
PRETRAIN_PASSES="${PRETRAIN_PASSES:-3}"

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
# The latest file may still be being played, and grow while it is counted and
# copied: it is carried over from a snapshot, so that the plan and the copy
# agree on its games. Every game is appended as a gzip member of its own, so a
# snapshot taken in the middle of a write misses only the end of the last
# game, and fails to read: it is taken again until it reads in full.
SNAPSHOT_DIR=$(mktemp -d)
trap 'rm -rf "$SNAPSHOT_DIR"' EXIT
SNAPSHOT="$SNAPSHOT_DIR"/games_"$LAST".sgf.gz
for ((try = 1; ; try++)); do
	cp "$OLD_GAMESDIR"/games_"$LAST".sgf.gz "$SNAPSHOT"
	"$BIN" count --games "$SNAPSHOT" >/dev/null 2>&1 && break
	((try < 10)) || {
		echo "Cannot take a snapshot of games_$LAST that reads in full" >&2
		exit 1
	}
	sleep 1
done
FILES=()
COUNTS=()
TOTAL=0
for ((i = LAST; i >= 0 && TOTAL < WANTED; i--)); do
	f="$OLD_GAMESDIR"/games_"$i".sgf.gz
	((i < LAST)) || f="$SNAPSHOT"
	[[ -e "$f" ]] || continue
	count=$(positions "$f")
	FILES=("$f" "${FILES[@]}")
	COUNTS=("$count" "${COUNTS[@]}")
	TOTAL=$((TOTAL + count))
done
# The loop tells a carried run from one of its own by its having more than one
# cycle of games and no model.
if ((${#FILES[@]} < 2)); then
	echo "Only one cycle to carry over; carry over more positions" >&2
	exit 1
fi

printf '%-6s %-12s %10s\n' cycle from positions
for ((n = 0; n < ${#FILES[@]}; n++)); do
	from="${FILES[n]##*/}"
	printf '%-6s %-12s %10d\n' "$n" "${from%.sgf.gz}" "${COUNTS[n]}"
done
echo "$TOTAL positions carried over, pretrained on $(awk -v t="$TOTAL" -v k="$PRETRAIN_PASSES" 'BEGIN { printf "%d", t * k }') positions"

[[ "$DRY_RUN" == 0 ]] || exit 0

mkdir -p "$NEW_GAMESDIR"
for ((n = 0; n < ${#FILES[@]}; n++)); do
	dst="$NEW_GAMESDIR"/games_"$n".sgf.gz
	cp --reflink=auto "${FILES[n]}" "$dst"
	echo "$(stat -c %s "$dst") ${COUNTS[n]}" >"${dst%.sgf.gz}.positions"
done
echo "Copied into $NEW_GAMESDIR; start synchronous_loop.sh on $NEW_BASEDIR with the new network's MODEL_CONFIG"
