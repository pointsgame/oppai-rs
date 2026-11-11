#!/usr/bin/env bash

set -euo pipefail
{

	# Runs the entire self-play training process synchronously in a loop:
	# self-play -> train -> optionally pit the new model against the previous one.
	#
	# Games are stored in BASEDIR/games as games_{N}.sgf.gz, where N is the number of
	# the model that played them. Models and optimizer states are stored in
	# BASEDIR/models as model_{N}.bpk / optimizer_{N}.bpk, together with
	# model_swa_{N}.bpk holding the averaged weights. Self-play, surprise
	# recalculation and pitting use the averaged weights: they are less noisy
	# than the last training step's, which is what playing strength wants.
	#
	# Everything about training is measured in positions rather than games, as
	# the positions a game yields vary a lot: random games are long and every
	# search in them counts in full, while a trained network decides games early
	# and searches their tail cheaply. The very first cycle plays random games
	# until there are MIN_POSITIONS positions; the network is initialized only
	# before the first training. Every training then trains on
	# TRAININGS_PER_POSITION times the positions played since the last accepted
	# model, drawn from a window of the most recent games that grows with the
	# positions played so far, and the learning rate warms up over the first
	# trained positions of the run.
	#
	# An interrupted run resumes when started again: it continues from the latest
	# model and only tops the current cycle's games up to the target count before
	# training.

	if [[ $# -lt 2 ]]; then
		echo "Usage: $0 BASEDIR DEVICE [USEGATING]"
		echo "Assumes the oppai-zero-train executable is already built with 'cargo build --release', or set BIN to another build,"
		echo "e.g. BIN=target-rocm/release/oppai-zero-train for one built with '--features rocm --target-dir target-rocm'."
		echo "To keep the other GPUs free, hide them from the process, e.g. 'ROCR_VISIBLE_DEVICES=1 $0 BASEDIR rocm:0'"
		echo "runs everything on the second ROCm GPU, which is then the only one and so rocm:0."
		echo "BASEDIR containing self-play games and models"
		echo "DEVICE device to use, like 'auto', 'rocm:0' or 'flex' (see 'oppai-zero-train devices')"
		echo "USEGATING = 1 to pit each new model against the previous one and reject it if it does not reach the win rate threshold, 0 to not use gating (default)"
		exit 0
	fi
	BASEDIR="$(realpath "$1")"
	DEVICE="$2"
	USEGATING="${3:-0}"

	GITROOTDIR="$(git rev-parse --show-toplevel)"
	BIN="$(realpath "${BIN:-"$GITROOTDIR"/target/release/oppai-zero-train}")"

	GAMESDIR="$BASEDIR"/games
	MODELSDIR="$BASEDIR"/models
	LOGSDIR="$BASEDIR"/logs
	mkdir -p "$GAMESDIR" "$MODELSDIR" "$LOGSDIR"
	# Positions trained so far by the accepted models, which the learning rate
	# warms up over.
	TRAINED_FILE="$BASEDIR"/trained_positions

	# Parameters for the training run.

	MODEL_CONFIG="$GITROOTDIR"/zero-burn/configs/b5c192nbt.json # Path to a JSON file with the model architecture configuration; empty for the default one
	WIDTHS="16 17 18 19 20 21 22 23 24"                          # Field widths to sample from during self-play, space separated
	HEIGHTS="16 17 18 19 20 21 22 23 24"                         # Field heights to sample from during self-play, space separated
	KOMIS_X2="0 1 -1"                                            # Komi values multiplied by 2 to sample from during self-play, space separated
	NUM_GAMES_PER_CYCLE=1024                                     # Every cycle, play this many games; random games are played in batches of this many until there are MIN_POSITIONS positions
	PARALLEL_GAMES=192                                           # How many games to play concurrently, merging their positions into shared forward passes; about 20 MB of memory each
	BATCH_GAMES=64                                               # How many games' positions are enough to dispatch a forward pass during self-play
	READOUTS=1                                                   # Paths a self-play search selects at once; more than 1 adds virtual losses that weaken the search, but fills the passes with fewer games
	RECALC_PARALLEL_GAMES=32                                     # How many games to recalculate the surprise of concurrently
	THREADS=                                                     # How many OS threads play games; empty for the default of one per physical core
	HALF=1                                                       # 1 to evaluate the network in half precision in self-play, surprise recalculation and pitting, many times faster on GPUs
	MIXED_PRECISION=1                                            # 1 to compute the convolutions and projections of training in brain floating point, keeping the weights and everything else in single precision; many times faster on GPUs and uses less memory
	GRADIENT_CHECKPOINTING=0                                     # 1 to recompute cheap activations in the backward pass instead of keeping them, cutting training memory by about a quarter, more for a transformer
	MIN_POSITIONS=100000                                         # Positions of random games to play before the first training, and the smallest window trained on
	TRAININGS_PER_POSITION=8                                     # Every training trains on this many times the positions played since the last accepted model; larger numbers may overfit
	MAX_TRAINING_POSITIONS=500000                                # But on no more positions than this per training
	WINDOW_EXPAND=0.4                                            # The window trained on grows by this many positions for each position played beyond MIN_POSITIONS at first ...
	WINDOW_EXPONENT=0.65                                         # ... and as this power of the positions played in the long run, so that it covers an ever shorter stretch of the run
	WINDOW_SCALE=50000                                           # Positions played beyond MIN_POSITIONS at which the window starts growing sublinearly
	SWA_POSITIONS=80000                                          # Train on this many positions between two snapshots averaged into the weights self-play uses
	BATCHSIZE=64                  # Training memory is mostly the activations kept for the backward pass, about 9 GB for a batch of 64 at 24x24 with mixed precision
	OPTIMIZER=sgd                 # sgd, adamw or muon, see 'oppai-zero-train train --help'; switching in the middle of a run starts the optimizer state afresh
	LEARNING_RATE=                # Learning rate after the warmup, stated for a batch of 256 and scaled to BATCHSIZE below; empty for 0.01 with sgd and 0.00004 with adamw and muon
	PIT_GAMES=50                  # Gating games per side, the total number of games is twice this
	WIN_RATE_THRESHOLD=0.55       # Win rate the new model has to reach against the previous one to be accepted

	# Training rotates a rectangular field, so it needs the maximum of the played sizes.
	TRAIN_WIDTH=$(printf '%s\n' $WIDTHS | sort -n | tail -1)
	TRAIN_HEIGHT=$(printf '%s\n' $HEIGHTS | sort -n | tail -1)

	# Bound on the norm of the whole gradient, so that a batch that happens to
	# produce an enormous gradient nudges the weights instead of wrecking them.
	# The 2500 is the bound for a network whose normalization layers carry no
	# batch statistics trained with SGD, stated for a loss summed over a batch of
	# 256 samples; Adam and Muon normalize their steps, so a big gradient does
	# less harm there, and their bound is looser. It grows with the square root
	# of the batch size - averaging more samples shrinks the noise it is there to
	# catch - and dividing by the batch size restates it for the loss trained on
	# here, which is a mean over the batch rather than a sum.
	if [[ "$OPTIMIZER" == sgd ]]; then
		GRADIENT_CLIPPING_BOUND=2500
		: "${LEARNING_RATE:=0.01}"
	else
		GRADIENT_CLIPPING_BOUND=11000
		: "${LEARNING_RATE:=0.00004}"
	fi
	GRADIENT_CLIPPING=$(awk -v c="$GRADIENT_CLIPPING_BOUND" -v b="$BATCHSIZE" 'BEGIN { printf "%.9g", c * sqrt(b / 256) / b }')

	# The loss is a mean over the batch, so an SGD step moves the weights as far
	# whatever the batch size; scaling its learning rate with it keeps how far
	# each trained position moves them, and so the learning rate above means the
	# same for any BATCHSIZE. Adam and Muon steps do not depend on the scale of
	# the gradient at all, and their learning rate grows with the square root of
	# the batch size instead, as averaging more samples makes their direction
	# more reliable.
	if [[ "$OPTIMIZER" == sgd ]]; then
		LEARNING_RATE=$(awk -v r="$LEARNING_RATE" -v b="$BATCHSIZE" 'BEGIN { printf "%.9g", r * b / 256 }')
	else
		LEARNING_RATE=$(awk -v r="$LEARNING_RATE" -v b="$BATCHSIZE" 'BEGIN { printf "%.9g", r * sqrt(b / 256) }')
	fi

	SWA_PERIOD=$(((SWA_POSITIONS + BATCHSIZE / 2) / BATCHSIZE))
	((SWA_PERIOD > 0)) || SWA_PERIOD=1

	# The fraction of the learning rate a training gets after this many trained
	# positions. A fresh network's gradients are large and erratic, and full
	# steps on them can throw it somewhere it never recovers from; the steps
	# grow as it settles.
	warmup() {
		awk -v t="$1" 'BEGIN {
			split("20 14 10 7 5 3 2 1.4", d, " ")
			i = int(t / 250000) + 1
			printf "%.9g", i <= 8 ? 1 / d[i] : 1
		}'
	}

	# How many positions the window trained on spans once this many positions
	# have been played: all of them up to MIN_POSITIONS, then growing as
	# WINDOW_EXPAND per position at first and as WINDOW_EXPONENT power of them
	# in the long run.
	window_positions() {
		awk -v n="$1" -v m="$MIN_POSITIONS" -v e="$WINDOW_EXPAND" -v p="$WINDOW_EXPONENT" -v s="$WINDOW_SCALE" 'BEGIN {
			if (n <= m) { printf "%d", m; exit }
			x = n - m + s
			printf "%d", m + e * (x ^ p - s ^ p) / (p * s ^ (p - 1))
		}'
	}

	# Positions of a games file, counted once and cached next to it for as long
	# as its size stays the same.
	positions() {
		local file="$1" cache="${1%.sgf.gz}.positions" size
		[[ -e "$file" ]] || {
			echo 0
			return
		}
		size=$(stat -c %s "$file")
		if [[ -e "$cache" ]] && [[ "$(cut -d' ' -f1 "$cache")" == "$size" ]]; then
			cut -d' ' -f2 "$cache"
			return
		fi
		local count
		count=$("$BIN" --device "$DEVICE" count --games "$file" | sed -n 's/^Games: [0-9]*; examples: \([0-9]*\).*/\1/p')
		echo "$size $count" >"$cache"
		echo "$count"
	}

	CONFIG_ARGS=()
	PIT_CONFIG_ARGS=()
	if [[ -n "$MODEL_CONFIG" ]]; then
		CONFIG_ARGS=(--model-config "$MODEL_CONFIG")
		PIT_CONFIG_ARGS=(--model-config "$MODEL_CONFIG" --model-config-new "$MODEL_CONFIG")
	fi

	HALF_ARGS=()
	if [[ "$HALF" == 1 ]]; then
		HALF_ARGS=(--half)
	fi

	MIXED_PRECISION_ARGS=()
	if [[ "$MIXED_PRECISION" == 1 ]]; then
		MIXED_PRECISION_ARGS=(--mixed-precision)
	fi

	GRADIENT_CHECKPOINTING_ARGS=()
	if [[ "$GRADIENT_CHECKPOINTING" == 1 ]]; then
		GRADIENT_CHECKPOINTING_ARGS=(--gradient-checkpointing)
	fi

	THREADS_ARGS=()
	if [[ -n "$THREADS" ]]; then
		THREADS_ARGS=(--threads "$THREADS")
	fi

	# The weights a model plays with: the averaged ones, which the very first
	# model, initialized rather than trained, does not have.
	played_model() {
		if [[ -e "$MODELSDIR"/model_swa_"$1".bpk ]]; then
			echo "$MODELSDIR"/model_swa_"$1".bpk
		else
			echo "$MODELSDIR"/model_"$1".bpk
		fi
	}

	play() {
		echo "Self-play"
		time "$BIN" --device "$DEVICE" "${HALF_ARGS[@]}" play "${CONFIG_ARGS[@]}" \
			"$@" \
			--width $WIDTHS --height $HEIGHTS \
			--games "$GAMESDIR"/games_"$N".sgf.gz \
			--parallel-games "$PARALLEL_GAMES" \
			--batch-games "$BATCH_GAMES" \
			--readouts "$READOUTS" \
			"${THREADS_ARGS[@]}" \
			--komi-x2 $KOMIS_X2 2>&1 | tee -a "$LOGSDIR"/play.log
	}

	# Continue from the latest model if there is one.
	N=0
	for f in "$MODELSDIR"/model_*.bpk; do
		[[ -e "$f" ]] || continue
		n="${f##*/model_}"
		n="${n%.bpk}"
		[[ "$n" =~ ^[0-9]+$ ]] || continue
		((n > N)) && N=$n
	done
	[[ -e "$TRAINED_FILE" ]] || echo 0 >"$TRAINED_FILE"

	# Begin cycling forever, running each step in order.
	REJECTED=0
	set -x
	while true; do
		if [[ ! -e "$MODELSDIR"/model_"$N".bpk ]]; then
			# The very first cycle plays random games, which are nearly free,
			# until there are enough positions for a first training that does
			# not just memorize them.
			while (($(positions "$GAMESDIR"/games_"$N".sgf.gz) < MIN_POSITIONS)); do
				play --count "$NUM_GAMES_PER_CYCLE"
			done

			echo "Init"
			time "$BIN" --device "$DEVICE" init "${CONFIG_ARGS[@]}" \
				--model "$MODELSDIR"/model_"$N".bpk \
				--optimizer "$MODELSDIR"/optimizer_"$N".bpk 2>&1 | tee -a "$LOGSDIR"/init.log
		else
			# Play only the games the current cycle is still missing, so an
			# interrupted run picks up where it stopped instead of playing a whole
			# new cycle. After a gating rejection the count is already at the
			# target, so a full extra cycle is played instead.
			TO_PLAY="$NUM_GAMES_PER_CYCLE"
			if ((REJECTED == 0)) && [[ -e "$GAMESDIR"/games_"$N".sgf.gz ]]; then
				PLAYED=$("$BIN" --device "$DEVICE" count --games "$GAMESDIR"/games_"$N".sgf.gz |
					sed -n 's/^Games: \([0-9]*\);.*/\1/p')
				TO_PLAY=$((NUM_GAMES_PER_CYCLE - PLAYED))
			fi
			if ((TO_PLAY > 0)); then
				play --model "$(played_model "$N")" --count "$TO_PLAY"
			fi
		fi
		REJECTED=0

		# The window: the games of the most recent cycles, back until they hold
		# as many positions as the window spans for all the positions played so
		# far. Always at least the current cycle's.
		TOTAL=0
		for ((i = N; i >= 0; i--)); do
			TOTAL=$((TOTAL + $(positions "$GAMESDIR"/games_"$i".sgf.gz)))
		done
		WINDOW=$(window_positions "$TOTAL")
		NEW=$(positions "$GAMESDIR"/games_"$N".sgf.gz)
		IN_WINDOW=$NEW
		OLD_GAMES=()
		for ((i = N - 1; i >= 0 && IN_WINDOW < WINDOW; i--)); do
			[[ -e "$GAMESDIR"/games_"$i".sgf.gz ]] || continue
			OLD_GAMES+=("$GAMESDIR"/games_"$i".sgf.gz)
			IN_WINDOW=$((IN_WINDOW + $(positions "$GAMESDIR"/games_"$i".sgf.gz)))
		done

		# The games of the previous cycles were played by older models, so their
		# policy surprise no longer reflects the network about to be trained on
		# them; recalculate it with the current model first. The current cycle's
		# games were just played by this model and are used as they are.
		GAMES=("$GAMESDIR"/games_"$N".sgf.gz)
		if ((${#OLD_GAMES[@]} > 0)); then
			echo "Recalculate surprise"
			# The recalculated games are only valid for this cycle's model, so the
			# file is transient and recreated every cycle; the tool appends to it,
			# hence the removal.
			RECALC_GAMES="$BASEDIR"/games_recalc.sgf.gz
			rm -f "$RECALC_GAMES"
			time "$BIN" --device "$DEVICE" "${HALF_ARGS[@]}" recalc-surprise "${CONFIG_ARGS[@]}" \
				--model "$(played_model "$N")" \
				--games "${OLD_GAMES[@]}" \
				--games-new "$RECALC_GAMES" \
				--parallel-games "$RECALC_PARALLEL_GAMES" 2>&1 | tee -a "$LOGSDIR"/recalc.log
			GAMES+=("$RECALC_GAMES")
		fi

		# The moving average continues from the previous one when it exists,
		# otherwise it starts from the loaded model.
		SWA_ARGS=(--model-swa-new "$MODELSDIR"/model_swa_"$((N + 1))".bpk --swa-period "$SWA_PERIOD")
		if [[ -e "$MODELSDIR"/model_swa_"$N".bpk ]]; then
			SWA_ARGS+=(--model-swa "$MODELSDIR"/model_swa_"$N".bpk)
		fi

		TRAINED=$(<"$TRAINED_FILE")
		TRAIN_POSITIONS=$((NEW * TRAININGS_PER_POSITION))
		((TRAIN_POSITIONS <= MAX_TRAINING_POSITIONS)) || TRAIN_POSITIONS=$MAX_TRAINING_POSITIONS
		((TRAIN_POSITIONS <= IN_WINDOW)) || TRAIN_POSITIONS=$IN_WINDOW
		CYCLE_LEARNING_RATE=$(awk -v r="$LEARNING_RATE" -v w="$(warmup "$TRAINED")" 'BEGIN { printf "%.9g", r * w }')

		echo "Train on $TRAIN_POSITIONS of $IN_WINDOW positions, $TRAINED trained so far, learning rate $CYCLE_LEARNING_RATE"
		time "$BIN" --device "$DEVICE" train "${CONFIG_ARGS[@]}" \
			--width "$TRAIN_WIDTH" --height "$TRAIN_HEIGHT" \
			--model "$MODELSDIR"/model_"$N".bpk \
			--optimizer "$MODELSDIR"/optimizer_"$N".bpk \
			--model-new "$MODELSDIR"/model_"$((N + 1))".bpk \
			--optimizer-new "$MODELSDIR"/optimizer_"$((N + 1))".bpk \
			"${SWA_ARGS[@]}" \
			--games "${GAMES[@]}" \
			--train-examples "$TRAIN_POSITIONS" \
			--learning-rate-start "$CYCLE_LEARNING_RATE" --learning-rate-end "$CYCLE_LEARNING_RATE" \
			--optimizer-kind "$OPTIMIZER" \
			--gradient-clipping "$GRADIENT_CLIPPING" \
			"${MIXED_PRECISION_ARGS[@]}" \
			"${GRADIENT_CHECKPOINTING_ARGS[@]}" \
			--batch-size "$BATCHSIZE" 2>&1 | tee -a "$LOGSDIR"/train.log
		BATCHES=$(sed -n 's/.*Training on \([0-9]*\) batches of.*/\1/p' "$LOGSDIR"/train.log | tail -1)

		if [[ "$USEGATING" == 1 ]]; then
			# Gating compares the models as they play.
			echo "Pit"
			STATUS=0
			time "$BIN" --device "$DEVICE" "${HALF_ARGS[@]}" pit "${PIT_CONFIG_ARGS[@]}" \
				--width $WIDTHS --height $HEIGHTS \
				--model "$(played_model "$N")" \
				--model-new "$MODELSDIR"/model_swa_"$((N + 1))".bpk \
				--games "$GAMESDIR"/pit_"$((N + 1))".sgf.gz \
				--count "$PIT_GAMES" \
				--win-rate-threshold "$WIN_RATE_THRESHOLD" 2>&1 | tee -a "$LOGSDIR"/pit.log || STATUS=$?
			if ((STATUS == 2)); then
				# Rejected: drop the new model and play another cycle of games
				# with the previous one; the next training will start over from
				# the same model with more data.
				rm -f "$MODELSDIR"/model_"$((N + 1))".bpk \
					"$MODELSDIR"/optimizer_"$((N + 1))".bpk \
					"$MODELSDIR"/model_swa_"$((N + 1))".bpk
				REJECTED=1
				continue
			elif ((STATUS != 0)); then
				exit "$STATUS"
			fi
		fi

		echo $((TRAINED + BATCHES * BATCHSIZE)) >"$TRAINED_FILE"
		N=$((N + 1))
	done

	exit 0
}
