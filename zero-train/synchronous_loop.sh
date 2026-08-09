#!/usr/bin/env bash

set -euo pipefail
{

	# Runs the entire self-play training process synchronously in a loop:
	# self-play -> train -> optionally pit the new model against the previous one.
	#
	# Games are stored in BASEDIR/games as games_{N}.sgf.gz, where N is the number of
	# the model that played them. Models and optimizer states are stored in
	# BASEDIR/models as model_{N}.mpk / optimizer_{N}.mpk, together with
	# model_swa_{N}.mpk holding the averaged weights that are used for self-play
	# and pitting.
	#
	# The very first cycle plays its games without a model (random play); the
	# network is initialized only before the first training step.
	#
	# An interrupted run resumes when started again: it continues from the latest
	# model and only tops the current cycle's games up to the target count before
	# training.

	if [[ $# -lt 2 ]]; then
		echo "Usage: $0 BASEDIR BACKEND [USEGATING]"
		echo "Assumes the oppai-zero-train executable is already built with 'cargo build --release'."
		echo "BASEDIR containing self-play games and models"
		echo "BACKEND backend to use, like 'Rocm' or 'Flex'"
		echo "USEGATING = 1 to pit each new model against the previous one and reject it if it does not reach the win rate threshold, 0 to not use gating (default)"
		exit 0
	fi
	BASEDIR="$(realpath "$1")"
	BACKEND="$2"
	USEGATING="${3:-0}"

	GITROOTDIR="$(git rev-parse --show-toplevel)"
	BIN="$GITROOTDIR"/target/release/oppai-zero-train

	GAMESDIR="$BASEDIR"/games
	MODELSDIR="$BASEDIR"/models
	LOGSDIR="$BASEDIR"/logs
	mkdir -p "$GAMESDIR" "$MODELSDIR" "$LOGSDIR"

	# Parameters for the training run.

	MODEL_CONFIG="$GITROOTDIR"/zero-burn/configs/b10c256nbt.json # Path to a JSON file with the model architecture configuration; empty for the default one
	WIDTHS="16 17 18 19 20 21 22 23 24"                          # Field widths to sample from during self-play, space separated
	HEIGHTS="16 17 18 19 20 21 22 23 24"                         # Field heights to sample from during self-play, space separated
	KOMIS_X2="0 1 -1"                                            # Komi values multiplied by 2 to sample from during self-play, space separated
	NUM_GAMES_PER_CYCLE=1024                                     # Every cycle, play this many games
	PARALLEL_GAMES=32                                            # How many games to play or recalculate concurrently, merging their positions into shared forward passes
	BATCH_GAMES=16                                               # How many games' positions are enough to dispatch a forward pass during self-play
	THREADS=                                                     # How many OS threads play games; empty for the default of one per physical core
	WINDOW=8                                                     # Train on the games of this many most recent cycles
	BATCHSIZE=256
	LEARNING_RATE=0.01            # Constant learning rate for every training after the first one
	LEARNING_RATE_START=0.0000001 # Learning rate at the first batch of the first training, kept low to warm up momentum
	LEARNING_RATE_END=0.01        # Learning rate at the last batch of the first training
	PIT_GAMES=50                  # Gating games per side, the total number of games is twice this
	WIN_RATE_THRESHOLD=0.55       # Win rate the new model has to reach against the previous one to be accepted

	# Training rotates a rectangular field, so it needs the maximum of the played sizes.
	TRAIN_WIDTH=$(printf '%s\n' $WIDTHS | sort -n | tail -1)
	TRAIN_HEIGHT=$(printf '%s\n' $HEIGHTS | sort -n | tail -1)

	# Bound on the norm of the whole gradient, so that a batch that happens to
	# produce an enormous gradient nudges the weights instead of wrecking them.
	# The 2500 is the bound for a network whose normalization layers carry no
	# batch statistics, stated for a loss summed over a batch of 256 samples. It
	# grows with the square root of the batch size - averaging more samples
	# shrinks the noise it is there to catch - and dividing by the batch size
	# restates it for the loss trained on here, which is a mean over the batch
	# rather than a sum.
	GRADIENT_CLIPPING=$(awk -v b="$BATCHSIZE" 'BEGIN { printf "%.9g", 2500 * sqrt(b / 256) / b }')

	CONFIG_ARGS=()
	PIT_CONFIG_ARGS=()
	if [[ -n "$MODEL_CONFIG" ]]; then
		CONFIG_ARGS=(--model-config "$MODEL_CONFIG")
		PIT_CONFIG_ARGS=(--model-config "$MODEL_CONFIG" --model-config-new "$MODEL_CONFIG")
	fi

	THREADS_ARGS=()
	if [[ -n "$THREADS" ]]; then
		THREADS_ARGS=(--threads "$THREADS")
	fi

	# Continue from the latest model if there is one.
	N=0
	for f in "$MODELSDIR"/model_*.mpk; do
		[[ -e "$f" ]] || continue
		n="${f##*/model_}"
		n="${n%.mpk}"
		[[ "$n" =~ ^[0-9]+$ ]] || continue
		((n > N)) && N=$n
	done

	# Begin cycling forever, running each step in order.
	REJECTED=0
	set -x
	while true; do
		# Play only the games the current cycle is still missing, so an
		# interrupted run picks up where it stopped instead of playing a whole new
		# cycle. After a gating rejection the count is already at the target, so a
		# full extra cycle is played instead.
		TO_PLAY="$NUM_GAMES_PER_CYCLE"
		if ((REJECTED == 0)) && [[ -e "$GAMESDIR"/games_"$N".sgf.gz ]]; then
			PLAYED=$("$BIN" --backend "$BACKEND" count --games "$GAMESDIR"/games_"$N".sgf.gz |
				sed -n 's/^Games: \([0-9]*\);.*/\1/p')
			TO_PLAY=$((NUM_GAMES_PER_CYCLE - PLAYED))
		fi
		REJECTED=0

		if ((TO_PLAY > 0)); then
			# The averaged weights are the ones to play with when they exist. On
			# the very first cycle there is no model at all and the games are
			# played with a random model.
			PLAY_MODEL_ARGS=()
			if [[ -e "$MODELSDIR"/model_swa_"$N".mpk ]]; then
				PLAY_MODEL_ARGS=(--model "$MODELSDIR"/model_swa_"$N".mpk)
			elif [[ -e "$MODELSDIR"/model_"$N".mpk ]]; then
				PLAY_MODEL_ARGS=(--model "$MODELSDIR"/model_"$N".mpk)
			fi

			echo "Self-play"
			time "$BIN" --backend "$BACKEND" play "${CONFIG_ARGS[@]}" \
				"${PLAY_MODEL_ARGS[@]}" \
				--width $WIDTHS --height $HEIGHTS \
				--games "$GAMESDIR"/games_"$N".sgf.gz \
				--count "$TO_PLAY" \
				--parallel-games "$PARALLEL_GAMES" \
				--batch-games "$BATCH_GAMES" \
				"${THREADS_ARGS[@]}" \
				--komi-x2 $KOMIS_X2 2>&1 | tee -a "$LOGSDIR"/play.log
		fi

		if [[ ! -e "$MODELSDIR"/model_"$N".mpk ]]; then
			echo "Init"
			time "$BIN" --backend "$BACKEND" init "${CONFIG_ARGS[@]}" \
				--model "$MODELSDIR"/model_"$N".mpk \
				--optimizer "$MODELSDIR"/optimizer_"$N".mpk 2>&1 | tee -a "$LOGSDIR"/init.log
		fi

		# Train on the games of the last WINDOW cycles that exist. The games of
		# the previous cycles were played by older models, so their policy
		# surprise no longer reflects the network about to be trained on them;
		# recalculate it with the current model first. The current cycle's games
		# were just played by this model and are used as they are.
		GAMES=("$GAMESDIR"/games_"$N".sgf.gz)
		OLD_GAMES=()
		for ((i = N - 1; i >= 0 && i > N - WINDOW; i--)); do
			[[ -e "$GAMESDIR"/games_"$i".sgf.gz ]] && OLD_GAMES+=("$GAMESDIR"/games_"$i".sgf.gz)
		done
		if ((${#OLD_GAMES[@]} > 0)); then
			CURRENT_MODEL="$MODELSDIR"/model_swa_"$N".mpk
			[[ -e "$CURRENT_MODEL" ]] || CURRENT_MODEL="$MODELSDIR"/model_"$N".mpk

			echo "Recalculate surprise"
			# The recalculated games are only valid for this cycle's model, so the
			# file is transient and recreated every cycle; the tool appends to it,
			# hence the removal.
			RECALC_GAMES="$BASEDIR"/games_recalc.sgf.gz
			rm -f "$RECALC_GAMES"
			time "$BIN" --backend "$BACKEND" recalc-surprise "${CONFIG_ARGS[@]}" \
				--model "$CURRENT_MODEL" \
				--games "${OLD_GAMES[@]}" \
				--games-new "$RECALC_GAMES" \
				--parallel-games "$PARALLEL_GAMES" 2>&1 | tee -a "$LOGSDIR"/recalc.log
			GAMES+=("$RECALC_GAMES")
		fi

		# The moving average continues from the previous one when it exists,
		# otherwise it starts from the loaded model.
		SWA_ARGS=(--model-swa-new "$MODELSDIR"/model_swa_"$((N + 1))".mpk)
		if [[ -e "$MODELSDIR"/model_swa_"$N".mpk ]]; then
			SWA_ARGS+=(--model-swa "$MODELSDIR"/model_swa_"$N".mpk)
		fi

		# Only the first training ramps the learning rate up, to warm up the fresh
		# momentum; afterwards it stays constant.
		if ((N == 0)); then
			LR_ARGS=(--learning-rate-start "$LEARNING_RATE_START" --learning-rate-end "$LEARNING_RATE_END")
		else
			LR_ARGS=(--learning-rate-start "$LEARNING_RATE" --learning-rate-end "$LEARNING_RATE")
		fi

		echo "Train"
		time "$BIN" --backend "$BACKEND" train "${CONFIG_ARGS[@]}" \
			--width "$TRAIN_WIDTH" --height "$TRAIN_HEIGHT" \
			--model "$MODELSDIR"/model_"$N".mpk \
			--optimizer "$MODELSDIR"/optimizer_"$N".mpk \
			--model-new "$MODELSDIR"/model_"$((N + 1))".mpk \
			--optimizer-new "$MODELSDIR"/optimizer_"$((N + 1))".mpk \
			"${SWA_ARGS[@]}" \
			--games "${GAMES[@]}" \
			"${LR_ARGS[@]}" \
			--gradient-clipping "$GRADIENT_CLIPPING" \
			--batch-size "$BATCHSIZE" 2>&1 | tee -a "$LOGSDIR"/train.log

		if [[ "$USEGATING" == 1 ]]; then
			OLD_MODEL="$MODELSDIR"/model_swa_"$N".mpk
			[[ -e "$OLD_MODEL" ]] || OLD_MODEL="$MODELSDIR"/model_"$N".mpk

			echo "Pit"
			STATUS=0
			time "$BIN" --backend "$BACKEND" pit "${PIT_CONFIG_ARGS[@]}" \
				--width $WIDTHS --height $HEIGHTS \
				--model "$OLD_MODEL" \
				--model-new "$MODELSDIR"/model_swa_"$((N + 1))".mpk \
				--games "$GAMESDIR"/pit_"$((N + 1))".sgf.gz \
				--count "$PIT_GAMES" \
				--win-rate-threshold "$WIN_RATE_THRESHOLD" 2>&1 | tee -a "$LOGSDIR"/pit.log || STATUS=$?
			if ((STATUS == 2)); then
				# Rejected: drop the new model and play another cycle of games
				# with the previous one; the next training will start over from
				# the same model with more data.
				rm -f "$MODELSDIR"/model_"$((N + 1))".mpk \
					"$MODELSDIR"/optimizer_"$((N + 1))".mpk \
					"$MODELSDIR"/model_swa_"$((N + 1))".mpk
				REJECTED=1
				continue
			elif ((STATUS != 0)); then
				exit "$STATUS"
			fi
		fi

		N=$((N + 1))
	done

	exit 0
}
