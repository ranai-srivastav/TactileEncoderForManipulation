#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-mmml-train}"
shift || true

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  cat <<EOF
Usage:
  scripts/start_tmux_training.sh [session_name] <command...>

Example:
  scripts/start_tmux_training.sh mmml-train \
    .venv-uv-cu124/bin/python train.py --arch ft_lstm --root_dir /home/aayush/TEMUDataset/Gelsight --L 7 --overfit --n_iters 100
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SESSION}_${TS}.log"

printf -v QCMD "%q " "$@"

TMUX_CMD="cd '$ROOT_DIR' && $QCMD 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION" "$TMUX_CMD"

echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Attach: tmux attach -t $SESSION"