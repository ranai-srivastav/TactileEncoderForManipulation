#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-mmml-train}"
STATE_DIR="${2:-.monitor_state}"
LINES="${3:-300}"

mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/tmux_${SESSION}.sha"

hash_str() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}

emit_if_changed() {
  local msg="$1"
  local h
  h="$(hash_str "$msg")"
  local prev=""
  if [[ -f "$STATE_FILE" ]]; then
    prev="$(cat "$STATE_FILE" 2>/dev/null || true)"
  fi
  if [[ "$h" == "$prev" ]]; then
    echo "NO_CHANGE"
  else
    echo "$h" > "$STATE_FILE"
    echo "$msg"
  fi
}

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  emit_if_changed "SESSION_MISSING"
  exit 0
fi

PANE="$(tmux capture-pane -t "$SESSION" -p -S -"$LINES" 2>/dev/null || true)"

ITER_LINE="$(printf '%s\n' "$PANE" | grep -E '\[iter[[:space:]]+[0-9]+' | tail -n1 || true)"
DONE_LINE="$(printf '%s\n' "$PANE" | grep -Ei 'Loading best checkpoint|test_loss=|test/f1|training complete|finished training|done\b' | tail -n1 || true)"
ERR_LINE="$(printf '%s\n' "$PANE" | grep -Ei 'traceback|runtimeerror|error:|exception|cuda out of memory|nan|inf' | tail -n1 || true)"

if [[ -n "$ERR_LINE" ]]; then
  emit_if_changed "ERROR|$ERR_LINE"
  exit 0
fi

if [[ -n "$DONE_LINE" ]]; then
  emit_if_changed "DONE|$DONE_LINE"
  exit 0
fi

if [[ -n "$ITER_LINE" ]]; then
  emit_if_changed "PROGRESS|$ITER_LINE"
  exit 0
fi

emit_if_changed "RUNNING|Session $SESSION active; waiting for first iter logs."