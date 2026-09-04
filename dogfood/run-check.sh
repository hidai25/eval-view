#!/usr/bin/env bash
# Preserve both complete logs and the real command status through tee.
set -uo pipefail
check_name="$1"
shift
"$@" 2>&1 | tee "${check_name}-output.txt"
statuses=("${PIPESTATUS[@]}")
status="${statuses[0]}"
if [ "$status" -eq 0 ] && [ "${statuses[1]}" -ne 0 ]; then
  status="${statuses[1]}"
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "exit_code=$status" >> "$GITHUB_OUTPUT"
fi
exit "$status"
