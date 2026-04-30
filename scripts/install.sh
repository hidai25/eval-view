#!/usr/bin/env bash
# EvalView installer — zero-friction setup
# Usage: curl -fsSL https://raw.githubusercontent.com/hidai25/eval-view/main/install.sh | bash
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
DIM='\033[2m'
RESET='\033[0m'

info()  { echo -e "${GREEN}$1${RESET}"; }
warn()  { echo -e "${YELLOW}$1${RESET}"; }
error() { echo -e "${RED}$1${RESET}" >&2; }

echo ""
echo -e "${BOLD}EvalView — Regression testing for AI agents${RESET}"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.9+ is required but not found."
    echo ""
    echo "Install Python from https://python.org or via your package manager:"
    echo "  brew install python3          # macOS"
    echo "  sudo apt install python3      # Ubuntu/Debian"
    echo "  sudo dnf install python3      # Fedora"
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
info "Found $PY_VERSION"

# ── Check pip ─────────────────────────────────────────────────────────────────

if ! "$PYTHON" -m pip --version &>/dev/null; then
    error "pip is not available for $PYTHON."
    echo "Install pip: $PYTHON -m ensurepip --upgrade"
    exit 1
fi

# ── Install EvalView ──────────────────────────────────────────────────────────

echo ""
info "Installing EvalView..."
echo ""

if "$PYTHON" -m pip install --upgrade evalview 2>&1; then
    echo ""
else
    error "Installation failed. Try manually: pip install evalview"
    exit 1
fi

# ── Verify ────────────────────────────────────────────────────────────────────

if ! command -v evalview &>/dev/null; then
    # pip installed but not on PATH — try common locations
    LOCAL_BIN="$HOME/.local/bin"
    if [ -f "$LOCAL_BIN/evalview" ]; then
        warn "evalview is installed at $LOCAL_BIN/evalview but not on your PATH."
        echo ""
        echo "Add to your shell config:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
    else
        warn "evalview installed but not found on PATH."
        echo "You may need to restart your terminal or add pip's bin directory to PATH."
    fi
else
    VERSION=$(evalview --version 2>&1 || echo "unknown")
    echo ""
    info "Installed: $VERSION"
fi

# ── Quick start ───────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Get started:${RESET}"
echo ""
echo "  evalview demo        # See it work in 30 seconds (no API key needed)"
echo "  evalview init        # Detect your agent, create starter tests"
echo "  evalview snapshot    # Capture current behavior as baseline"
echo "  evalview check       # Catch regressions after every change"
echo ""
echo -e "${DIM}Docs: https://github.com/hidai25/eval-view${RESET}"
echo ""
