#!/usr/bin/env bash
#
# benchmark-uv.sh - Benchmark uv package manager performance for EvalView
#
# Usage: ./benchmark-uv.sh [iterations]
#   iterations: Number of benchmark iterations (default: 3)
#
# Output: benchmark-uv-results.json
#

set -uo pipefail

#------------------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="${SCRIPT_DIR}"
readonly VENV_DIR="${PROJECT_DIR}/.venv"
readonly RESULTS_FILE="${PROJECT_DIR}/benchmark-uv-results.json"
readonly DEFAULT_ITERATIONS=3

# Temp file for iteration results
readonly ITER_RESULTS_FILE="${PROJECT_DIR}/.benchmark-iteration-result.json"

# Colors for terminal output
if [[ -t 1 ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly NC='\033[0m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly NC=''
fi

#------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

die() {
    log_error "$*"
    exit 1
}

# Get current time in seconds with subsecond precision using Python
get_time() {
    python3 -c 'import time; print(time.time())'
}

# Calculate elapsed time
calc_elapsed() {
    local start="$1"
    local end="$2"
    python3 -c "print(round(${end} - ${start}, 2))"
}

#------------------------------------------------------------------------------
# CACHE CLEARING FUNCTIONS
#------------------------------------------------------------------------------

clear_uv_cache() {
    log_info "Clearing uv cache..."
    uv cache clean 2>/dev/null || true
}

clear_python_caches() {
    log_info "Clearing Python caches..."
    find "${PROJECT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyo" -delete 2>/dev/null || true
}

clear_tool_caches() {
    log_info "Clearing tool caches..."
    rm -rf "${PROJECT_DIR}/.pytest_cache" 2>/dev/null || true
    rm -rf "${PROJECT_DIR}/.mypy_cache" 2>/dev/null || true
    rm -rf "${PROJECT_DIR}/.ruff_cache" 2>/dev/null || true
    find "${PROJECT_DIR}" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
}

clear_build_artifacts() {
    log_info "Clearing build artifacts..."
    rm -rf "${PROJECT_DIR}/build" 2>/dev/null || true
    rm -rf "${PROJECT_DIR}/dist" 2>/dev/null || true
    rm -rf "${PROJECT_DIR}"/*.egg-info 2>/dev/null || true
    rm -rf "${PROJECT_DIR}"/evalview.egg-info 2>/dev/null || true
}

remove_venv() {
    log_info "Removing virtual environment..."
    rm -rf "${VENV_DIR}" 2>/dev/null || true
}

clear_all_caches() {
    log_info "=== Clearing ALL caches ==="
    remove_venv
    clear_uv_cache
    clear_python_caches
    clear_tool_caches
    clear_build_artifacts
    log_success "All caches cleared"
}

#------------------------------------------------------------------------------
# BENCHMARK FUNCTIONS
#------------------------------------------------------------------------------

run_single_iteration() {
    local iteration=$1

    log_info "=========================================="
    log_info "ITERATION ${iteration}"
    log_info "=========================================="

    # Clear everything first
    clear_all_caches

    # === TIMED: Install (uv sync creates venv automatically) ===
    log_info "Installing dependencies with uv sync..."
    local install_start install_end install_time
    install_start=$(get_time)
    uv sync --all-extras --quiet
    install_end=$(get_time)
    install_time=$(calc_elapsed "$install_start" "$install_end")
    log_success "Install completed in ${install_time}s"

    # === TIMED: pytest ===
    log_info "Running pytest..."
    local pytest_start pytest_end pytest_time
    pytest_start=$(get_time)
    uv run pytest tests/ -v --tb=short > /dev/null 2>&1 || true
    pytest_end=$(get_time)
    pytest_time=$(calc_elapsed "$pytest_start" "$pytest_end")
    log_success "Pytest completed in ${pytest_time}s"

    # === TIMED: black ===
    log_info "Running black --check..."
    local black_start black_end black_time
    black_start=$(get_time)
    uv run black evalview/ tests/ --line-length 100 --check > /dev/null 2>&1 || true
    black_end=$(get_time)
    black_time=$(calc_elapsed "$black_start" "$black_end")
    log_success "Black completed in ${black_time}s"

    # === TIMED: ruff ===
    log_info "Running ruff check..."
    local ruff_start ruff_end ruff_time
    ruff_start=$(get_time)
    uv run ruff check evalview/ tests/ > /dev/null 2>&1 || true
    ruff_end=$(get_time)
    ruff_time=$(calc_elapsed "$ruff_start" "$ruff_end")
    log_success "Ruff completed in ${ruff_time}s"

    # === TIMED: mypy ===
    log_info "Running mypy --strict..."
    local mypy_start mypy_end mypy_time
    mypy_start=$(get_time)
    uv run mypy evalview/ --strict > /dev/null 2>&1 || true
    mypy_end=$(get_time)
    mypy_time=$(calc_elapsed "$mypy_start" "$mypy_end")
    log_success "Mypy completed in ${mypy_time}s"

    # Calculate total
    local total_time
    total_time=$(python3 -c "print(round(${install_time} + ${pytest_time} + ${black_time} + ${ruff_time} + ${mypy_time}, 2))")

    log_success "Iteration ${iteration} total: ${total_time}s"

    # Write JSON to temp file (avoids subshell capture issues)
    cat > "${ITER_RESULTS_FILE}" <<EOF
{"iteration":${iteration},"install_time":${install_time},"pytest_time":${pytest_time},"black_time":${black_time},"ruff_time":${ruff_time},"mypy_time":${mypy_time},"total_time":${total_time}}
EOF
}

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

main() {
    local num_iterations="${1:-$DEFAULT_ITERATIONS}"

    # Validate
    if ! [[ "$num_iterations" =~ ^[0-9]+$ ]] || [[ "$num_iterations" -lt 1 ]]; then
        die "Invalid iterations: ${num_iterations}. Must be a positive integer."
    fi

    log_info "=========================================="
    log_info "UV BENCHMARK FOR EVALVIEW"
    log_info "Iterations: ${num_iterations}"
    log_info "Project: ${PROJECT_DIR}"
    log_info "=========================================="

    cd "${PROJECT_DIR}" || die "Cannot change to project directory"

    # Arrays to store results
    local -a install_times=()
    local -a pytest_times=()
    local -a black_times=()
    local -a ruff_times=()
    local -a mypy_times=()
    local -a total_times=()
    local -a run_jsons=()

    # Run iterations
    for ((i = 1; i <= num_iterations; i++)); do
        run_single_iteration "$i"

        # Read results from temp file
        local result
        result=$(cat "${ITER_RESULTS_FILE}")
        run_jsons+=("$result")

        # Parse values using Python
        install_times+=($(python3 -c "import json; print(json.loads('${result}')['install_time'])"))
        pytest_times+=($(python3 -c "import json; print(json.loads('${result}')['pytest_time'])"))
        black_times+=($(python3 -c "import json; print(json.loads('${result}')['black_time'])"))
        ruff_times+=($(python3 -c "import json; print(json.loads('${result}')['ruff_time'])"))
        mypy_times+=($(python3 -c "import json; print(json.loads('${result}')['mypy_time'])"))
        total_times+=($(python3 -c "import json; print(json.loads('${result}')['total_time'])"))
    done

    # Clean up temp file
    rm -f "${ITER_RESULTS_FILE}"

    # Calculate averages using Python for reliability
    local install_avg pytest_avg black_avg ruff_avg mypy_avg total_avg
    install_avg=$(python3 -c "print(round(sum([${install_times[*]/%/,}0]) / ${num_iterations}, 2))")
    pytest_avg=$(python3 -c "print(round(sum([${pytest_times[*]/%/,}0]) / ${num_iterations}, 2))")
    black_avg=$(python3 -c "print(round(sum([${black_times[*]/%/,}0]) / ${num_iterations}, 2))")
    ruff_avg=$(python3 -c "print(round(sum([${ruff_times[*]/%/,}0]) / ${num_iterations}, 2))")
    mypy_avg=$(python3 -c "print(round(sum([${mypy_times[*]/%/,}0]) / ${num_iterations}, 2))")
    total_avg=$(python3 -c "print(round(sum([${total_times[*]/%/,}0]) / ${num_iterations}, 2))")

    # Build runs JSON array
    local runs_json=""
    for ((i = 0; i < ${#run_jsons[@]}; i++)); do
        if [[ $i -gt 0 ]]; then
            runs_json+=","
        fi
        runs_json+="${run_jsons[$i]}"
    done

    # Get system info
    local timestamp python_version uv_version os_name os_version
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    uv_version=$(uv --version 2>/dev/null | awk '{print $2}' || echo "unknown")
    os_name=$(uname -s)
    os_version=$(uname -r)

    # Generate final JSON
    cat > "${RESULTS_FILE}" <<EOF
{
  "benchmark": "uv",
  "timestamp": "${timestamp}",
  "iterations": ${num_iterations},
  "system": {
    "os": "${os_name}",
    "os_version": "${os_version}",
    "python_version": "${python_version}",
    "uv_version": "${uv_version}"
  },
  "project": {
    "name": "evalview",
    "path": "${PROJECT_DIR}"
  },
  "runs": [${runs_json}],
  "averages": {
    "install_time": ${install_avg},
    "pytest_time": ${pytest_avg},
    "black_time": ${black_avg},
    "ruff_time": ${ruff_avg},
    "mypy_time": ${mypy_avg},
    "total_time": ${total_avg}
  }
}
EOF

    # Print summary
    log_info ""
    log_info "=========================================="
    log_success "BENCHMARK COMPLETE"
    log_info "=========================================="
    log_info "Results written to: ${RESULTS_FILE}"
    log_info ""
    log_info "AVERAGES (${num_iterations} iterations):"
    log_info "  Install:  ${install_avg}s"
    log_info "  Pytest:   ${pytest_avg}s"
    log_info "  Black:    ${black_avg}s"
    log_info "  Ruff:     ${ruff_avg}s"
    log_info "  Mypy:     ${mypy_avg}s"
    log_info "  ─────────────────"
    log_info "  TOTAL:    ${total_avg}s"
    log_info ""

    # Pretty print the JSON
    log_info "JSON Output:"
    cat "${RESULTS_FILE}"
}

main "$@"
