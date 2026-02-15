"""Constants for skills testing module.

Centralized constants to avoid magic numbers and ensure consistency.
"""

# Character budget for Claude Code skill descriptions
CLAUDE_CODE_CHAR_BUDGET = 15000

# Average characters per skill description (for estimating ignored skills)
AVG_CHARS_PER_SKILL = 500

# Score thresholds for test results
SCORE_THRESHOLD_HIGH = 80  # >= 80% is good
SCORE_THRESHOLD_MEDIUM = 60  # >= 60% is acceptable
SCORE_THRESHOLD_LOW = 50  # >= 50% is marginal

# Pass rate thresholds for test suites
PASS_RATE_HIGH = 0.8  # >= 80% pass rate is good
PASS_RATE_MEDIUM = 0.5  # >= 50% pass rate is acceptable

# Output truncation lengths (for console display)
TRUNCATE_OUTPUT_SHORT = 200  # For query preview
TRUNCATE_OUTPUT_MEDIUM = 400  # For response preview
TRUNCATE_OUTPUT_LONG = 500  # For JSON output

# Character budget warning thresholds
CHAR_BUDGET_WARNING_PCT = 75  # Warn at 75% usage
CHAR_BUDGET_CRITICAL_PCT = 100  # Critical at 100% usage

# Spinner animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_REFRESH_RATE = 10  # Refreshes per second

# Threading defaults
THREAD_JOIN_TIMEOUT = 300.0  # 5 minutes max wait for thread
SPINNER_SLEEP_INTERVAL = 0.1  # 100ms between spinner updates

# Preview limits
MAX_PREVIEW_LINES = 8  # Maximum lines to show in preview
MAX_DESCRIPTION_LENGTH = 60  # Maximum description length in list view
