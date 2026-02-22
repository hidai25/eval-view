# Skills Testing — Validate and Test SKILL.md for Claude Code, OpenAI Codex, and OpenClaw

> **Problem:** SKILL.md files for Claude Code and OpenAI Codex can silently fail. Claude ignores skills that exceed its 15k character budget, and there's no built-in way to validate skill structure or test skill behavior before deployment.
>
> **Solution:** EvalView provides `evalview skill validate` for deterministic structure validation (no API key needed) and `evalview skill test` for behavior testing. It catches character budget overflows, missing sections, and broken skill logic before they reach users.

> **This is an advanced feature.** For standard agent regression testing, see [Getting Started](GETTING_STARTED.md).
> Use skills testing if you maintain SKILL.md workflows for Claude Code, Codex, or OpenClaw.

> **Your Claude Code skills might be broken.** Claude silently ignores skills that exceed its [15k char budget](https://blog.fsck.com/2025/12/17/claude-code-skills-not-triggering/). EvalView catches this.

**Common symptoms:**
- Skills installed but never trigger
- Claude says "I don't have that skill"
- Works locally, breaks in production
- No errors, just... silence

**Why it happens:** Claude Code has a [15k character budget](https://blog.fsck.com/2025/12/17/claude-code-skills-not-triggering/) for skill descriptions. Exceed it and skills aren't loaded. No warning. No error.

**EvalView catches this before you waste hours debugging.**

---

## 30 Seconds: Validate Your Skill

```bash
pip install evalview
evalview skill validate ./SKILL.md
```

That's it. Catches naming errors, missing fields, reserved words, and spec violations.

**Try it now** with the included example:
```bash
evalview skill validate examples/skills/test-skill/SKILL.md
```

---

## Why Is Claude Ignoring My Skills?

Run the doctor to find out:

```bash
evalview skill doctor ~/.claude/skills/
```

```
⚠️  Character Budget: 127% OVER - Claude is ignoring 4 of your 24 skills

ISSUE: Character budget exceeded
  Claude Code won't see all your skills.
  Fix: Set SLASH_COMMAND_TOOL_CHAR_BUDGET=30000 or reduce descriptions

ISSUE: Duplicate skill names
  code-reviewer defined in:
    - ~/.claude/skills/old/SKILL.md
    - ~/.claude/skills/new/SKILL.md

✗ 4 skills are INVISIBLE to Claude - fix now
```

This is why your skills "don't work." Claude literally can't see them.

---

## 2 Minutes: Add Behavior Tests + CI

**1. Create a test file** next to your SKILL.md:
```yaml
# tests.yaml
name: my-skill-tests
skill: ./SKILL.md

tests:
  - name: basic-test
    input: "Your test prompt"
    expected:
      output_contains: ["expected", "words"]
```

**2. Run locally**
```bash
echo "ANTHROPIC_API_KEY=your-key" > .env.local
evalview skill test tests.yaml
```

**3. Add to CI** — copy [examples/skills/test-skill/.github/workflows/skill-tests.yml](../examples/skills/test-skill/.github/workflows/skill-tests.yml) to your repo

> **Starter template:** See [examples/skills/test-skill/](../examples/skills/test-skill/) for a complete copy-paste example with GitHub Actions.

---

## Validate Skill Structure

Catch errors before Claude ever sees your skill:

```bash
# Validate a single skill
evalview skill validate ./my-skill/SKILL.md

# Validate all skills in a directory
evalview skill validate ~/.claude/skills/ -r

# CI-friendly JSON output
evalview skill validate ./skills/ -r --json
```

**Validates against [official Anthropic spec](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):**
- `name`: max 64 chars, lowercase/numbers/hyphens only, no reserved words ("anthropic", "claude")
- `description`: max 1024 chars, non-empty, no XML tags
- Token size (warns if >5k tokens)
- Policy compliance (no prompt injection patterns)
- Best practices (examples, guidelines sections)

### Example Output

```
━━━ Skill Validation Results ━━━

✓ skills/code-reviewer/SKILL.md
   Name: code-reviewer
   Tokens: ~2,400

✓ skills/doc-writer/SKILL.md
   Name: doc-writer
   Tokens: ~1,800

✗ skills/broken/SKILL.md
   ERROR [MISSING_DESCRIPTION] Skill description is required

Summary: 2 valid, 1 invalid
```

---

## Test Skill Behavior

Validation catches syntax errors. Behavior tests catch **logic errors**.

Define what your skill should do, then verify it actually does it:

```yaml
# tests/code-reviewer.yaml
name: test-code-reviewer
skill: ./skills/code-reviewer/SKILL.md

tests:
  - name: detects-sql-injection
    input: |
      Review this code:
      query = f"SELECT * FROM users WHERE id = {user_id}"
    expected:
      output_contains: ["SQL injection", "parameterized"]
      output_not_contains: ["looks good", "no issues"]

  - name: approves-safe-code
    input: |
      Review this code:
      query = db.execute("SELECT * FROM users WHERE id = ?", [user_id])
    expected:
      output_contains: ["secure", "parameterized"]
      output_not_contains: ["vulnerability", "injection"]
```

Run it:

```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY=your-key

# Option 2: Create .env.local file (auto-loaded)
echo "ANTHROPIC_API_KEY=your-key" > .env.local

# Run the tests
evalview skill test tests/code-reviewer.yaml
```

### Example Output

```
━━━ Running Skill Tests ━━━

Suite:  test-code-reviewer
Skill:  ./skills/code-reviewer/SKILL.md
Model:  claude-sonnet-4-20250514
Tests:  2

Results:

  PASS detects-sql-injection
  PASS approves-safe-code

Summary: ✓
  Pass rate: 100% (2/2)
  Avg latency: 1,240ms
  Total tokens: 3,847
```

---

## Why Test Skills?

**You can test skills manually in Claude Code. So why use EvalView?**

Manual testing works for development. EvalView is for **automation**:

| Manual Testing | EvalView |
|----------------|----------|
| Test while you write | **Test on every commit** |
| You remember to test | **CI blocks bad merges** |
| Test a few cases | **Test 50+ scenarios** |
| "It works for me" | **Reproducible results** |
| Catch bugs after publish | **Catch bugs before publish** |

**Who needs automated skill testing?**

- **Skill authors** publishing to marketplaces
- **Enterprise teams** rolling out skills to thousands of employees
- **Open source maintainers** accepting contributions from the community
- **Anyone** who wants CI/CD for their skills

Skills are code. Code needs tests. EvalView brings the rigor of software testing to the AI skills ecosystem.

---

## Compatible Platforms

| Platform | Status |
|----------|--------|
| Claude Code | Supported |
| Claude.ai Skills | Supported |
| OpenAI Codex CLI | Same SKILL.md format |
| OpenClaw | Supported (AgentSkills / SKILL.md format) |
| Custom Skills | Any SKILL.md file |

---

## CLI Commands

### `evalview skill validate`

```bash
evalview skill validate PATH [OPTIONS]

Options:
  -r, --recursive    Validate all skills in directory
  --json             Output as JSON (CI-friendly)
```

### `evalview skill test`

```bash
evalview skill test TEST_FILE [OPTIONS]

Options:
  --model TEXT       Claude model to use (default: claude-sonnet-4-20250514)
  --agent TEXT       Agent type: system-prompt, claude-code, codex, openclaw,
                     langgraph, crewai, openai-assistants, custom
```

### Testing with OpenClaw

OpenClaw uses [AgentSkills](https://docs.openclaw.ai/tools/skills) (SKILL.md files with YAML frontmatter) to extend its capabilities. EvalView supports testing OpenClaw skills directly:

```bash
# Test a skill through the OpenClaw CLI
evalview skill test tests.yaml --agent openclaw

# YAML configuration for OpenClaw
```

```yaml
# tests/my-openclaw-skill.yaml
name: test-my-skill
skill: ./skills/my-skill/SKILL.md
agent:
  type: openclaw
  timeout: 120
  max_turns: 10

tests:
  - name: basic-test
    input: "Use the skill to do something"
    expected:
      output_contains: ["expected result"]
      files_created: ["output.txt"]
```

OpenClaw must be installed and accessible in your PATH (`pip install openclaw`).

### `evalview skill doctor`

```bash
evalview skill doctor PATH
```

Diagnoses common issues:
- Character budget exceeded
- Duplicate skill names
- Invalid skill structure
- Missing required fields

---

## Related Documentation

- [CLI Reference](CLI_REFERENCE.md)
- [CI/CD Integration](CI_CD.md)
