# Skill Testing Starter

Copy this folder to add CI testing to your Claude Code skills.

## Quick Start

1. Copy `.github/workflows/skill-tests.yml` to your repo
2. Add your `ANTHROPIC_API_KEY` to GitHub Secrets
3. Put your skills in `.claude/skills/`
4. Create test files in `tests/`

## File Structure

```
your-repo/
├── .github/
│   └── workflows/
│       └── skill-tests.yml   # Copy from here
├── .claude/
│   └── skills/
│       └── your-skill/
│           └── SKILL.md
└── tests/
    └── your-skill-tests.yaml
```

## Test File Format

```yaml
name: my-skill-tests
skill: .claude/skills/your-skill/SKILL.md

tests:
  - name: test-name
    input: "Your prompt here"
    expected:
      output_contains: ["expected", "words"]
      output_not_contains: ["unwanted"]
```

## Run Locally

```bash
pip install evalview

# Validate structure
evalview skill validate .claude/skills/ -r

# Test behavior
echo "ANTHROPIC_API_KEY=your-key" > .env.local
evalview skill test tests/your-skill-tests.yaml
```
