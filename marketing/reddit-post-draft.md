# Reddit Marketing Post Draft - EvalView

## Target Subreddits
- r/LangChain (most relevant - LangGraph users)
- r/LocalLLaMA (Ollama users, DIY crowd)
- r/MachineLearning (broader reach)
- r/LLMDevs (if it exists)
- r/artificial (general AI discussion)

---

## Post Option 1: "The Question" Approach (Recommended)
**Best for:** r/LangChain, r/LocalLLaMA

### Title:
**How are you catching agent regressions before users do? Our approach after getting burned**

### Body:

After our LangGraph agent started hallucinating tool calls in production last month (after a "safe" prompt tweak), I became obsessed with solving this problem properly.

**The painful cycle we kept repeating:**
1. Change prompt/model/tool
2. Manual spot-check: "looks good"
3. Ship it
4. User reports something broke
5. Spend 2 days reproducing
6. Fix it, break something else
7. Repeat

The LangChain survey saying 32% cite quality as the top production barrier hit home. We were flying blind.

**What we tried:**
- **Observability tools** (Langfuse, etc.) → Great for debugging *after* users complain. Doesn't prevent bad deploys.
- **Generic eval frameworks** → Designed for model benchmarks, not agent behavior. No regression detection.
- **Custom eval harnesses** → We spent 3 weeks building one. It was brittle and nobody maintained it.

**What actually worked:**

We ended up building something that treats agent testing like pytest treats code: save a "golden" trace when things work, then diff against it on every change.

The key insight was tracking **4 states**, not just pass/fail:
- **PASSED** - Matches baseline, ship it
- **TOOLS_CHANGED** - Agent used different tools (requires review)
- **OUTPUT_CHANGED** - Same tools, different output (might be fine, might not)
- **REGRESSION** - Score dropped (fix before deploy)

Then we wired it into CI to actually *block* merges when things regress, not just alert.

For the non-determinism problem (same input → different outputs), we added statistical mode: run the test 10x, check if 80% pass. That eliminated our flaky test problem.

We open-sourced it as [evalview](https://github.com/hidai25/eval-view) since we figured others are dealing with the same pain.

**Curious how other teams handle this:**
- Do you have regression testing for agent behavior?
- How do you handle model upgrades? (We saw GPT-4o → 4.1 drop our injection resistance from 94% to 71% on one agent)
- Are you blocking deploys on eval failures, or just monitoring?

Would love to hear what's working for others. Still iterating on this.

---

## Post Option 2: "The Story" Approach
**Best for:** r/MachineLearning, r/artificial

### Title:
**We stopped shipping broken agents by treating prompts like code (lessons from 6 months of pain)**

### Body:

Six months ago, our team had a recurring nightmare: "the agent worked yesterday."

We'd tweak a system prompt, the agent would pass our manual checks, we'd ship... and then hear from customers that it was calling the wrong tools or hallucinating data. Every. Single. Time.

After reading the Anthropic engineering blog post on evals, something clicked: **we were treating agent development like it was 2005-era web development.** No tests. No CI. Just vibes and prayers.

**The core problem:**

Observability tools (Langfuse, DataDog LLM) are amazing at telling you *what* your agent did. But they're reactive - you find out something's broken after users complain. We needed something that would fail CI *before* bad code merges.

**What changed everything:**

We started thinking about agent testing differently:

1. **Golden baselines** - When the agent works correctly, save that trace. Every future run diffs against it.

2. **Tool change detection** - Prompt changes often cause agents to use *different* tools without obviously "failing". We needed to catch this explicitly.

3. **Statistical testing** - LLMs are non-deterministic. A test that passes 7/10 times isn't reliable. We run tests 10x and require 80%+ pass rate.

4. **CI gates that actually block** - Not dashboards that nobody checks. Actual merge blocking.

We packaged this into an open-source tool called [evalview](https://github.com/hidai25/eval-view) and it's been running in our CI for 4 months now.

**Results:**
- Caught 3 regressions before production in the first month
- Model upgrade testing went from "2 weeks of manual QA" to "overnight CI run"
- Our "it worked yesterday" complaints dropped to near zero

**The reality check:**

This doesn't catch everything. Agents can still fail in production in novel ways. But the *obvious* regressions - the ones that would've been embarrassing - now get caught automatically.

For anyone building agents with LangGraph, CrewAI, or even custom setups: I'd really recommend adding some form of regression testing. Doesn't have to be our tool - even a simple "save expected outputs and diff" script is better than nothing.

Happy to answer questions about our setup.

---

## Post Option 3: "The Hot Take" Approach
**Best for:** Engagement/controversy, r/LocalLLaMA

### Title:
**Hot take: "Observability" is not the same as "testing" and the agent ecosystem has it backwards**

### Body:

I keep seeing teams add LangSmith/Langfuse/Phoenix and think they've solved agent reliability. They haven't.

Observability tells you what happened. Testing tells you whether you should ship.

**The gap:**

| | Observability | Testing |
|--|--|--|
| When | After production | Before production |
| Action | Debug & alert | Block bad deploys |
| Output | Dashboards | Pass/fail gates |

Right now, the agent ecosystem has great observability but terrible testing. We have:

✅ Traces
✅ Token counts
✅ Latency metrics
✅ Prompt history

But we don't have:

❌ Regression detection (did the agent get worse?)
❌ Tool change detection (is it calling different tools?)
❌ CI gates (block merge if quality drops)
❌ Statistical testing (handle non-determinism)

**The result:**

Teams "observe" their way into production failures instead of testing their way out of them.

**What should exist:**

pytest for agents. Something that:
1. Saves a "golden" baseline when things work
2. Diffs every change against that baseline
3. Fails CI on regressions
4. Handles LLM non-determinism with statistical runs

I built this for my team ([evalview](https://github.com/hidai25/eval-view) if curious), but the broader point is: **the industry needs to shift from "observe and react" to "test and prevent"**.

Change my mind?

---

## Post Option 4: "Seeking Feedback" Approach
**Best for:** Appears humble, gets engagement

### Title:
**Built an open-source agent testing framework - looking for feedback from teams in production**

### Body:

After getting burned by silent agent regressions one too many times, I built a testing framework specifically for AI agents. Would love feedback from others who've dealt with this problem.

**The problem I was solving:**

My LangGraph agent would "pass" manual testing but fail in subtle ways in production:
- Using wrong tools after prompt changes
- Output quality degrading after model upgrades
- Latency spiking without anyone noticing
- Hallucinations sneaking through

Observability tools helped debug, but didn't *prevent* bad deploys.

**My solution (evalview):**

Think pytest but for agents:

```yaml
# test-case.yaml
name: "Book flight test"
input:
  query: "Book a flight to NYC for tomorrow"
expected:
  tools:
    - search_flights
    - book_flight
  output:
    contains: ["confirmation", "NYC"]
thresholds:
  min_score: 80
  max_cost: 0.10
```

Key features:
- **Golden baselines** - Save working traces, diff future runs
- **4-state diff** - PASSED / TOOLS_CHANGED / OUTPUT_CHANGED / REGRESSION
- **Statistical mode** - `--runs 10 --pass-rate 0.8` for non-deterministic outputs
- **CI integration** - Blocks merges on regression
- **Works with** LangGraph, CrewAI, Ollama, OpenAI, custom APIs

**What I'm unsure about:**

1. Is YAML the right format for tests? Or should it be Python?
2. Is 4-state diff (PASSED/TOOLS_CHANGED/OUTPUT_CHANGED/REGRESSION) the right granularity?
3. What's a reasonable default pass rate for statistical mode?

Repo: https://github.com/hidai25/eval-view

Genuinely looking for feedback - what would make this useful for your workflow?

---

## Engagement Strategy

### Initial Post
- Post during US morning hours (9-11am EST) for maximum visibility
- Respond to every comment within first 2 hours
- Ask follow-up questions to keep threads going

### Key Talking Points to Weave In
- "It worked yesterday" is the worst phrase in agent development
- Model upgrades break things in subtle ways (cite the GPT-4o → 4.1 regression example)
- Observability ≠ Testing (this is a hot take that generates discussion)
- Non-determinism requires statistical testing, not single runs
- YAML tests > Python tests for agent configs (lower barrier)
- Works offline with Ollama (appeals to r/LocalLLaMA)

### Comments to Seed (from alt account or ask a friend)
- "Does this work with CrewAI?" → Yes, built-in adapter
- "How do you handle non-deterministic outputs?" → Statistical mode with configurable pass rates
- "What's the difference between this and Langfuse?" → Langfuse = observability (after the fact), evalview = testing (before deploy)
- "Can I use this with local models?" → Yes, Ollama adapter included, fully offline

### Awards/Upvotes
- First 30 minutes are critical for Reddit algorithm
- Cross-post to multiple relevant subs (with slight title variations)
- Share in relevant Discord servers (LangChain, AI Engineering, etc.)

---

## Don't Do
- ❌ Don't shill too hard - let the value speak
- ❌ Don't post identical content to multiple subs (Reddit penalizes this)
- ❌ Don't ignore criticism - engage genuinely
- ❌ Don't claim it's perfect - acknowledge limitations
- ❌ Don't compare directly to paid competitors (looks defensive)

---

## Hashtags/Flairs to Use
- LangChain: "Discussion" or "Tools"
- LocalLLaMA: "Discussion" or "Resources"
- MachineLearning: "Discussion" or "Project"
