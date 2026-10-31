---
name: week-report
description: Help write a weekXX report for the learning plan. When the user asks to "write the week04 report" or "finish the week03 day7 report" or similar. Reads experiment results, log files, and PLAN.md context, then writes the report to reports/week{N}.md following the established template.
---

# Week Report Writer

Help the user write a report for a specific week (or day) of the AI Agent Engineering learning plan.

## Step 1 — Parse the target

Extract `weekXX` from the user's request. Optionally a specific day. If ambiguous, ask.

## Step 2 — Gather context

Read these files to understand what happened:
- `PLAN.md` — the week's goals, each day's Learn + Build spec
- `reports/week{N}.md` — existing report (if any, for continuation)
- `experiments/{weekXX}/` — all experiment scripts and result JSON files
- `evals/last_results.json` — if the root eval harness was used
- `logs/` — any daily log files written that week

## Step 3 — Read the experiment results

Find all `*_results_*.json` files in `experiments/{weekXX}/` and its subdirectories. For each:
- Extract: model, date, runs_per_task, total_passed, total_runs, pass_rate, per_task_passes
- Compare multiple runs of the same experiment if available (e.g. Day 1 substring-only vs Day 2 mixed-checkers)
- Note which tasks consistently pass vs. which are flaky

## Step 4 — Identify the key findings

Look for:
- **Surprises** — results that contradict expectations (e.g. a task went from 100%→0% when switching checkers)
- **Patterns** — do hard tasks actually fail more? Do certain tool types cause problems?
- **Regressions** — did a change make things worse?
- **False positives** — did substring scoring hide real failures?

## Step 5 — Write the report

Write to `reports/week{N}.md`. Follow this structure:

```markdown
# Week N Report — <topic from PLAN.md>

**Dates:** YYYY-MM-DD to YYYY-MM-DD

## Day 1 — <learn topic>
<!-- Learning notes: key concepts, definitions, mental models -->

## Day 2 — <learn topic>
<!-- ... -->

## Day N Experiment — <experiment name>

### Question
<!-- One sentence, falsifiable -->

### Setup
- **Model:**
- **Tasks:**
- **Varied:** (the ONE thing you changed)
- **Held constant:** (everything else)
- **Runs per condition:**

### Result

| Condition | Success rate | Notes |
|-----------|-------------|-------|
| A         |             |       |

### Surprise
<!-- What didn't match your expectation -->

### Next question
<!-- Feeds next week's experiment -->

## What I learned this week (beyond the experiment)
<!-- Synthesis across all days -->
```

## Style guidelines

- **Numbers over adjectives.** Write "45/50 (90%)" not "mostly passed."
- **Show the data.** Include a results table for every experiment.
- **Be honest about failures.** A 0% pass rate on a task is more interesting than 100%.
- **Connect to the plan.** Tag findings with the week's goal from PLAN.md.
- **Keep learning notes concise.** The Day 1–5 sections are reference notes, not essays. Bullet points, tables, and short paragraphs.
- **The Surprise section is the most important.** If nothing surprised you, you weren't paying attention.

## After writing

- Confirm the file was saved to `reports/week{N}.md`
- Show the user a summary of what was written
- Ask if they want to add anything from their own observations