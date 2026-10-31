---
name: week-experiment
description: Help finish a weekXX dayXX experiment from the learning plan. When the user asks to "finish the week04 day1 experiment" or "run the experiment for week02 day6" or similar week-day experiment tasks. Reads PLAN.md for the day's build spec, explores existing code, implements or runs the experiment, and saves results.
---

# Week Experiment Runner

Help the user complete a specific day's **experiment** (usually Day 6, sometimes earlier days with a Build task) from the AI Agent Engineering learning plan.

## Step 1 — Parse the target

Extract `weekXX` and `dayY` from the user's request. If ambiguous, ask.
- **weekXX** is always two digits: `week04`, `week02`, etc.
- **dayY** is a digit 1–7.

## Step 2 — Read the plan

Read `PLAN.md` and find the row for the target week + day in the table. The **Build** column tells you exactly what to do. Print the relevant plan line so the user sees the task spec.

## Step 3 — Explore the codebase

Check what already exists:
- `experiments/{weekXX}/` — existing experiment scripts, results, and agent code
- `evals/run_evals.py` — the root eval harness (may need extending)
- `evals/tasks.json` — existing eval tasks (may need new ones)
- Earlier weeks' agent modules (e.g. `experiments/week03/agent.py`) — these are often reused

## Step 4 — Identify the experiment type

The plan's Day 6 row (or the specific day's Build) falls into one of these patterns:

### A. Eval-task experiment (Week 1, 4, etc.)
- Create or update tasks in `experiments/{weekXX}/evals/tasks.json` (or root `evals/tasks.json`)
- Write or reuse `experiments/{weekXX}/evals/run_evals.py`
- Run with ≥5 runs per task: `python experiments/{weekXX}/evals/run_evals.py --runs 5`
- Save results as `dayY_results_{timestamp}.json`

### B. Parameter-sweep experiment (Week 2, 3, 5, etc.)
- Vary ONE variable (e.g. temperature, tool count, prompt style)
- Write `experiments/{weekXX}/run.py` that iterates over conditions
- Run ≥5 runs per condition
- Save results with condition labels

### C. Comparison experiment (Week 9–12)
- Run two agent variants on the same tasks
- Write a harness that runs both sequentially
- Score and compare

## Step 5 — Implement

For new experiments:
- Follow existing patterns from earlier weeks (e.g. `experiments/week03/day7_evals.py`)
- Reuse the agent module from the most recent week that has the required tools
- Always use non-interactive mode (no permission prompts) for reproducibility
- Include token/cost tracking where the agent already supports it
- Save timestamped JSON results

For existing experiments (re-running):
- Just run the existing script with appropriate `--runs`

## Step 6 — Verify

After running:
- Check the results JSON — confirm pass rates, token counts
- Report the summary to the user: total passed, per-task breakdown, any failures
- Note any flaky tasks (pass rate between 0% and 100%)

## Conventions

- Results filename: `day{Y}_results_{YYYYMMDD_HHMMSS}.json`
- Always ≥5 runs per condition for experiments
- Never delete old results — they are the audit trail
- If the experiment requires the agent to call APIs, remind the user to set `DASHSCOPE_API_KEY`
- Model: check `MODEL_NAME` env var, default is `qwen3.7-plus`