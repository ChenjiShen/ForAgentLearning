# Experiments

One folder per weekly experiment: `experiments/week01_tool_descriptions/`, etc.

Each experiment folder should contain:

```
weekNN_short_name/
├── run.py          # the experiment script (runs both conditions N times, prints the table)
├── results.json    # raw per-run results (keep them — you'll want to re-analyze)
└── notes.md        # anything that doesn't fit the report
```

## Rules for a valid experiment

1. **Change one thing.** Two conditions differing in exactly one variable.
2. **Run each condition at least 5 times.** Agent behavior is noisy; one run proves nothing.
3. **Score automatically.** Pass/fail per task, decided by code, not by eyeballing.
4. **Record raw results.** Save every run to `results.json`, aggregate afterward.

## Starter experiment ideas

- Terse vs. detailed tool descriptions → wrong-tool-call rate
- How error-message wording in `tool_result` (with `is_error: true`) affects recovery
- Task success rate vs. number of available tools (3 vs 10 vs 30)
- Effect of `effort` level (low / medium / high) on success rate and token cost
- One agent with 10 tools vs. a router + two specialists
- Does asking the agent to plan first improve multi-step task success?