"""Week 2 Day 6: does temperature change whether the agent SUCCEEDS?

Day 5 measured *text* variance — how much the words wobble. That's cosmetic.
The question that actually matters for an agent is: does turning temperature up
make it get the task WRONG more often? A creative paraphrase is fine; a wrong
number is a failure.

So here we change exactly one variable — temperature — and hold everything else
fixed (same agent, same tools, same eval tasks, same automatic grader). For each
temperature we run every eval task N times and score pass/fail with the same
`expected_substring` check the real eval harness uses (evals/run_evals.py). Then
we compare success RATE across temperatures.

The hypothesis worth forming before you look: temperature 0 and 0.7 are both
usually fine because the tool results pin the answer, but 1.2 starts to break
the exact-substring tasks (a stray "$730.65 total" vs "730.65", a reordered
computation, an over-creative answer) even though the plan still looks right.
Whatever you find, the number is the point — never eyeball it.

    3 temperatures x 5 runs x 8 tasks = 120 agent runs by default.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week02/day6_temperature_success.py
    python experiments/week02/day6_temperature_success.py --runs 5 --temps 0 0.7 1.2
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from agent import MODEL, run_agent

# Reuse the SHARED eval task set — the same tasks the compounding harness uses,
# so this experiment measures the agent you're actually shipping.
TASKS_FILE = Path(__file__).resolve().parents[2] / "evals" / "tasks.json"

DEFAULT_TEMPS = [0.0, 0.7, 1.2]


def score_run(task: dict, temperature: float) -> dict:
    """Run one task once at `temperature`; grade with expected_substring."""
    try:
        # capture=True → silent structured trace; we still only grade the text,
        # exactly like evals/run_evals.py, so the numbers stay comparable.
        trace = run_agent(task["prompt"], temperature=temperature, capture=True)
        answer = trace["answer"]
        ok = task["expected_substring"] in answer
        signature = trace["tool_signature"] or "(no tools)"
        err = None
    except Exception as e:
        answer, ok, signature, err = f"[error: {e}]", False, "(error)", str(e)
    return {"answer": answer, "pass": ok, "tool_signature": signature, "error": err}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="runs per (task, temp) (default 5)")
    parser.add_argument("--temps", type=float, nargs="+", default=DEFAULT_TEMPS,
                        help="temperatures to sweep (default 0 0.7 1.2)")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    tasks = json.loads(TASKS_FILE.read_text())
    print(f"model: {MODEL}   tasks: {len(tasks)}   "
          f"temps: {args.temps}   runs/task: {args.runs}")

    results: list[dict] = []
    # per-temp[task_id] -> passes; and per-temp total passes
    per_task: dict[float, dict[str, int]] = {t: {} for t in args.temps}
    per_temp_pass: dict[float, int] = {t: 0 for t in args.temps}
    per_temp_total: dict[float, int] = {t: 0 for t in args.temps}

    for temperature in args.temps:
        print(f"\n{'=' * 70}\nTEMPERATURE {temperature}\n{'=' * 70}")
        for task in tasks:
            passes = 0
            for i in range(args.runs):
                r = score_run(task, temperature)
                passes += r["pass"]
                results.append({
                    "temperature": temperature, "task": task["id"], "run": i,
                    "pass": r["pass"], "tool_signature": r["tool_signature"],
                    "answer": r["answer"], "error": r["error"],
                })
            per_task[temperature][task["id"]] = passes
            per_temp_pass[temperature] += passes
            per_temp_total[temperature] += args.runs
            rate = passes / args.runs
            mark = "✅" if rate == 1 else "❌" if rate == 0 else "⚠️"
            print(f"  {task['id']:<28} {passes}/{args.runs}  {mark}")

    # ---- headline: success rate by temperature ----
    print(f"\n{'=' * 70}\nSUCCESS RATE BY TEMPERATURE\n{'=' * 70}")
    header = f"{'temp':>6}{'passed':>10}{'total':>8}{'rate':>9}"
    print(header)
    print("-" * len(header))
    summary_by_temp = {}
    for t in args.temps:
        rate = per_temp_pass[t] / per_temp_total[t] if per_temp_total[t] else 0.0
        summary_by_temp[t] = {"passed": per_temp_pass[t], "total": per_temp_total[t], "rate": rate}
        print(f"{t:>6}{per_temp_pass[t]:>10}{per_temp_total[t]:>8}{rate:>8.0%}")

    # ---- per-task breakdown: which tasks are temperature-sensitive? ----
    print(f"\n{'=' * 70}\nPASS COUNT PER TASK  (out of {args.runs} runs each)\n{'=' * 70}")
    tcols = "".join(f"{('t=' + str(t)):>10}" for t in args.temps)
    print(f"{'task':<28}{tcols}")
    print("-" * (28 + 10 * len(args.temps)))
    for task in tasks:
        row = "".join(f"{per_task[t][task['id']]:>10}" for t in args.temps)
        print(f"{task['id']:<28}{row}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"day6_results_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL, "runs": args.runs, "temps": args.temps,
                "summary_by_temp": {str(k): v for k, v in summary_by_temp.items()},
                "pass_per_task": {str(k): v for k, v in per_task.items()},
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
