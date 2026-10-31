"""Week 4 Day 2: Run the week04 eval tasks against the week03 sandboxed agent.

Each task supports a ``checker`` field that controls how the answer is scored:

  - ``"substring"`` (default) — ``expected_substring in answer``
  - ``"exact"``       — ``answer.strip() == expected``
  - ``"python"``      — execute ``check_code`` (a snippet that sets ``passed``)
    The snippet has access to: ``answer`` (str), ``sandbox_root`` (Path),
    and a pre-defined ``passed = False``.  Set ``passed = True`` when the
    answer (or a produced file) satisfies the task.

Usage:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week04/evals/run_evals.py
    python experiments/week04/evals/run_evals.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the week03 agent (reuse its sandbox, tools, and agent loop)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.week03.agent import (  # noqa: E402
    MODEL,
    SANDBOX_ROOT,
    ALLOWED_EXECUTABLES,
    TOOLS,
    TOOL_FUNCTIONS,
    _make_error,
    ErrorType,
)
from openai import OpenAI  # noqa: E402

TASKS_FILE = _THIS_DIR / "tasks.json"
HISTORY_FILE = _PROJECT_ROOT / "evals" / "history.jsonl"


# ============================================================================
# Checker dispatch
# ============================================================================


def _llm_judge(task_prompt: str, rubric: str, answer: str) -> tuple[bool, str]:
    """Call a separate LLM to judge the answer against a rubric.

    Returns (passed, judge_raw_response).

    Mitigations for known biases:
      - The rubric appears BEFORE the answer so the judge reads criteria first.
      - We ask for PASS/FAIL as the first token to reduce verbosity bias.
      - A deterministic temperature (0) is used for reproducibility.

    Known limitation: using the same model as the agent introduces
    self-preference bias.  A truly independent eval uses a different model.
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    judge_prompt = (
        "You are an impartial evaluator. Your job is to judge whether an AI "
        "assistant's answer correctly addresses a given task.\n\n"
        "## Task\n"
        f"{task_prompt}\n\n"
        "## Rubric (what a correct answer must satisfy)\n"
        f"{rubric}\n\n"
        "## Assistant's Answer\n"
        f"{answer}\n\n"
        "## Instructions\n"
        "Compare the answer against the rubric. Be strict but fair:\n"
        "- PASS: the answer correctly and completely satisfies the rubric.\n"
        "- FAIL: the answer is wrong, incomplete, or does not meet the rubric.\n\n"
        "Reply with EXACTLY one word: PASS or FAIL."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
    except Exception as exc:
        return False, f"llm_judge (API error: {exc})"

    if verdict.startswith("PASS"):
        return True, f"llm_judge ({verdict[:50]})"
    elif verdict.startswith("FAIL"):
        return False, f"llm_judge ({verdict[:50]})"
    else:
        # The judge didn't follow the format — treat as ambiguous → fail
        return False, f"llm_judge (unexpected response: {verdict[:80]})"


def check_task(task: dict, answer: str, sandbox_root: Path | None = None) -> tuple[bool, str]:
    """Score *answer* against *task* using its declared checker.

    Returns (passed, checker_name).
    """
    checker = task.get("checker", "substring")
    sandbox = sandbox_root or SANDBOX_ROOT

    # -- substring (default, backward-compatible) ----------------------------
    if checker == "substring":
        expected = task.get("expected_substring", "")
        return expected in answer, "substring"

    # -- exact -----------------------------------------------------------------
    if checker == "exact":
        expected = task.get("expected", "")
        return answer.strip() == expected.strip(), "exact"

    # -- python (code snippet) -------------------------------------------------
    if checker == "python":
        code = task.get("check_code", "")
        if not code:
            return False, "python (no check_code provided)"

        namespace: dict = {
            "answer": answer,
            "sandbox_root": sandbox,
            "passed": False,
            "details": "",
        }
        try:
            exec(code, namespace)
        except Exception as exc:
            return False, f"python (check_code raised {type(exc).__name__}: {exc})"

        passed = bool(namespace.get("passed", False))
        return passed, "python"

    # -- llm_judge (second API call with rubric) ------------------------------
    if checker == "llm_judge":
        rubric = task.get("rubric", "")
        if not rubric:
            return False, "llm_judge (no rubric provided)"
        return _llm_judge(task["prompt"], rubric, answer)

    # -- unknown checker -------------------------------------------------------
    return False, f"unknown checker {checker!r}"


# ============================================================================
# Non-interactive agent loop (same pattern as week03 day7_evals.py)
# ============================================================================


def run_agent_noninteractive(
    user_input: str,
    *,
    max_turns: int = 12,
) -> str:
    """Run the week03 agent silently — no prints, no permission prompts."""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "You are a helpful assistant with sandboxed file, command, and web tools. "
        f"All file paths and shell commands run inside {SANDBOX_ROOT}. "
        "File tools cannot access paths outside the sandbox. "
        f"run_command only permits: {', '.join(sorted(ALLOWED_EXECUTABLES))} "
        "(no shell — pipes/semicolons unsupported). "
        "Complete the user's task using the available tools. Be direct and "
        "concise — when you have the answer, state it clearly and stop.\n\n"
        "IMPORTANT — All tool results use a consistent JSON envelope:\n"
        '  {"status":"success"|"error", "data":{…}, "error":null|{…}, '
        '"truncated":bool, "truncated_at":int|null}\n'
        "- On error: read .error.hint for an actionable suggestion."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )

        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return msg.content or "[no text in response]"

        if choice.finish_reason == "length":
            return "[Response was truncated — raise max_tokens.]"

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                output = _make_error(
                    ErrorType.INVALID_INPUT,
                    f"Invalid JSON arguments: {e}",
                )
            else:
                try:
                    output = TOOL_FUNCTIONS[tc.function.name](args)
                except Exception as e:
                    output = _make_error(ErrorType.UNKNOWN, str(e))
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": output}
            )

    return "[Agent stopped after 12 turns — possible loop.]"


# ============================================================================
# Scoring
# ============================================================================


def score_run(task: dict) -> dict:
    """Run one task once and return a scored result record."""
    checker_used = "?"
    try:
        answer = run_agent_noninteractive(task["prompt"])
        passed, checker_used = check_task(task, answer)
    except Exception as e:
        answer = f"[exception: {e}]"
        passed, checker_used = False, "exception"

    return {
        "task_id": task["id"],
        "pass": passed,
        "answer": answer,
        "checker": checker_used,
    }


# ============================================================================
# History tracking (Week 4 Day 5 — regression tracking)
# ============================================================================


def append_history(
    model: str,
    agent_version: str,
    runs_per_task: int,
    total_passed: int,
    total_runs: int,
    per_task_passes: dict[str, int],
    date: str,
) -> None:
    """Append a one-line JSON summary to evals/history.jsonl."""
    record = {
        "date": date,
        "agent_version": agent_version,
        "model": model,
        "runs_per_task": runs_per_task,
        "total_passed": total_passed,
        "total_runs": total_runs,
        "pass_rate": round(total_passed / total_runs, 4) if total_runs else 0.0,
        "per_task_passes": per_task_passes,
    }
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"📝 History appended → {HISTORY_FILE}")


def compare_with_previous(
    current_rate: float,
    current_passed: int,
    current_runs: int,
    per_task_passes: dict[str, int],
    runs_per_task: int,
) -> None:
    """Read the previous entry from history.jsonl and print a comparison."""
    if not HISTORY_FILE.exists():
        print("\n📊 No previous run to compare against (first run).")
        return

    # Read all lines, get the last one (most recent previous run)
    lines = [ln.strip() for ln in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print("\n📊 No previous run to compare against (empty history).")
        return

    try:
        prev = json.loads(lines[-1])
    except json.JSONDecodeError:
        print("\n⚠️  Could not parse previous history entry.")
        return

    prev_rate = prev.get("pass_rate", 0.0)
    prev_passed = prev.get("total_passed", 0)
    prev_runs = prev.get("total_runs", 0)
    prev_date = prev.get("date", "unknown")
    prev_agent = prev.get("agent_version", "?")
    prev_task_passes = prev.get("per_task_passes", {})
    prev_runs_per = prev.get("runs_per_task", runs_per_task)

    # Compute delta
    delta_rate = current_rate - prev_rate
    delta_str = (
        f"+{delta_rate:+.1%}" if delta_rate > 0
        else f"{delta_rate:+.1%}" if delta_rate < 0
        else "unchanged"
    )

    print("\n" + "=" * 60)
    print("REGRESSION CHECK — Comparison with Previous Run")
    print("=" * 60)
    print(f"  Previous: {prev_date}  |  agent={prev_agent}  |  "
          f"{prev_passed}/{prev_runs}  ({prev_rate:.0%})")
    print(f"  Current:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
          f"agent=week03  |  {current_passed}/{current_runs}  ({current_rate:.0%})")
    print(f"  Delta:    {delta_str}  "
          f"({current_passed - prev_passed:+d} passes)")

    # Per-task comparison (only for tasks that appear in both runs)
    if prev_task_passes:
        regressions: list[tuple[str, float, float]] = []
        improvements: list[tuple[str, float, float]] = []
        same: list[str] = []

        for task_id, passes in per_task_passes.items():
            if task_id in prev_task_passes:
                cur_rate = passes / runs_per_task
                prev_t_rate = prev_task_passes[task_id] / max(prev_runs_per, 1)
                if cur_rate < prev_t_rate:
                    regressions.append((task_id, prev_t_rate, cur_rate))
                elif cur_rate > prev_t_rate:
                    improvements.append((task_id, prev_t_rate, cur_rate))
                else:
                    same.append(task_id)

        if regressions:
            print(f"\n  🔴 REGRESSIONS ({len(regressions)} task{'s' if len(regressions) > 1 else ''}):")
            for tid, old_r, new_r in regressions:
                print(f"     {tid}:  {old_r:.0%} → {new_r:.0%}  ({new_r - old_r:+.0%})")

        if improvements:
            print(f"\n  🟢 IMPROVEMENTS ({len(improvements)} task{'s' if len(improvements) > 1 else ''}):")
            for tid, old_r, new_r in improvements:
                print(f"     {tid}:  {old_r:.0%} → {new_r:.0%}  ({new_r - old_r:+.0%})")

        if same and not regressions and not improvements:
            print("\n  ✅ All tasks unchanged from previous run.")

        # New/removed tasks
        new_tasks = set(per_task_passes) - set(prev_task_passes)
        removed_tasks = set(prev_task_passes) - set(per_task_passes)
        if new_tasks:
            print(f"\n  🆕 NEW TASKS ({len(new_tasks)}): {', '.join(sorted(new_tasks))}")
        if removed_tasks:
            print(f"\n  🗑️  REMOVED TASKS ({len(removed_tasks)}): {', '.join(sorted(removed_tasks))}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Runs per task (default 3).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    tasks = json.loads(TASKS_FILE.read_text())

    print(f"Model:      {MODEL}")
    print(f"Tasks:      {len(tasks)}")
    print(f"Runs/task:  {args.runs}")
    print(f"Sandbox:    {SANDBOX_ROOT.resolve()}")
    print()

    all_results: list[dict] = []
    per_task_passes: dict[str, int] = {}

    for task in tasks:
        passes = 0
        for i in range(args.runs):
            r = score_run(task)
            passes += r["pass"]
            all_results.append(r)

            icon = "✅" if r["pass"] else "❌"
            preview = r["answer"][:120].replace("\n", " ")
            print(f"  [{icon}] {task['id']:<32} run {i + 1}/{args.runs}  "
                  f"→ {preview}")

        per_task_passes[task["id"]] = passes
        rate = passes / args.runs
        mark = "✅" if rate == 1.0 else "❌" if rate == 0.0 else "⚠️"
        difficulty = task.get("difficulty", "?")
        declared = task.get("checker", "substring")
        print(f"  {mark} {task['id']} [{difficulty}, {declared}]: {passes}/{args.runs} passed ({rate:.0%})")
        print()

    # ---- Summary ----
    total_passed = sum(r["pass"] for r in all_results)
    total_runs = len(all_results)
    overall_rate = total_passed / total_runs if total_runs else 0.0

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total: {total_passed}/{total_runs} passed ({overall_rate:.0%})")
    print()

    # Per-task pass count
    print(f"{'task':<34} {'difficulty':<10} {'passed':>8}  {'rate':>6}")
    print("-" * 62)
    for task in tasks:
        p = per_task_passes[task["id"]]
        d = task.get("difficulty", "?")
        print(f"{task['id']:<34} {d:<10} {p:>3}/{args.runs:<3}  {p/args.runs:>5.0%}")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = _THIS_DIR / f"day5_results_{timestamp}.json"
    report = {
        "experiment": "week04_day5_regression",
        "model": MODEL,
        "date": run_date,
        "runs_per_task": args.runs,
        "total_passed": total_passed,
        "total_runs": total_runs,
        "pass_rate": overall_rate,
        "per_task_passes": per_task_passes,
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")

    # ---- Regression tracking (Week 4 Day 5) ----
    append_history(
        model=MODEL,
        agent_version="week03",
        runs_per_task=args.runs,
        total_passed=total_passed,
        total_runs=total_runs,
        per_task_passes=per_task_passes,
        date=run_date,
    )
    compare_with_previous(
        current_rate=overall_rate,
        current_passed=total_passed,
        current_runs=total_runs,
        per_task_passes=per_task_passes,
        runs_per_task=args.runs,
    )


if __name__ == "__main__":
    main()
