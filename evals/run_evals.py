"""The compounding eval harness.

Runs the current agent against every task in tasks.json and scores pass/fail.
Every week, ADD the week's tasks here — never delete old ones. By week 12 your
agent is automatically tested against week 3's tasks, and regressions show up
as failing rows.

Each task supports a ``checker`` field that controls how the answer is scored:

  - ``"substring"`` (default) — ``expected_substring in answer``
  - ``"exact"``       — ``answer.strip() == expected``
  - ``"python"``      — execute ``check_code`` (a snippet that sets ``passed``)
    The snippet has access to: ``answer`` (str), ``sandbox_root`` (Path),
    and a pre-defined ``passed = False``.  Set ``passed = True`` when the
    answer (or a produced file) satisfies the task.

Run:
    python evals/run_evals.py [runs_per_task] [--agent week01|week03]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_FILE = Path(__file__).parent / "last_results.json"
HISTORY_FILE = Path(__file__).parent / "history.jsonl"


# ---- Checker dispatch -------------------------------------------------------

def _llm_judge(task_prompt: str, rubric: str, answer: str) -> tuple[bool, str]:
    """Call a separate LLM to judge the answer against a rubric.

    Returns (passed, judge_raw_response).
    """
    import os as _os

    client = __import__("openai").OpenAI(
        api_key=_os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = _os.getenv("MODEL_NAME", "qwen3.7-plus")

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
            model=model,
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
        return False, f"llm_judge (unexpected response: {verdict[:80]})"


def check_task(task: dict, answer: str, sandbox_root: Path | None = None) -> tuple[bool, str]:
    """Score *answer* against *task* using its declared checker (or substring).

    Returns (passed, checker_name) so the caller can log which checker ran.
    """
    checker = task.get("checker", "substring")

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

        # Build a clean namespace for the snippet.
        namespace: dict = {
            "answer": answer,
            "sandbox_root": sandbox_root or Path.cwd(),
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


# ---- Agent selection -------------------------------------------------------
# Map agent names to their (module_path, MODEL, run_agent) imports.
# New agents register here as the project grows.

AGENT_REGISTRY = {
    "week01": {
        "module": "week01_minimal_agent.agent",
        "sandbox_root": Path(__file__).resolve().parent.parent,
    },
    "week03": {
        "module": "experiments/week03/agent",
        "sandbox_root": (
            Path(__file__).resolve().parent.parent
            / "experiments" / "week03" / "sandbox"
        ),
    },
}

# Task-to-agent compatibility: which tasks can which agent run?
# Tasks without a tool requirement work with any agent.
# Tasks requiring week03 tools (list_dir, run_command with sandbox, etc.)
# only work with week03+.

WEEK03_ONLY_TASKS = frozenset({
    "sandbox-list-py",
    "sandbox-read-json",
    "sandbox-write-verify",
    "sandbox-count-lines",
    "sandbox-grep-urllib",
    "sandbox-multi-step",
})


def _load_agent(agent_name: str):
    """Import the agent module and return (MODEL, run_agent, kwargs)."""
    if agent_name not in AGENT_REGISTRY:
        raise SystemExit(
            f"Unknown agent: {agent_name!r}. "
            f"Available: {list(AGENT_REGISTRY)}"
        )

    registry = AGENT_REGISTRY[agent_name]
    # Insert the project root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if agent_name == "week01":
        from week01_minimal_agent.agent import MODEL, run_agent
        sandbox = Path(__file__).resolve().parent.parent
        return MODEL, run_agent, {"sandbox_root": sandbox}
    elif agent_name == "week03":
        from experiments.week03.agent import (  # type: ignore[import-not-found]
            MODEL, run_agent, SANDBOX_ROOT, TOOLS, TOOL_FUNCTIONS, ALLOWED_EXECUTABLES,
            _make_error, ErrorType,
        )
        from openai import OpenAI

        # Build a non-interactive wrapper that auto-approves commands and doesn't print
        def _run_agent_week03_noninteractive(user_input: str) -> str:
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
            for _ in range(12):
                response = client.chat.completions.create(
                    model=MODEL, tools=TOOLS, messages=messages
                )
                choice = response.choices[0]
                msg = choice.message

                assistant_msg: dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
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

        return MODEL, _run_agent_week03_noninteractive, {"sandbox_root": SANDBOX_ROOT}

    raise SystemExit(f"Agent {agent_name!r} not implemented.")


# ---- History tracking (Week 4 Day 5 — regression tracking) ------------------


def _append_history(
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


def _compare_with_previous(
    current_rate: float,
    current_passed: int,
    current_runs: int,
    per_task_passes: dict[str, int],
    runs_per_task: int,
    agent: str,
) -> None:
    """Read the previous entry from history.jsonl and print a comparison."""
    if not HISTORY_FILE.exists():
        print("\n📊 No previous run to compare against (first run).")
        return

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
          f"agent={agent}  |  {current_passed}/{current_runs}  ({current_rate:.0%})")
    print(f"  Delta:    {delta_str}  "
          f"({current_passed - prev_passed:+d} passes)")

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

        new_tasks = set(per_task_passes) - set(prev_task_passes)
        removed_tasks = set(prev_task_passes) - set(per_task_passes)
        if new_tasks:
            print(f"\n  🆕 NEW TASKS ({len(new_tasks)}): {', '.join(sorted(new_tasks))}")
        if removed_tasks:
            print(f"\n  🗑️  REMOVED TASKS ({len(removed_tasks)}): {', '.join(sorted(removed_tasks))}")

    print()


def main(runs_per_task: int = 5, agent: str = "week01") -> None:
    tasks = json.loads(TASKS_FILE.read_text())
    MODEL, run_agent, _kwargs = _load_agent(agent)
    sandbox_root = _kwargs.get("sandbox_root")

    # Filter out tasks that require tools this agent doesn't have
    if agent == "week01":
        tasks = [t for t in tasks if t["id"] not in WEEK03_ONLY_TASKS]
        if tasks_skipped := len([t for t in json.loads(TASKS_FILE.read_text())
                                if t["id"] in WEEK03_ONLY_TASKS]):
            print(f"⚠️  Skipping {tasks_skipped} week03-only task(s) — week01 agent "
                  f"doesn't have sandbox/command tools.\n")

    results = []
    per_task_passes: dict[str, int] = {}

    for task in tasks:
        passes = 0
        for i in range(runs_per_task):
            checker_used = "?"
            try:
                answer = run_agent(task["prompt"])
                ok, checker_used = check_task(task, answer, sandbox_root)
            except Exception as e:
                answer, ok, checker_used = f"[error: {e}]", False, "exception"
            passes += ok
            results.append(
                {
                    "task": task["id"],
                    "run": i,
                    "pass": ok,
                    "answer": answer,
                    "checker": checker_used,
                }
            )
        per_task_passes[task["id"]] = passes
        rate = passes / runs_per_task
        # Show which checker each task uses
        declared = task.get("checker", "substring")
        print(f"{task['id']:<28} {passes}/{runs_per_task}  "
              f"[{declared}]  "
              f"{'✅' if rate == 1 else '❌' if rate == 0 else '⚠️'}")

    total = sum(r["pass"] for r in results)
    total_runs = len(results)
    overall_rate = total / total_runs if total_runs else 0.0
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "model": MODEL,
        "agent": agent,
        "date": run_date,
        "runs_per_task": runs_per_task,
        "total_passed": total,
        "total_runs": total_runs,
        "per_task_passes": per_task_passes,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nTotal: {total}/{total_runs} passed.  Agent: {agent}  "
          f"Raw results → {RESULTS_FILE.name}")

    # ---- Regression tracking (Week 4 Day 5) ----
    _append_history(
        model=MODEL,
        agent_version=agent,
        runs_per_task=runs_per_task,
        total_passed=total,
        total_runs=total_runs,
        per_task_passes=per_task_passes,
        date=run_date,
    )
    _compare_with_previous(
        current_rate=overall_rate,
        current_passed=total,
        current_runs=total_runs,
        per_task_passes=per_task_passes,
        runs_per_task=runs_per_task,
        agent=agent,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs", type=int, nargs="?", default=5,
        help="Runs per task (default 5).",
    )
    parser.add_argument(
        "--agent", type=str, default="week01",
        choices=list(AGENT_REGISTRY),
        help="Which agent to test (default week01).",
    )
    args = parser.parse_args()
    main(args.runs, args.agent)