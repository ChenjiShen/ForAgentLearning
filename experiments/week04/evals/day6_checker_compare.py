"""Week 4 Day 6: Substring checker vs. LLM judge — disagreement analysis.

Runs all 10 eval tasks through the agent, scores each answer with BOTH:
  1. Substring checker (expected_substring in answer)
  2. LLM judge (API call with rubric)

Then reports: how often do they disagree, and which one is wrong?

The script produces a detailed JSON results file. After running, manually
review each disagreement to adjudicate ground truth — update the JSON's
"disagreements" entries with "ground_truth" (the correct label) and
"wrong_checker" (which checker was wrong).

Usage:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week04/evals/day6_checker_compare.py
    python experiments/week04/evals/day6_checker_compare.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & imports
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
from experiments.week04.evals.run_evals import _llm_judge  # noqa: E402


# ============================================================================
# Task definitions — each has expected_substring AND rubric
# ============================================================================

COMPARE_TASKS = [
    {
        "id": "knowledge-un-members",
        "prompt": "How many member states are there in the United Nations as of 2026? Reply with just the number.",
        "difficulty": "easy",
        "category": "knowledge",
        "expected_substring": "193",
        "rubric": (
            "The answer must state 193 (the correct number of UN member states). "
            "Accept answers that contain or are just the number 193. "
            "Reject any other number."
        ),
    },
    {
        "id": "capital-france",
        "prompt": "What is the capital of France? Answer in one word.",
        "difficulty": "easy",
        "category": "knowledge",
        "expected_substring": "Paris",
        "rubric": (
            "The answer must be Paris. Accept 'Paris' regardless of capitalization. "
            "Do not accept any other city."
        ),
    },
    {
        "id": "math-expression",
        "prompt": "What is 127 + 68 - 45? Reply with just the number.",
        "difficulty": "easy",
        "category": "math",
        "expected_substring": "150",
        "rubric": (
            "The answer must be 150. Accept the number 150 as a standalone answer "
            "or embedded in a short sentence."
        ),
    },
    {
        "id": "sandbox-read-day",
        "prompt": "Read the file 'subdir/data.json' from the sandbox. What is the value of the 'day' field? Reply with just the number.",
        "difficulty": "easy",
        "category": "file-read",
        "expected_substring": "1",
        "rubric": (
            "The answer must be 1 (the 'day' field in data.json). "
            "Accept just the number or a short phrase containing 1."
        ),
    },
    {
        "id": "sandbox-count-md",
        "prompt": "Use list_dir on the sandbox root directory. How many .md files are there? Reply with just the number.",
        "difficulty": "easy",
        "category": "file-list",
        "expected_substring": "3",
        "rubric": (
            "The answer must be 3. Accept just the number 3 or a short phrase "
            "containing it."
        ),
    },
    {
        "id": "sandbox-grep-example-lines",
        "prompt": "Use run_command with 'grep -c' to count how many lines in the sandbox file 'notes.md' contain the word 'Example'. Reply with just the number.",
        "difficulty": "medium",
        "category": "command-grep",
        "expected_substring": "5",
        "rubric": (
            "The answer must be 5. Accept just the number 5 or a short phrase "
            "containing it."
        ),
    },
    {
        "id": "sandbox-write-numbers",
        "prompt": "Write a file named 'numbers.txt' to the sandbox containing the numbers 1 through 10, one per line. Then use run_command with 'wc -l numbers.txt' to verify the line count. Reply with just the number that wc reports.",
        "difficulty": "medium",
        "category": "multi-tool",
        "expected_substring": "10",
        "rubric": (
            "The answer must be 10 (the line count of numbers.txt after writing "
            "numbers 1 through 10, one per line). Accept just the number 10 or a "
            "short phrase containing it."
        ),
    },
    {
        "id": "sandbox-second-heading",
        "prompt": "Use run_command with grep to extract all lines that start with '## ' from the sandbox file 'notes.md'. What is the SECOND heading in the output? Reply with just the heading text (without the '## ' prefix).",
        "difficulty": "medium",
        "category": "command-grep",
        "expected_substring": "5 Core Workflow Patterns",
        "rubric": (
            "The answer must contain 'The 5 Core Workflow Patterns' or at minimum "
            "'5 Core Workflow Patterns'. Minor variations in capitalization are "
            "acceptable. The key distinction is '5 Core Workflow Patterns' — "
            "not '3 Core Workflow Patterns' or any other heading."
        ),
    },
    {
        "id": "sandbox-fetch-import",
        "prompt": "Read the file 'fetch_url.py' from the sandbox. What is the exact import statement on line 1? Reply with just that line of code.",
        "difficulty": "medium",
        "category": "file-read",
        "expected_substring": "import urllib.request",
        "rubric": (
            "The answer must contain the import statement 'import urllib.request'. "
            "Extra formatting like markdown code blocks (```) is acceptable as "
            "long as the import statement itself is correct. Any other import "
            "statement (like 'import requests') is wrong."
        ),
    },
    {
        "id": "sandbox-compare-headings",
        "prompt": "Read both 'notes_day4_before.md' and 'notes.md' from the sandbox — two versions of an article about building AI agents. Count how many lines start with '## ' (top-level markdown headings) in EACH file. Reply with the two numbers separated by a comma, with the older version ('notes_day4_before.md') first. Example format: '4,3'.",
        "difficulty": "hard",
        "category": "multi-step-compare",
        "expected_substring": "4, 3",  # note: space after comma — substring is strict
        "rubric": (
            "The answer must contain '4,3' or '4, 3' — meaning 4 headings in "
            "notes_day4_before.md and 3 in notes.md. The numbers must be in the "
            "correct order (before first, after second). Accept with or without "
            "space after the comma."
        ),
    },
    {
        "id": "sandbox-py-line-total",
        "prompt": "Count the total number of lines across ALL .py files in the sandbox root directory. Use run_command with 'wc -l *.py' for efficiency. Then write the total line count (just the number) to a new file called 'py_total.txt'. Finally, read 'py_total.txt' back and reply with its exact content.",
        "difficulty": "hard",
        "category": "multi-step-pipeline",
        "expected_substring": "11",
        "rubric": (
            "The answer must be 11 or contain 11 as the total line count across "
            "all .py files in the sandbox. The key fact is that the total is 11. "
            "Reject any other number."
        ),
    },
    {
        "id": "sandbox-most-common-import",
        "prompt": "A module is 'commonly imported' if it appears in multiple .py files. Use list_dir to find all .py files in the sandbox root, then use run_command with grep on those files to find all 'import' statements. Which module is imported by the MOST .py files in the sandbox? Ignore files that don't import anything. Reply with just the module name (the part after 'import ').",
        "difficulty": "hard",
        "category": "multi-step-analysis",
        "expected_substring": "os",
        "rubric": (
            "The answer must be 'os' — the most commonly imported module across "
            ".py files in the sandbox. Accept 'os' regardless of capitalization "
            "or surrounding whitespace. 'glob' is also tied at 3 files so either "
            "'os' or 'glob' is acceptable, but the answer should be a single "
            "module name."
        ),
    },
]


# ============================================================================
# Agent runner (non-interactive)
# ============================================================================

def run_agent_noninteractive(user_input: str, max_turns: int = 12) -> str:
    """Run the week03 agent silently."""
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
            model=MODEL, tools=TOOLS, messages=messages
        )
        choice = response.choices[0]
        msg = choice.message

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
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
                output = _make_error(ErrorType.INVALID_INPUT, f"Invalid JSON arguments: {e}")
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

def substring_check(expected_substring: str, answer: str) -> bool:
    """Check if expected_substring appears in answer."""
    return expected_substring in answer


def score_one_answer(task: dict, answer: str) -> dict:
    """Score a single answer with both substring and llm_judge checkers."""
    # Substring check
    sub_pass = substring_check(task["expected_substring"], answer)

    # LLM judge check
    judge_pass, judge_detail = _llm_judge(task["prompt"], task["rubric"], answer)

    return {
        "task_id": task["id"],
        "answer": answer,
        "substring_pass": sub_pass,
        "judge_pass": judge_pass,
        "judge_detail": judge_detail,
        "agree": sub_pass == judge_pass,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per task (default 5).",
    )
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    print("=" * 70)
    print("Week 4 Day 6: Substring Checker vs. LLM Judge")
    print("=" * 70)
    print(f"Model:      {MODEL}")
    print(f"Tasks:      {len(COMPARE_TASKS)}")
    print(f"Runs/task:  {args.runs}")
    print(f"Sandbox:    {SANDBOX_ROOT.resolve()}")
    print()

    all_scores: list[dict] = []
    per_task_agreements: dict[str, list[bool]] = {}

    for task in COMPARE_TASKS:
        task_agreements: list[bool] = []
        print(f"── {task['id']} ({task['difficulty']}, {task['category']}) ──")

        for run_i in range(args.runs):
            # 1. Run the agent
            try:
                answer = run_agent_noninteractive(task["prompt"])
            except Exception as e:
                answer = f"[exception: {e}]"

            # 2. Score with both checkers
            score = score_one_answer(task, answer)
            all_scores.append(score)
            task_agreements.append(score["agree"])

            # Print per-run result
            sub_icon = "✅" if score["substring_pass"] else "❌"
            judge_icon = "✅" if score["judge_pass"] else "❌"
            agree_icon = "✅" if score["agree"] else "⚠️  DISAGREE"
            preview = answer[:100].replace("\n", " ")
            print(f"  run {run_i + 1}/{args.runs}  substring={sub_icon}  "
                  f"judge={judge_icon}  {agree_icon}  → {preview}")

        per_task_agreements[task["id"]] = task_agreements
        n_agree = sum(task_agreements)
        rate = n_agree / len(task_agreements)
        print(f"  Agreement: {n_agree}/{len(task_agreements)} ({rate:.0%})")
        print()

    # ---- Summary ----
    total_runs = len(all_scores)
    n_agree = sum(s["agree"] for s in all_scores)
    n_disagree = total_runs - n_agree
    agree_rate = n_agree / total_runs if total_runs else 0.0

    # Breakdown: when they agree, what do they say?
    agree_both_pass = sum(1 for s in all_scores if s["agree"] and s["substring_pass"])
    agree_both_fail = sum(1 for s in all_scores if s["agree"] and not s["substring_pass"])

    # When they disagree: substring-pass-judge-fail vs substring-fail-judge-pass
    disagree_sub_pass = sum(
        1 for s in all_scores if not s["agree"] and s["substring_pass"]
    )
    disagree_judge_pass = sum(
        1 for s in all_scores if not s["agree"] and s["judge_pass"]
    )

    print("=" * 70)
    print("RESULTS: Substring Checker vs. LLM Judge")
    print("=" * 70)
    print(f"  Total runs:           {total_runs}")
    print(f"  Agreed:               {n_agree} ({agree_rate:.0%})")
    print(f"    Both PASS:          {agree_both_pass}")
    print(f"    Both FAIL:          {agree_both_fail}")
    print(f"  Disagreed:            {n_disagree} ({n_disagree / total_runs:.0%})")
    print(f"    Sub PASS, Judge FAIL:  {disagree_sub_pass}")
    print(f"    Sub FAIL, Judge PASS:  {disagree_judge_pass}")
    print()

    # Per-task summary
    print(f"{'task':<34} {'agree':>7} {'disagree':>9} {'rate':>6}")
    print("-" * 62)
    for task in COMPARE_TASKS:
        tid = task["id"]
        agrees = per_task_agreements[tid]
        n_a = sum(agrees)
        n_d = len(agrees) - n_a
        print(f"{tid:<34} {n_a:>3}/{len(agrees):<4} {n_d:>5}      {n_a/len(agrees):>5.0%}")
    print()

    # Per-task confusion breakdown
    print("── Per-task confusion matrix ──")
    for task in COMPARE_TASKS:
        tid = task["id"]
        task_scores = [s for s in all_scores if s["task_id"] == tid]
        both_pass = sum(1 for s in task_scores if s["agree"] and s["substring_pass"])
        both_fail = sum(1 for s in task_scores if s["agree"] and not s["substring_pass"])
        sub_only = sum(1 for s in task_scores if not s["agree"] and s["substring_pass"])
        judge_only = sum(1 for s in task_scores if not s["agree"] and s["judge_pass"])
        d_mark = " ⚠️" if (sub_only + judge_only) > 0 else ""
        print(f"  {tid}:{d_mark}")
        print(f"    agree-PASS={both_pass}  agree-FAIL={both_fail}  "
              f"sub_only={sub_only}  judge_only={judge_only}")

    # Disagreement details for manual adjudication
    disagreements = [s for s in all_scores if not s["agree"]]
    if disagreements:
        print(f"\n{'=' * 70}")
        print(f"DISAGREEMENTS ({len(disagreements)} total — needs manual adjudication)")
        print(f"{'=' * 70}")
        for i, d in enumerate(disagreements, 1):
            direction = (
                "sub=PASS / judge=FAIL (substring too lenient, or judge too strict?)"
                if d["substring_pass"]
                else "sub=FAIL / judge=PASS (substring too strict, or judge too lenient?)"
            )
            print(f"\n── Disagreement #{i}: {d['task_id']} ──")
            print(f"  Direction: {direction}")
            print(f"  Answer:    {d['answer'][:300].replace(chr(10), '¶')}")
            print(f"  Substring: {'PASS' if d['substring_pass'] else 'FAIL'}")
            print(f"  Judge:     {'PASS' if d['judge_pass'] else 'FAIL'} ({d['judge_detail']})")
            print(f"  Expected:  '{task_for_id(d['task_id'])}'")

    # Per-task agreement rate for quick reference
    print(f"\n{'=' * 70}")
    print("PER-TASK AGREEMENT RATES (sorted by disagreement)")
    print(f"{'=' * 70}")
    task_agree_rates = sorted(
        [(tid, sum(agrees) / len(agrees), sum(agrees), len(agrees))
         for tid, agrees in per_task_agreements.items()],
        key=lambda x: x[1],
    )
    for tid, rate, n_a, total in task_agree_rates:
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {tid:<34} {bar} {rate:.0%} ({n_a}/{total})")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = _THIS_DIR / f"day6_results_{timestamp}.json"

    # Build per-task summary
    per_task_summary = {}
    for task in COMPARE_TASKS:
        tid = task["id"]
        agrees = per_task_agreements[tid]
        per_task_summary[tid] = {
            "difficulty": task["difficulty"],
            "category": task["category"],
            "runs": len(agrees),
            "agreed": sum(agrees),
            "disagreed": len(agrees) - sum(agrees),
            "agreement_rate": sum(agrees) / len(agrees),
        }

    report = {
        "experiment": "week04_day6_checker_comparison",
        "description": (
            "Substring checker vs. LLM judge disagreement analysis. "
            "Each answer scored by both methods. Disagreements need manual "
            "adjudication — fill in ground_truth and wrong_checker fields."
        ),
        "model": MODEL,
        "date": run_date,
        "runs_per_task": args.runs,
        "summary": {
            "total_runs": total_runs,
            "agreed": n_agree,
            "disagreed": n_disagree,
            "agreement_rate": agree_rate,
            "agree_both_pass": agree_both_pass,
            "agree_both_fail": agree_both_fail,
            "disagree_sub_pass_judge_fail": disagree_sub_pass,
            "disagree_sub_fail_judge_pass": disagree_judge_pass,
        },
        "per_task_summary": per_task_summary,
        "all_scores": all_scores,
        "disagreements": [
            {
                "task_id": d["task_id"],
                "answer": d["answer"],
                "substring_pass": d["substring_pass"],
                "judge_pass": d["judge_pass"],
                "judge_detail": d["judge_detail"],
                "ground_truth": None,  # ← fill in: true (PASS) or false (FAIL)
                "wrong_checker": None,  # ← fill in: "substring" or "llm_judge" or "both" or "neither"
                "notes": "",
            }
            for d in disagreements
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")

    if disagreements:
        print(f"\n⚠️  {len(disagreements)} disagreement(s) found.")
        print("To complete the experiment:")
        print(f"  1. Open {out_path.name}")
        print("  2. For each disagreement, review the answer and fill in:")
        print("     - ground_truth: true (answer is correct) or false (answer is wrong)")
        print("     - wrong_checker: 'substring' or 'llm_judge' or 'both' or 'neither'")
        print("  3. Report the final tally: which checker was wrong more often?")
    else:
        print("\n🎉 Perfect agreement between substring and LLM judge!")


def task_for_id(task_id: str) -> str:
    """Get the expected_substring for a task id (used in debug printing)."""
    for t in COMPARE_TASKS:
        if t["id"] == task_id:
            return t["expected_substring"]
    return "?"


if __name__ == "__main__":
    main()
