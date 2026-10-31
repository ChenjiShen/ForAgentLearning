"""Week 4 Day 3: LLM-as-Judge validation experiment.

Two-phase experiment:
  Phase 1 — Collect: run the agent on 10 tasks, save answers → hand-label file.
  Phase 2 — Validate: read hand labels, run llm_judge on the same answers,
            compute agreement (accuracy, Cohen's kappa, precision, recall).

Usage:
  # Phase 1: collect answers (then hand-label the output file)
  python experiments/week04/evals/day3_llm_judge.py --collect

  # Phase 2: run judge + compute agreement against hand labels
  python experiments/week04/evals/day3_llm_judge.py --validate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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

ANSWERS_FILE = _THIS_DIR / "day3_answers.json"
LABELS_FILE = _THIS_DIR / "day3_labels.json"


# ============================================================================
# 10 diverse tasks — each has a rubric for the llm_judge
# ============================================================================

JUDGE_TASKS = [
    {
        "id": "knowledge-un-members",
        "prompt": "How many member states are there in the United Nations as of 2026? Reply with just the number.",
        "difficulty": "easy",
        "category": "knowledge",
        "rubric": "The answer must state 193 (the correct number of UN member states). Accept answers that contain or are just the number 193.",
    },
    {
        "id": "capital-france",
        "prompt": "What is the capital of France? Answer in one word.",
        "difficulty": "easy",
        "category": "knowledge",
        "rubric": "The answer must be Paris. Accept 'Paris' regardless of capitalization. Do not accept any other city.",
    },
    {
        "id": "math-expression",
        "prompt": "What is 127 + 68 - 45? Reply with just the number.",
        "difficulty": "easy",
        "category": "math",
        "rubric": "The answer must be 150. Accept the number 150 as a standalone answer or embedded in a short sentence.",
    },
    {
        "id": "sandbox-read-day",
        "prompt": "Read the file 'subdir/data.json' from the sandbox. What is the value of the 'day' field? Reply with just the number.",
        "difficulty": "easy",
        "category": "file-read",
        "rubric": "The answer must be 1 (the 'day' field in data.json). Accept just the number or a short phrase containing 1.",
    },
    {
        "id": "sandbox-count-md",
        "prompt": "Use list_dir on the sandbox root directory. How many .md files are there? Reply with just the number.",
        "difficulty": "easy",
        "category": "file-list",
        "rubric": "The answer must be 3. Accept just the number 3 or a short phrase containing it.",
    },
    {
        "id": "sandbox-grep-example",
        "prompt": "Use run_command with 'grep -c' to count how many lines in the sandbox file 'notes.md' contain the word 'Example'. Reply with just the number.",
        "difficulty": "medium",
        "category": "command-grep",
        "rubric": "The answer must be 5. Accept just the number 5 or a short phrase containing it.",
    },
    {
        "id": "sandbox-second-heading",
        "prompt": "Use run_command with grep to extract all lines that start with '## ' from the sandbox file 'notes.md'. What is the SECOND heading in the output? Reply with just the heading text (without the '## ' prefix).",
        "difficulty": "medium",
        "category": "command-grep",
        "rubric": "The answer must be 'The 5 Core Workflow Patterns' (or a close match). The answer is correct if it contains the phrase '5 Core Workflow Patterns' — minor variations in capitalization are acceptable.",
    },
    {
        "id": "sandbox-fetch-import",
        "prompt": "Read the file 'fetch_url.py' from the sandbox. What is the exact import statement on line 1? Reply with just that line of code.",
        "difficulty": "medium",
        "category": "file-read",
        "rubric": "The answer must be 'import urllib.request' exactly. Whitespace-only differences are acceptable. Any other import statement (like 'import requests') is wrong.",
    },
    {
        "id": "sandbox-py-total",
        "prompt": "Count the total number of lines across ALL .py files in the sandbox root directory. Use run_command with 'wc -l *.py'. Then write the total to 'py_total.txt', read it back, and reply with its exact content.",
        "difficulty": "hard",
        "category": "multi-step-pipeline",
        "rubric": "The answer must be 11 or must contain 11 (the total line count across all .py files in the sandbox). The key fact is that the total is 11.",
    },
    {
        "id": "sandbox-compare-headings",
        "prompt": "Read both 'notes_day4_before.md' and 'notes.md' from the sandbox. Count how many lines start with '## ' in EACH file. Reply with the two numbers separated by a comma: older version first, e.g. '4,3'.",
        "difficulty": "hard",
        "category": "multi-step-compare",
        "rubric": "The answer must contain '4,3' or '4, 3' — meaning 4 headings in notes_day4_before.md and 3 in notes.md. The numbers must be in the correct order (before first, after second). Accept with or without space after the comma.",
    },
]


# ============================================================================
# Agent runner (same non-interactive pattern as run_evals.py)
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
# Phase 1 — Collect
# ============================================================================

def phase_collect() -> None:
    """Run the agent on all 10 tasks and save answers for hand-labeling."""
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    print("=" * 60)
    print("PHASE 1 — Collecting agent answers")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Tasks: {len(JUDGE_TASKS)}")
    print()

    answers_data = []
    for i, task in enumerate(JUDGE_TASKS, 1):
        print(f"[{i:2d}/{len(JUDGE_TASKS)}] Running: {task['id']} ({task['difficulty']})")
        try:
            answer = run_agent_noninteractive(task["prompt"])
        except Exception as e:
            answer = f"[exception: {e}]"

        preview = answer[:150].replace("\n", " ")
        print(f"         → {preview}")

        answers_data.append({
            "task_id": task["id"],
            "prompt": task["prompt"],
            "difficulty": task["difficulty"],
            "category": task["category"],
            "rubric": task["rubric"],
            "answer": answer,
        })

    # Save answers
    output = {
        "description": "Agent answers for Week 4 Day 3 LLM-judge validation",
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tasks": len(answers_data),
        "tasks": answers_data,
    }
    ANSWERS_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n✅ Answers saved → {ANSWERS_FILE}")
    print(f"\nNext: edit {LABELS_FILE} (just created) to add your hand labels.")
    print("  Then run: python experiments/week04/evals/day3_llm_judge.py --validate")

    # Also create a template labels file
    labels_template = {
        "description": "Hand labels for Week 4 Day 3 — fill in 'hand_label' as true/false for each task",
        "date_labeled": "",
        "labels": [
            {
                "task_id": t["task_id"],
                "prompt_preview": t["prompt"][:100],
                "answer": t["answer"],
                "rubric": t["rubric"],
                "hand_label": None,  # ← set to true (PASS) or false (FAIL)
                "notes": "",
            }
            for t in answers_data
        ],
    }
    LABELS_FILE.write_text(json.dumps(labels_template, indent=2, ensure_ascii=False))
    print(f"✅ Label template → {LABELS_FILE}")


# ============================================================================
# Phase 2 — Validate
# ============================================================================

def cohens_kappa(judge: list[bool], human: list[bool]) -> float:
    """Compute Cohen's kappa for two binary raters."""
    n = len(judge)
    if n != len(human) or n == 0:
        return 0.0

    a = sum(1 for j, h in zip(judge, human) if j and h)       # both pass
    b = sum(1 for j, h in zip(judge, human) if j and not h)   # judge pass, human fail
    c = sum(1 for j, h in zip(judge, human) if not j and h)   # judge fail, human pass
    d = sum(1 for j, h in zip(judge, human) if not j and not h)  # both fail

    po = (a + d) / n  # observed agreement
    p_judge_pass = (a + b) / n
    p_human_pass = (a + c) / n
    pe = p_judge_pass * p_human_pass + (1 - p_judge_pass) * (1 - p_human_pass)  # expected by chance

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def phase_validate() -> None:
    """Read hand labels, run llm_judge, and compute agreement."""
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    if not LABELS_FILE.exists():
        raise SystemExit(
            f"Labels file not found: {LABELS_FILE}\n"
            "Run --collect first to generate it, then hand-label the answers."
        )

    labels_data = json.loads(LABELS_FILE.read_text())
    label_entries = labels_data["labels"]

    # Check all labels are filled
    unlabeled = [e["task_id"] for e in label_entries if e["hand_label"] is None]
    if unlabeled:
        raise SystemExit(
            f"These tasks still need hand labels: {unlabeled}\n"
            f"Edit {LABELS_FILE} and set 'hand_label' to true or false for each."
        )

    print("=" * 60)
    print("PHASE 2 — Validate llm_judge against hand labels")
    print("=" * 60)
    print(f"Model (agent):       {MODEL}")
    print(f"Model (judge):       {MODEL}  ⚠️  same model → self-preference bias possible")
    print(f"Tasks:                {len(label_entries)}")
    print()

    human_labels: list[bool] = []
    judge_labels: list[bool] = []
    agreements: list[dict] = []

    for entry in label_entries:
        tid = entry["task_id"]
        answer = entry["answer"]
        rubric = entry.get("rubric", entry.get("rubric", ""))
        prompt = entry.get("prompt_preview", entry.get("prompt", ""))
        human = bool(entry["hand_label"])

        # Run the judge
        judge_pass, judge_verdict = _llm_judge(prompt, rubric, answer)
        human_labels.append(human)
        judge_labels.append(judge_pass)

        agree = "✅" if human == judge_pass else "❌ DISAGREE"
        print(f"  {agree} {tid}")
        print(f"       Human: {'PASS' if human else 'FAIL'}  |  Judge: {'PASS' if judge_pass else 'FAIL'}  ({judge_verdict})")
        print(f"       Answer: {answer[:120].replace(chr(10), ' ')}")

        agreements.append({
            "task_id": tid,
            "human_label": human,
            "judge_label": judge_pass,
            "agree": human == judge_pass,
            "judge_verdict": judge_verdict,
            "answer_preview": answer[:200],
        })

    # ---- Metrics ----
    n = len(human_labels)
    n_agree = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    accuracy = n_agree / n if n else 0.0

    tp = sum(1 for j, h in zip(judge_labels, human_labels) if j and h)
    fp = sum(1 for j, h in zip(judge_labels, human_labels) if j and not h)
    fn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and h)
    tn = sum(1 for j, h in zip(judge_labels, human_labels) if not j and not h)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    kappa = cohens_kappa(judge_labels, human_labels)

    # ---- Report ----
    print()
    print("=" * 60)
    print("AGREEMENT REPORT")
    print("=" * 60)
    print(f"  Accuracy:        {accuracy:.0%}  ({n_agree}/{n} agreed)")
    print(f"  Cohen's κ:       {kappa:+.3f}")
    print(f"  Precision:       {precision:.0%}  (of judge PASS, how many human PASS)")
    print(f"  Recall:          {recall:.0%}  (of human PASS, how many judge caught)")
    print(f"  F1:              {f1:.3f}")
    print()
    print("  Confusion Matrix (Judge vs Human):")
    print(f"                    Human PASS   Human FAIL")
    print(f"    Judge PASS         {tp:5d}        {fp:5d}")
    print(f"    Judge FAIL         {fn:5d}        {tn:5d}")
    print()

    # Individual disagreements
    disagreements = [a for a in agreements if not a["agree"]]
    if disagreements:
        print(f"Disagreements ({len(disagreements)}):")
        for d in disagreements:
            direction = "Judge PASS / Human FAIL (false positive)" if d["judge_label"] else "Judge FAIL / Human PASS (false negative)"
            print(f"  - {d['task_id']}: {direction}")
            print(f"    Answer: {d['answer_preview'][:150]}")
            print(f"    Judge said: {d['judge_verdict']}")
    else:
        print("No disagreements — perfect agreement! 🎉")

    # Kappa interpretation
    print()
    if kappa < 0.0:
        print("⚠️  Kappa < 0: worse than chance — judge may be anti-correlated")
    elif kappa < 0.2:
        print("⚠️  Kappa < 0.2: slight agreement — judge adds little value")
    elif kappa < 0.4:
        print("   Kappa 0.2–0.4: fair agreement")
    elif kappa < 0.6:
        print("   Kappa 0.4–0.6: moderate agreement — usable with caution")
    elif kappa < 0.8:
        print("✅ Kappa 0.6–0.8: substantial agreement — judge is reliable")
    else:
        print("🎉 Kappa ≥ 0.8: almost perfect agreement")

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _THIS_DIR / f"day3_results_{timestamp}.json"
    report = {
        "experiment": "week04_day3_llm_judge_validation",
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tasks": n,
        "metrics": {
            "accuracy": accuracy,
            "cohens_kappa": kappa,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "agreements": agreements,
        "disagreements": [d for d in agreements if not d["agree"]],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect", action="store_true", help="Phase 1: collect agent answers")
    group.add_argument("--validate", action="store_true", help="Phase 2: run judge + compute agreement")
    args = parser.parse_args()

    if args.collect:
        phase_collect()
    else:
        phase_validate()


if __name__ == "__main__":
    main()
