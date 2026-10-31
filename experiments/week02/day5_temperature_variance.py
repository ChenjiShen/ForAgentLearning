"""Week 2 Day 5: variance measurement — same prompt, 10x, three temperatures.

The lesson: an LLM is a *sampler*, not a function. The same prompt does not
give you the same answer twice — how much it wanders is set by `temperature`.
This is your first real variance measurement, and it splits into TWO questions
that people usually conflate:

    1. Text variance   — are the final answers identical, character for character?
    2. Plan variance   — did the agent call the SAME tools in the SAME order?

They come apart. At a low temperature the plan is often rock-stable (always
`query_orders -> calculator`) while the prose around the number still wiggles.
At a high temperature the model may pick a different plan entirely (skip a
tool, add an extra call, reorder). We measure both.

For each temperature we run the identical prompt N times (default 10) and report:

    identical_text   — are all N answers byte-for-byte identical?  (yes/no)
    distinct_text    — how many DIFFERENT answers appeared
    modal_text_share — how often the single most common answer showed up
    identical_plan   — did every run make the same tool-call sequence?
    distinct_plans   — how many DIFFERENT tool-call signatures appeared

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week02/day5_temperature_variance.py
    python experiments/week02/day5_temperature_variance.py --runs 10 --temps 0 0.7 1.2
    python experiments/week02/day5_temperature_variance.py --prompt "What is 17 * 23 + 4?"
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime

from agent import MODEL, run_agent

# A prompt that both (a) forces a multi-tool PLAN and (b) leaves room for the
# prose to wander — so we can watch text-variance and plan-variance separately.
# Expected plan: query_orders(Alice) + query_orders(Bob) -> calculator.
DEFAULT_PROMPT = "Who has spent more in total, Alice or Bob? By how much?"

# The three canonical temperatures from the assignment.
DEFAULT_TEMPS = [0.0, 0.7, 1.2]


def normalize(text: str) -> str:
    """Collapse trivial whitespace differences so we compare *content*.

    We keep the raw answer too (see records) — this is only for the "how many
    distinct answers" tally, where a trailing newline shouldn't count as new.
    """
    return " ".join(text.split())


def run_one_temperature(prompt: str, temperature: float, runs: int) -> dict:
    """Run `prompt` `runs` times at one temperature; return per-temp stats."""
    records: list[dict] = []
    print(f"\n{'=' * 70}\nTEMPERATURE {temperature}  ({runs} runs)\n{'=' * 70}")

    for i in range(runs):
        try:
            trace = run_agent(prompt, temperature=temperature, capture=True)
            answer = trace["answer"]
            signature = trace["tool_signature"] or "(no tools)"
            err = None
        except Exception as e:  # keep the sweep going; a dead run is still data
            answer, signature, trace, err = f"[error: {e}]", "(error)", None, str(e)

        records.append(
            {
                "run": i,
                "answer": answer,
                "answer_norm": normalize(answer),
                "tool_signature": signature,
                "tool_calls": trace["tool_calls"] if trace else None,
                "error": err,
            }
        )
        print(f"  run {i:>2}: [{signature}]  {normalize(answer)[:90]}")

    # ---- tallies ----
    text_counts = Counter(r["answer_norm"] for r in records)
    plan_counts = Counter(r["tool_signature"] for r in records)
    modal_text, modal_text_n = text_counts.most_common(1)[0]
    modal_plan, modal_plan_n = plan_counts.most_common(1)[0]

    return {
        "temperature": temperature,
        "runs": runs,
        "identical_text": len(text_counts) == 1,
        "distinct_text": len(text_counts),
        "modal_text_share": f"{modal_text_n}/{runs}",
        "identical_plan": len(plan_counts) == 1,
        "distinct_plans": len(plan_counts),
        "modal_plan": modal_plan,
        "modal_plan_share": f"{modal_plan_n}/{runs}",
        "text_counts": dict(text_counts),
        "plan_counts": dict(plan_counts),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="runs per temperature (default 10)")
    parser.add_argument("--temps", type=float, nargs="+", default=DEFAULT_TEMPS,
                        help="temperatures to sweep (default 0 0.7 1.2)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="the prompt to repeat")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    print(f"model:  {MODEL}")
    print(f"prompt: {args.prompt!r}")

    summaries = [run_one_temperature(args.prompt, t, args.runs) for t in args.temps]

    # ---- comparison table ----
    print(f"\n{'=' * 70}\nVARIANCE BY TEMPERATURE\n{'=' * 70}")
    header = (f"{'temp':>6}{'identical_text':>16}{'distinct_text':>15}"
              f"{'modal_text':>12}{'identical_plan':>16}{'distinct_plans':>16}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['temperature']:>6}"
            f"{('yes' if s['identical_text'] else 'NO'):>16}"
            f"{s['distinct_text']:>15}"
            f"{s['modal_text_share']:>12}"
            f"{('yes' if s['identical_plan'] else 'NO'):>16}"
            f"{s['distinct_plans']:>16}"
        )

    print("\nplans seen per temperature (tool-call signature -> count):")
    for s in summaries:
        print(f"  temp {s['temperature']}:")
        for plan, n in sorted(s["plan_counts"].items(), key=lambda kv: -kv[1]):
            print(f"      {n:>2}x  {plan}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"day5_results_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL, "prompt": args.prompt, "runs": args.runs,
             "temps": args.temps, "summaries": summaries},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
