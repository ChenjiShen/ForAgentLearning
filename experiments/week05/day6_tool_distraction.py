"""Week 5 Day 6: Does adding realistic-but-irrelevant tools degrade tool selection?

A proper measurement of the most surprising effect from Days 1-5.

The surprise (Day 3): wrong-tool rate did NOT rise with tool count. At 5 real
tools the wrong-tool rate was 5.4%; at 15 tools (5 real + 10 dummy) it fell to
3.6%, and at 30 tools (5 real + 25 dummy) it was 4.0%. The dummy-call rate was
exactly 0.0 in every condition — 80 runs with realistic distractors
(send_email, query_database, search_documents, ...) and the model never called
one. Week 3's Day 6 saw the same null effect on easy tasks (4 vs 12 tools).
Conventional wisdom says more tools => more confusion. Days 1-5 say otherwise.

Day 3's weaknesses, fixed here:
  1. Only 8 prompts (P5/P7 were dropped because web_fetch wasn't in the core
     set). -> Full 10-prompt set with all 6 core tools (adds web_fetch),
     matching the baseline used by Days 1/4/5.
  2. Fixed dummy slices (always the same first 10 / first 25 dummies).
     -> Randomized dummy subset per run (seeded => reproducible), with the
     4 hard near-miss distractors always included.
  3. No interval estimates, no significance tests.
     -> Wilson 95% CIs on pooled rates + Fisher exact tests between
     conditions.
  4. Condition order was confounded with time.
     -> Cell order shuffled with a fixed seed.

Conditions (dummies are sampled fresh per run):
  - 6_tools:  6 real tools, no dummies          (baseline, like Days 1/4/5)
  - 15_tools: 6 real + 9 dummy tools
  - 30_tools: 6 real + 24 dummy tools

Default: 3 conditions x 10 prompts x 5 runs = 150 agent runs.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week05/day6_tool_distraction.py
    python experiments/week05/day6_tool_distraction.py --runs 3
    python experiments/week05/day6_tool_distraction.py --prompts P1-read-known-file P9-no-tool-trivia
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Day 1's harness: the 6-tool core (after-descriptions) + full 10-prompt set.
from day1_tool_descriptions import (  # noqa: E402
    EVAL_PROMPTS,
    MODEL,
    TOOL_FUNCTIONS,
    build_tools,
    _estimate_cost,
)
# Day 3's harness: the 25 realistic dummy schemas + a runner that returns a
# plausible success response when the model calls a dummy tool.
from day3_tool_count import (  # noqa: E402
    DUMMY_TOOL_SCHEMAS,
    run_experiment_agent,
)

# ============================================================================
# Design constants
# ============================================================================
NEAR_MISS_DUMMIES = [
    # Descriptions overlap with real capabilities -> the hardest distractors.
    "search_documents",    # "search/find/reference material" vs web_search
    "query_database",      # "look up / retrieve structured data" vs read_file
    "summarize_text",      # "summarize / get the gist" vs just answering
    "extract_entities",    # "identify names/places/orgs mentioned" vs answering
]

CORE_TOOLS = build_tools("after")  # 6 tools: read_file, write_file, list_dir,
#                                        run_command, web_fetch, web_search
CORE_TOOL_NAMES = frozenset(t["function"]["name"] for t in CORE_TOOLS)
assert len(CORE_TOOLS) == 6, f"expected 6 core tools, got {len(CORE_TOOLS)}"

# (label, n_real, n_dummy) — dummies are sampled fresh per run.
CONDITIONS = [
    ("6_tools", 6, 0),
    ("15_tools", 6, 9),
    ("30_tools", 6, 24),
]

DUMMY_BY_NAME = {t["function"]["name"]: t for t in DUMMY_TOOL_SCHEMAS}
GENERIC_DUMMIES = [t for t in DUMMY_TOOL_SCHEMAS
                   if t["function"]["name"] not in NEAR_MISS_DUMMIES]
assert len(DUMMY_TOOL_SCHEMAS) == 25
assert len(GENERIC_DUMMIES) == 21
assert all(n in DUMMY_BY_NAME for n in NEAR_MISS_DUMMIES)


# ============================================================================
# Deterministic RNG (stable across Python runs — hash() is salted, md5 isn't)
# ============================================================================
def _rng_for(seed_str: str) -> random.Random:
    digest = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    return random.Random(int(digest, 16) & 0x7FFFFFFF)


def build_dummy_subset(n_dummies: int, seed_str: str) -> list[dict]:
    """Sample n_dummies for one run. Near-miss distractors are always
    included; the rest are sampled from the generic pool. Deterministic in
    seed_str (prompt, condition, run)."""
    if n_dummies <= 0:
        return []
    picks = [DUMMY_BY_NAME[n] for n in NEAR_MISS_DUMMIES]
    if n_dummies <= len(picks):
        return picks[:n_dummies]
    generic = list(GENERIC_DUMMIES)
    _rng_for(seed_str).shuffle(generic)
    picks += generic[: n_dummies - len(picks)]
    return picks


# ============================================================================
# Scoring — explicit denominators, comparable with both day1 and day3 metrics
# ============================================================================
def score_run(trace: dict, expected_tools: list[str]) -> dict:
    """Two rate families:

    - unique-name rates (denominator = distinct tools called): comparable
      with Days 1/4/5, which used ``wrong-tool rate = wrong / distinct tools``.
    - call-count rates (denominator = every call): comparable with Day 3,
      which used ``dummy-call rate = dummy calls / all calls``.
    """
    called_list = [tc["name"] for tc in trace.get("tool_calls", [])]
    called_set = set(called_list)
    expected = set(expected_tools)

    n_calls = len(called_list)
    n_unique = len(called_set)
    n_dummy_calls = sum(1 for n in called_list if n not in CORE_TOOL_NAMES)
    n_wrong_unique = len(called_set - expected)

    if len(expected) == 0:
        should_call_rate = 1.0 if n_calls == 0 else 0.0
    else:
        should_call_rate = len(called_set & expected) / len(expected)

    return {
        "called": called_list,
        "expected": sorted(expected),
        "n_calls": n_calls,
        "n_unique_tools": n_unique,
        # call-count based (day3-comparable)
        "dummy_call_rate": n_dummy_calls / n_calls if n_calls else 0.0,
        "n_dummy_calls": n_dummy_calls,
        # unique-name based (day1/4/5-comparable)
        "wrong_tool_rate": n_wrong_unique / n_unique if n_unique else 0.0,
        "n_wrong_unique": n_wrong_unique,
        "should_call_rate": should_call_rate,
        "exact_match": called_set == expected,
        "missed": sorted(expected - called_set),
        "extra": sorted(called_set - expected),
    }


# ============================================================================
# Statistics (pure stdlib — no scipy dependency)
# ============================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]].

    Sums hypergeometric probabilities <= the observed table's probability,
    with fixed margins. Exact for small counts (which is what we expect here).
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)

    def prob(x: int) -> float:
        return (math.comb(row1, x) * math.comb(n - row1, col1 - x)
                / math.comb(n, col1))

    p_obs = prob(a)
    p_val = 0.0
    for x in range(lo, hi + 1):
        px = prob(x)
        if px <= p_obs + 1e-15:
            p_val += px
    return min(1.0, p_val)


def _fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k/n:.1%} [{lo:.1%}, {hi:.1%}]  (n={n})"


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per (prompt, condition) cell (default 5).")
    parser.add_argument("--prompts", type=str, nargs="*",
                        help="Prompt IDs to run (default: all 10).")
    parser.add_argument("--conditions", type=str, nargs="*",
                        help="Condition labels to run (default: all 3).")
    parser.add_argument("--seed", type=str, default="20260816",
                        help="Master seed for cell order + dummy sampling.")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    prompts = EVAL_PROMPTS
    if args.prompts:
        prompts = [p for p in EVAL_PROMPTS if p["id"] in args.prompts]
        if not prompts:
            raise SystemExit(f"No prompts matched: {args.prompts}")

    conditions = CONDITIONS
    if args.conditions:
        conditions = [c for c in CONDITIONS if c[0] in args.conditions]
        if not conditions:
            raise SystemExit(f"No conditions matched: {args.conditions}")

    tool_functions = {name: TOOL_FUNCTIONS[name] for name in CORE_TOOL_NAMES}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_cells = len(prompts) * len(conditions)
    print("=" * 76)
    print("WEEK 5 DAY 6 — TOOL DISTRACTION, MEASURED PROPERLY")
    print("  Question: does adding realistic-but-irrelevant tools degrade tool selection?")
    print("=" * 76)
    print(f"  Model: {MODEL}")
    print(f"  Prompts: {len(prompts)}  |  Runs per cell: {args.runs}")
    print(f"  Conditions: {[c[0] for c in conditions]}")
    print(f"  Total runs: {n_cells * args.runs}")
    print(f"  Seed: {args.seed}  (cell order shuffled; dummy subsets sampled per run)")
    print(f"  Near-miss dummies always included: {NEAR_MISS_DUMMIES}")
    print()

    # Cells in deterministic random order so time drift hits all conditions
    # evenly instead of confounding the comparison.
    cells = [(p["id"], c[0]) for c in conditions for p in prompts]
    _rng_for(f"{args.seed}|cells").shuffle(cells)

    all_results: list[dict] = []
    per_cond: dict[str, dict] = {
        label: {
            "n_dummy_calls": 0, "n_calls": 0,
            "n_wrong_unique": 0, "n_unique_tools": 0,
            "exact": 0, "n_runs": 0,
            "dummy_call_rates": [], "wrong_tool_rates": [],
            "should_call_rates": [], "tokens": [], "latencies": [],
            "costs": [], "per_prompt": {},
        }
        for label, _, _ in conditions
    }
    cond_by_label = {c[0]: c for c in conditions}
    prompt_by_id = {p["id"]: p for p in prompts}

    for cell_idx, (prompt_id, cond_label) in enumerate(cells, 1):
        _, n_real, n_dummy = cond_by_label[cond_label]
        prompt = prompt_by_id[prompt_id]
        per_cond[cond_label]["per_prompt"].setdefault(prompt_id, {
            "exact": 0, "runs": 0, "dummy_calls": 0, "wrong_unique": 0,
            "n_calls": 0, "n_unique": 0,
        })

        for run_idx in range(1, args.runs + 1):
            run_number = (cell_idx - 1) * args.runs + run_idx
            dummy_seed = f"{args.seed}|{prompt_id}|{cond_label}|{run_idx}"
            dummies = build_dummy_subset(n_dummy, dummy_seed)
            tools = CORE_TOOLS + dummies

            try:
                t0 = time.perf_counter()
                trace = run_experiment_agent(prompt["prompt"], tools, tool_functions)
                latency = time.perf_counter() - t0
            except Exception as e:
                print(f"  [{cond_label}] {prompt_id} run {run_idx}: API ERROR — {e}")
                all_results.append({
                    "condition": cond_label, "prompt_id": prompt_id,
                    "run": run_idx, "api_error": str(e),
                    "dummy_tools_offered": [t["function"]["name"] for t in dummies],
                })
                continue

            score = score_run(trace, prompt["expected_tools"])
            cost = trace.get("estimated_cost", 0.0)

            per_cond[cond_label]["n_runs"] += 1
            per_cond[cond_label]["n_dummy_calls"] += score["n_dummy_calls"]
            per_cond[cond_label]["n_calls"] += score["n_calls"]
            per_cond[cond_label]["n_wrong_unique"] += score["n_wrong_unique"]
            per_cond[cond_label]["n_unique_tools"] += score["n_unique_tools"]
            per_cond[cond_label]["exact"] += 1 if score["exact_match"] else 0
            per_cond[cond_label]["dummy_call_rates"].append(score["dummy_call_rate"])
            per_cond[cond_label]["wrong_tool_rates"].append(score["wrong_tool_rate"])
            per_cond[cond_label]["should_call_rates"].append(score["should_call_rate"])
            per_cond[cond_label]["tokens"].append(trace["usage"]["total_tokens"])
            per_cond[cond_label]["latencies"].append(latency)
            per_cond[cond_label]["costs"].append(cost)

            pp = per_cond[cond_label]["per_prompt"][prompt_id]
            pp["runs"] += 1
            pp["exact"] += 1 if score["exact_match"] else 0
            pp["dummy_calls"] += score["n_dummy_calls"]
            pp["wrong_unique"] += score["n_wrong_unique"]
            pp["n_calls"] += score["n_calls"]
            pp["n_unique"] += score["n_unique_tools"]

            all_results.append({
                "condition": cond_label,
                "prompt_id": prompt_id,
                "run": run_idx,
                "dummy_tools_offered": [t["function"]["name"] for t in dummies],
                **score,
                "answer": trace["answer"][:300],
                "tool_call_details": trace["tool_calls"],
                "usage": trace["usage"],
                "estimated_cost": cost,
                "latency_s": round(latency, 3),
                "error": trace.get("error"),
            })

            icon = "✅" if score["exact_match"] else "❌"
            call_str = ", ".join(score["called"]) if score["called"] else "(none)"
            exp_str = ", ".join(score["expected"]) if score["expected"] else "(none)"
            extra = ""
            if score["n_dummy_calls"]:
                extra += f"  ⚠️ DUMMY: {[c for c in score['called'] if c not in CORE_TOOL_NAMES]}"
            if score["missed"]:
                extra += f"  missed: {score['missed']}"
            print(
                f"  [{run_number}/{n_cells * args.runs}] [{icon}] {cond_label:<9} "
                f"{prompt_id:<32} run {run_idx}/{args.runs}"
                f"  called=[{call_str}]  expected=[{exp_str}]{extra}"
            )

    # =================================================================
    # Summary
    # =================================================================
    print()
    print("=" * 76)
    print("SUMMARY — DUMMY-CALL RATE (the key metric; call-count based, like Day 3)")
    print("=" * 76)
    for label, _, _ in conditions:
        pc = per_cond[label]
        k, n = pc["n_dummy_calls"], pc["n_calls"]
        rates = pc["dummy_call_rates"]
        mean = sum(rates) / len(rates) if rates else 0.0
        sd = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5 if rates else 0.0
        print(f"  {label:<9} pooled {k}/{n} calls -> {_fmt_ci(k, n)}"
              f"   per-run mean {mean:.3f} ± {sd:.3f}")

    print()
    print("SUMMARY — WRONG-TOOL RATE (unique-name based, like Days 1/4/5)")
    for label, _, _ in conditions:
        pc = per_cond[label]
        k, n = pc["n_wrong_unique"], pc["n_unique_tools"]
        print(f"  {label:<9} pooled {k}/{n} unique tools -> {_fmt_ci(k, n)}")

    print()
    print("SUMMARY — EXACT-MATCH RATE")
    for label, _, _ in conditions:
        pc = per_cond[label]
        print(f"  {label:<9} {pc['exact']}/{pc['n_runs']} -> {_fmt_ci(pc['exact'], pc['n_runs'])}")

    print()
    print("SUMMARY — TOKENS / LATENCY / COST PER RUN (mean ± sd)")
    for label, _, _ in conditions:
        pc = per_cond[label]
        toks = pc["tokens"]; lats = pc["latencies"]; costs = pc["costs"]
        if not toks:
            continue
        m_t = sum(toks) / len(toks); s_t = (sum((x - m_t) ** 2 for x in toks) / len(toks)) ** 0.5
        m_l = sum(lats) / len(lats); s_l = (sum((x - m_l) ** 2 for x in lats) / len(lats)) ** 0.5
        m_c = sum(costs) / len(costs); s_c = (sum((x - m_c) ** 2 for x in costs) / len(costs)) ** 0.5
        print(f"  {label:<9} {m_t:>8,.0f} ± {s_t:>6,.0f} tok   "
              f"{m_l:>5.1f} ± {s_l:>4.1f} s   "
              f"${m_c:.6f} ± ${s_c:.6f}")

    # ------------------------------------------------------------------
    # Fisher exact tests between adjacent conditions
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print("FISHER EXACT TESTS (two-sided, pooled call counts)")
    print("  H0: no association between tool count and error rate")
    print("=" * 76)
    fisher_results = {}
    labels = [c[0] for c in conditions]
    for i in range(1, len(labels)):
        lo_l, hi_l = labels[i - 1], labels[i]
        a = per_cond[lo_l]["n_dummy_calls"]
        b = per_cond[lo_l]["n_calls"] - a
        c = per_cond[hi_l]["n_dummy_calls"]
        d = per_cond[hi_l]["n_calls"] - c
        p_dummy = fisher_exact_two_sided(a, b, c, d)

        a2 = per_cond[lo_l]["n_wrong_unique"]
        b2 = per_cond[lo_l]["n_unique_tools"] - a2
        c2 = per_cond[hi_l]["n_wrong_unique"]
        d2 = per_cond[hi_l]["n_unique_tools"] - c2
        p_wrong = fisher_exact_two_sided(a2, b2, c2, d2)

        fisher_results[f"{lo_l}_vs_{hi_l}"] = {
            "dummy_calls": {"table": [[a, b], [c, d]], "p_two_sided": p_dummy},
            "wrong_tools": {"table": [[a2, b2], [c2, d2]], "p_two_sided": p_wrong},
        }
        sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {lo_l} vs {hi_l}:  dummy-calls {[[a, b], [c, d]]} p={p_dummy:.4f} {sig(p_dummy)}"
              f"   wrong-tools {[[a2, b2], [c2, d2]]} p={p_wrong:.4f} {sig(p_wrong)}")

    # ------------------------------------------------------------------
    # Per-prompt table
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print(f"PER-PROMPT BREAKDOWN (exact matches out of {args.runs} runs; dummy calls in parens)")
    print("=" * 76)
    cond_labels = [c[0] for c in conditions]
    header = f"{'prompt':<34}" + "".join(f"{cl:>18}" for cl in cond_labels)
    print(header)
    print("-" * len(header))
    per_prompt_table = {}
    for prompt in prompts:
        pid = prompt["id"]
        row = f"{pid:<34}"
        per_prompt_table[pid] = {}
        for cl in cond_labels:
            pp = per_cond[cl]["per_prompt"].get(pid, {})
            runs = pp.get("runs", 0)
            ex = pp.get("exact", 0)
            dc = pp.get("dummy_calls", 0)
            cell_str = f"{ex}/{runs}" if runs else "-"
            if dc:
                cell_str += f" (⚠{dc})"
            row += f"{cell_str:>18}"
            per_prompt_table[pid][cl] = {
                "exact": ex, "runs": runs, "dummy_calls": dc,
                "wrong_unique": pp.get("wrong_unique", 0),
                "n_calls": pp.get("n_calls", 0), "n_unique": pp.get("n_unique", 0),
            }
        print(row)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "experiment": "week05_day6_tool_distraction",
        "question": ("Does adding realistic-but-irrelevant tools degrade "
                     "tool selection? (proper measurement of the Day-3 surprise)"),
        "model": MODEL,
        "timestamp": timestamp,
        "seed": args.seed,
        "runs_per_cell": args.runs,
        "conditions": [
            {"label": l, "n_real": nr, "n_dummy": nd, "n_total": nr + nd}
            for l, nr, nd in conditions
        ],
        "near_miss_dummies_always_included": NEAR_MISS_DUMMIES,
        "per_condition": {},
        "fisher_exact_tests": fisher_results,
        "per_prompt": per_prompt_table,
    }
    for label, _, _ in conditions:
        pc = per_cond[label]
        rates = pc["dummy_call_rates"]
        wrongs = pc["wrong_tool_rates"]
        mean_rates = sum(rates) / len(rates) if rates else 0.0
        mean_wrongs = sum(wrongs) / len(wrongs) if wrongs else 0.0
        summary["per_condition"][label] = {
            "n_runs": pc["n_runs"],
            "dummy_calls": {
                "k": pc["n_dummy_calls"], "n_calls": pc["n_calls"],
                "pooled_rate": pc["n_dummy_calls"] / pc["n_calls"] if pc["n_calls"] else 0.0,
                "wilson_ci_95": wilson_ci(pc["n_dummy_calls"], pc["n_calls"]),
                "per_run_mean": mean_rates,
                "per_run_std": (sum((r - mean_rates) ** 2 for r in rates) / len(rates)) ** 0.5 if rates else 0.0,
            },
            "wrong_tools": {
                "k": pc["n_wrong_unique"], "n_unique_tools": pc["n_unique_tools"],
                "pooled_rate": pc["n_wrong_unique"] / pc["n_unique_tools"] if pc["n_unique_tools"] else 0.0,
                "wilson_ci_95": wilson_ci(pc["n_wrong_unique"], pc["n_unique_tools"]),
                "per_run_mean": mean_wrongs,
                "per_run_std": (sum((r - mean_wrongs) ** 2 for r in wrongs) / len(wrongs)) ** 0.5 if wrongs else 0.0,
            },
            "exact_match": {
                "k": pc["exact"], "n": pc["n_runs"],
                "rate": pc["exact"] / pc["n_runs"] if pc["n_runs"] else 0.0,
                "wilson_ci_95": wilson_ci(pc["exact"], pc["n_runs"]),
            },
            "tokens_mean": sum(pc["tokens"]) / len(pc["tokens"]) if pc["tokens"] else 0.0,
            "tokens_total": sum(pc["tokens"]),
            "latency_s_mean": sum(pc["latencies"]) / len(pc["latencies"]) if pc["latencies"] else 0.0,
            "cost_total": sum(pc["costs"]),
            "cost_mean": sum(pc["costs"]) / len(pc["costs"]) if pc["costs"] else 0.0,
        }

    out_path = _THIS_DIR / f"day6_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            **summary,
            "prompt_details": [
                {"id": p["id"], "prompt": p["prompt"],
                 "expected_tools": p["expected_tools"],
                 "category": p.get("category")}
                for p in prompts
            ],
            "dummy_tool_names": [t["function"]["name"] for t in DUMMY_TOOL_SCHEMAS],
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
