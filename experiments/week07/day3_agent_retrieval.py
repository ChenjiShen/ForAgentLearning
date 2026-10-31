"""Week 7 Day 3: RAG for agents — search_knowledge tool vs everything-in-context.

Experiment question (PLAN.md week 7 day 3):
    RAG *for agents*: retrieval as a tool the agent calls vs. always-stuffed
    context.  Build: convert your RAG pipeline into a `search_knowledge` tool.
    Compare: agent-with-retrieval-tool vs. everything-in-context on 5 questions.

Design:

    INDEX — day 2's best chunking strategy (fixed_400, 5/6 accuracy) chunks the
    6 weekly reports, embedded once with text-embedding-v3 (day-1 client,
    reused unchanged).  One (doc, text) list + one numpy matrix — day 2's
    pipeline, now standing behind a tool boundary.

    CONDITIONS — the same 5 questions answered under three regimes:
      agent_with_tool       an agent with ONE tool, search_knowledge(query, k).
                            It decides whether to call it, when, and how often;
                            the chunks come back as tool results and the agent
                            answers from them.  Max 6 turns (typically 2: search
                            + answer).
      always_stuffed        no agent, no choice: day 2's pipeline verbatim —
                            top-4 chunks embedded/retrieved/stuffed on EVERY
                            question, whether retrieval was needed or not.
      everything_in_context no retrieval at all: the ENTIRE notebook (~500K
                            chars across the 6 reports, at the time of the
                            run) is stuffed into the user message on every
                            question.  This is the extreme baseline retrieval
                            exists to avoid — and week 6's lost-in-the-middle
                            lesson predicts facts buried mid-corpus get
                            missed.

    QUESTIONS — 5 of day 2's 6 (q_week6 dropped: it failed in ALL day-2
    conditions — its query phrasing matches the wrong passages, so it would
    only add retrieval noise to this comparison).  Each carries a rubric with
    the exact expected facts, verified in the reports.

    GRADING — the week-4 LLM judge (rubric + answer -> PASS/FAIL, validated
    against hand labels in week 4 day 3), not day 2's evidence grader: the
    judge accepts "10 out of 11" for 91% where the substring grader failed.
    Transport fix from week 7: trust_env=False so the dead macOS proxy
    cannot break the call.

    METRICS — per condition: judge pass rate, mean LLM turns, mean tool calls,
    mean prompt/completion tokens, total tokens, elapsed.  The contrast the
    plan wants: does an agent that pays for retrieval only when it decides to
    retrieve beat (a) always paying for stuffed top-k and (b) paying for the
    whole notebook on every question?

Run:
    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week07/day3_agent_retrieval.py --selftest   # offline checks, no API
    python experiments/week07/day3_agent_retrieval.py --smoke      # truncated docs, sanity pass
    python experiments/week07/day3_agent_retrieval.py              # full run

Outputs (in experiments/week07/):
    day3_results_{ts}.json — config, index stats, every answer + judge verdict
                             + trace + usage
    day3_plot_{ts}.png     — judge pass rate + mean prompt tokens per condition
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Day 1 handoff: embedding client + numpy search carry over unchanged.
from day1_embeddings import (  # noqa: E402
    embed_texts,
    get_client,
    normalize_rows,
    top_k_similarity,
)
# Day 2 handoff: chunker, prompt builder, system prompt (all reused verbatim).
# (Day 2's chat client is replaced by a local one with a longer read timeout —
# the everything-in-context call carries ~200K prompt tokens.)
from day2_mini_rag import (  # noqa: E402
    CHAT_BASE,
    CHAT_MODEL,
    DOCS,
    DOCS_DIR,
    SYSTEM_WITH_CONTEXT,
    build_user_prompt,
    chunk_fixed,
)

K = 4        # default top-k for the tool and for always_stuffed
MAX_K = 8    # tool clamps k to [1, MAX_K]
MAX_TURNS = 6  # agent safety valve: a question should need ~2 turns
CHUNK_SIZE = 400  # day 2's winning strategy (5/6)

# ---------------------------------------------------------------------------
# Questions — 5 of day 2's 6, same wording so results compare.  Each rubric
# carries the exact facts, verified in the reports (day-2 gold strings turned
# into natural language the judge can check without a substring grader).
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"id": "q_week1",
     "question": ("In the week 1 day 6 experiment on the trap_binary_file "
                  "task, what were the wrong-tool rates of terse vs detailed "
                  "tool descriptions?"),
     "rubric": ("The answer must state the wrong-tool rates from the week 1 "
                "day 6 trap_binary_file experiment: terse descriptions chose "
                "the wrong tool on 4 of 5 runs (4/5 or 80%), detailed "
                "descriptions on 0 of 5 runs (0/5 or 0%). Both numbers must "
                "be present and attributed to the correct description style.")},
    {"id": "q_week2",
     "question": ("In the week 2 temperature sweep, what success rates did "
                  "temperatures 0.0, 0.7, and 1.2 achieve on the eval tasks?"),
     "rubric": ("The answer must give the success rates for all three "
                "temperatures: 0.0 -> 97.5%, 0.7 -> 100%, 1.2 -> 97.5%. "
                "All three rates must be present and matched to the right "
                "temperature.")},
    {"id": "q_week3",
     "question": ("Roughly how many tokens does a raw HTML page cost, and "
                  "what fraction of that is useful content?"),
     "rubric": ("The answer must state that a raw HTML page cost roughly "
                "30,000 tokens and that only about 10% of those tokens were "
                "useful content. Both figures must be present.")},
    {"id": "q_week4",
     "question": ("In the week 4 substring-vs-LLM-judge experiment, what "
                  "fraction of the disagreements were substring false "
                  "negatives?"),
     "rubric": ("The answer must state that 91% of the disagreements were "
                "substring false negatives (the substring checker failing "
                "answers that were actually correct). The 91% figure must "
                "appear.")},
    {"id": "q_week5",
     "question": ("In week 5 day 1, how did rewriting the tool descriptions "
                  "with when-to-call guidance change token consumption?"),
     "rubric": ("The answer must state that rewriting the tool descriptions "
                "with when-to-call guidance increased token consumption by "
                "33.6%.")},
]

# ---------------------------------------------------------------------------
# Index — day 2's fixed_400 chunks over the 6 reports, embedded once.
# ---------------------------------------------------------------------------

def build_index(docs: dict[str, str], embed_client) -> tuple[list, np.ndarray, dict]:
    """Chunk every doc with fixed_400, embed once -> (flat, E_hat, stats)."""
    flat: list[tuple[str, str]] = []
    for name, text in docs.items():
        flat += [(name, c) for c in chunk_fixed(text, CHUNK_SIZE)]
    E, tokens = embed_texts(embed_client, [t for _, t in flat])
    lens = [len(t) for _, t in flat]
    stats = {"n_chunks": len(flat),
             "mean_chars": round(sum(lens) / len(lens), 1),
             "embed_tokens": tokens}
    return flat, normalize_rows(E), stats


# ---------------------------------------------------------------------------
# The search_knowledge tool — day 2's embed -> retrieve -> return, now behind
# a function boundary the agent can call.  Same envelope shape as the week-3
# tools (status/data/error/truncated) so a week-3-style agent needs no new
# result-parsing code.
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "Search the personal notebook (weekly reports week1-week6 of AI "
            "agent engineering experiments) and return the top-k most similar "
            "chunks with their source report. Call this whenever the question "
            "asks about specific experiments, measurements, or numbers "
            "recorded in the notebook. Reformulate the query and search again "
            "if the first results do not contain the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": ("Search query — keywords describing "
                                          "the fact you need, e.g. 'temperature "
                                          "sweep success rates'.")},
                "k": {"type": "integer",
                      "description": f"Number of chunks to return (default {K}, max {MAX_K})."},
            },
            "required": ["query"],
        },
    },
}


def _search_envelope(query: str, hits: list[tuple[str, str, float]]) -> str:
    """Wrap (doc, text, similarity) hits in the week-3 tool-result envelope."""
    results = [{"doc": doc, "text": text, "similarity": round(sim, 4)}
               for doc, text, sim in hits]
    return json.dumps({"status": "success",
                       "data": {"query": query, "results": results,
                                "count": len(results)},
                       "error": None, "truncated": False,
                       "truncated_at": None},
                      ensure_ascii=False)


def make_search(flat: list[tuple[str, str]], E_hat: np.ndarray, embed_client):
    """Close over the index; return search_knowledge(query, k) -> envelope str."""
    def search_knowledge(query: str, k: int | None = None) -> str:
        query = (query or "").strip()
        if not query:
            return json.dumps(
                {"status": "error", "data": None,
                 "error": {"type": "invalid_input",
                           "message": "query must be a non-empty string"},
                 "truncated": False, "truncated_at": None})
        try:
            k = max(1, min(int(k or K), MAX_K))
        except (TypeError, ValueError):
            k = K
        q_vec, _ = embed_texts(embed_client, [query])
        q_hat = normalize_rows(q_vec)[0]
        idx, sims = top_k_similarity(E_hat, q_hat, min(k, len(flat)))
        return _search_envelope(query,
                                [(flat[i][0], flat[i][1], float(s))
                                 for i, s in zip(idx, sims)])
    return search_knowledge


# ---------------------------------------------------------------------------
# Condition 1: agent-with-retrieval-tool — one tool, the agent decides.
# ---------------------------------------------------------------------------

AGENT_SYSTEM = (
    "You are a research assistant with access to a personal notebook of AI "
    "agent engineering experiment notes (weekly reports week1-week6). The "
    "notebook is NOT in your context — the only way to read it is your "
    "search_knowledge tool, which returns the top-k most similar chunks. "
    "Call search_knowledge whenever the question asks about specific "
    "experiments, measurements, or numbers from the notebook. Make AT MOST "
    "3 searches in total — if the answer is still missing after 3 searches, "
    "answer from what you have or say exactly: I don't know from the "
    "notebook. Answer ONLY from what search_knowledge returns. Give exact "
    "numbers when the chunks contain them."
)


def run_agent_with_tool(chat_client, question: str, search_fn) -> dict:
    """Non-interactive agent loop (week-3 day7 pattern) with one tool."""
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": question}]
    usage = {"prompt": 0, "completion": 0}
    trace: list[dict] = []

    for turn in range(1, MAX_TURNS + 1):
        r = chat_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=[SEARCH_TOOL],
            temperature=0.0)
        if r.usage:
            usage["prompt"] += r.usage.prompt_tokens or 0
            usage["completion"] += r.usage.completion_tokens or 0

        msg = r.choices[0].message
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return {"answer": msg.content or "[no text]", "n_turns": turn,
                    "tool_calls": trace, "usage": usage, "error": None}

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                trace.append({"name": tc.function.name, "error": f"bad JSON: {e}"})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": _search_envelope("", [])})
                continue
            output = search_fn(args.get("query", ""), args.get("k"))
            try:
                data = json.loads(output).get("data", {})
                docs_hit = sorted({x["doc"] for x in data.get("results", [])})
                n_res = data.get("count", 0)
            except json.JSONDecodeError:
                docs_hit, n_res = [], 0
            trace.append({"name": tc.function.name,
                          "query": args.get("query", ""),
                          "n_results": n_res, "docs_hit": docs_hit})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": output})

    return {"answer": f"[stopped after {MAX_TURNS} turns]",
            "n_turns": MAX_TURNS, "tool_calls": trace, "usage": usage,
            "error": "max_turns_exceeded"}


# ---------------------------------------------------------------------------
# Condition 2: always_stuffed — day 2's pipeline verbatim (single call, top-k
# chunks stuffed on every question).
# ---------------------------------------------------------------------------

def run_always_stuffed(chat_client, question: str, q_hat: np.ndarray,
                       flat, E_hat) -> dict:
    """Day-2 path: retrieve top-k, stuff, answer. No agent, no choice."""
    idx, _ = top_k_similarity(E_hat, q_hat, K)
    retrieved = [{"chunk_id": f"{flat[i][0]}#{i:04d}", "doc": flat[i][0]}
                 for i in idx]
    r = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_WITH_CONTEXT},
                  {"role": "user",
                   "content": build_user_prompt(question, [flat[i] for i in idx])}],
        temperature=0.0)
    usage = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}
    return {"answer": r.choices[0].message.content.strip(), "retrieved": retrieved,
            "usage": usage, "error": None}


# ---------------------------------------------------------------------------
# Condition 3: everything_in_context — the whole notebook in the user message.
# ---------------------------------------------------------------------------

EVERYTHING_SYSTEM = (
    "You are answering questions from a personal notebook on AI agent "
    "engineering. The complete notebook is in the user message. Answer ONLY "
    "from the notebook. Give exact numbers when the notebook has them. If the "
    "notebook does not contain the answer, say exactly: I don't know from the "
    "notebook."
)


def build_corpus(docs: dict[str, str],
                 cap_chars: int | None = None) -> tuple[str, list[str]]:
    """Concatenate all documents with source headers, notebook order.

    With cap_chars, drop whole trailing documents until the corpus fits —
    a document is the unit, so nothing is ever cut mid-way.  Returns
    (corpus, docs_included) so the results record exactly what was in
    context when the cap kicked in.
    """
    names = list(docs)
    pieces = [f"### {n}\n{docs[n]}" for n in names]
    if cap_chars is None:
        return "\n\n".join(pieces), names
    text, included = "", []
    for n, p in zip(names, pieces):
        candidate = f"{text}\n\n{p}" if text else p
        if len(candidate) > cap_chars:
            break
        text, included = candidate, included + [n]
    return text, included


def _looks_like_context_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("context", "maximum", "too long",
                                  "exceed", "reduce the length"))


def run_everything_in_context(chat_client, question: str,
                              attempts: list[dict]) -> dict:
    """One chat call with the whole notebook stuffed.  Each attempt dict
    carries (text, docs_included, truncated); if the API rejects an attempt
    for exceeding the context window, the next (smaller) attempt runs —
    and the result records exactly what was included."""
    for i, att in enumerate(attempts):
        try:
            r = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": EVERYTHING_SYSTEM},
                          {"role": "user",
                           "content": (f"NOTEBOOK (complete):\n\n{att['text']}\n\n"
                                       f"Question: {question}\n\nAnswer:")}],
                temperature=0.0)
            usage = {"prompt": r.usage.prompt_tokens,
                     "completion": r.usage.completion_tokens}
            return {"answer": r.choices[0].message.content.strip(),
                    "corpus_chars": len(att["text"]),
                    "docs_included": att["docs"],
                    "truncated": att["truncated"],
                    "usage": usage, "error": None}
        except Exception as e:
            if _looks_like_context_error(e) and i < len(attempts) - 1:
                print(f"    ! corpus of {len(att['text']):,} chars rejected "
                      f"({str(e)[:50]}...) -> retrying with "
                      f"{len(attempts[i + 1]['text']):,} chars "
                      f"(docs: {', '.join(attempts[i + 1]['docs'])})")
                continue
            return {"answer": f"[API error: {type(e).__name__}: {str(e)[:200]}]",
                    "corpus_chars": len(att["text"]),
                    "docs_included": att["docs"], "truncated": None,
                    "usage": {"prompt": 0, "completion": 0}, "error": str(e)[:200]}
    return {"answer": "[unreachable]", "corpus_chars": 0, "docs_included": [],
            "truncated": None, "usage": {"prompt": 0, "completion": 0},
            "error": "unreachable"}


# ---------------------------------------------------------------------------
# Grading — the week-4 LLM judge (rubric before answer, PASS/FAIL first token,
# temperature 0).  Week-4 prompt reused; week-7 transport (trust_env=False).
# ---------------------------------------------------------------------------

def llm_judge(judge_client, question: str, rubric: str, answer: str) -> dict:
    judge_prompt = (
        "You are an impartial evaluator. Your job is to judge whether an AI "
        "assistant's answer correctly addresses a given task.\n\n"
        "## Task\n"
        f"{question}\n\n"
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
        r = judge_client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.0,
            messages=[{"role": "user", "content": judge_prompt}])
        verdict = (r.choices[0].message.content or "").strip().upper()
        usage = {"prompt": r.usage.prompt_tokens if r.usage else 0,
                 "completion": r.usage.completion_tokens if r.usage else 0}
    except Exception as e:
        return {"pass": False, "verdict": f"llm_judge (API error: {e})",
                "usage": {"prompt": 0, "completion": 0}}
    if verdict.startswith("PASS"):
        return {"pass": True, "verdict": f"llm_judge ({verdict[:50]})", "usage": usage}
    if verdict.startswith("FAIL"):
        return {"pass": False, "verdict": f"llm_judge ({verdict[:50]})", "usage": usage}
    return {"pass": False, "verdict": f"llm_judge (unexpected: {verdict[:80]})",
            "usage": usage}


# ---------------------------------------------------------------------------
# Summarize + report + plot.
# ---------------------------------------------------------------------------

def summarize(runs: list[dict], conditions: list[str]) -> dict:
    metrics = {}
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        n = len(rs)
        metrics[cond] = {
            "accuracy": round(sum(r["judge"]["pass"] for r in rs) / n, 3),
            "mean_prompt_tokens": round(sum(r["usage"]["prompt"] for r in rs) / n, 1),
            "mean_completion_tokens": round(sum(r["usage"]["completion"] for r in rs) / n, 1),
            "total_tokens_gen": sum(r["usage"]["prompt"] + r["usage"]["completion"] for r in rs),
            "total_tokens_judge": sum(r["judge"]["usage"]["prompt"] + r["judge"]["usage"]["completion"]
                                      for r in rs),
            "mean_elapsed_s": round(sum(r["elapsed_s"] for r in rs) / n, 1),
            "mean_tool_calls": round(sum(len(r["trace"].get("tool_calls", []))
                                         for r in rs) / n, 2),
            # only the agent condition has turns; the other two are single calls
            "mean_turns": (round(sum(r["trace"].get("n_turns", 0) for r in rs) / n, 2)
                           if cond == "agent_with_tool" else None),
        }
    return {"metrics": metrics, "n_questions": len(QUESTIONS), "k": K}


def print_report(results: dict) -> None:
    m = results["metrics"]
    print(f"\n{'condition':<24} {'acc':>5} {'turns':>6} {'tools/q':>8} "
          f"{'prompt/q':>10} {'compl/q':>8} {'tot_gen':>9} {'judge_tok':>9} "
          f"{'sec/q':>7}")
    for cond, mm in m.items():
        turns = f"{mm['mean_turns']:.1f}" if mm["mean_turns"] is not None else "—"
        print(f"{cond:<24} {mm['accuracy']:>5.3f} {turns:>6} "
              f"{mm['mean_tool_calls']:>8.2f} {mm['mean_prompt_tokens']:>10.1f} "
              f"{mm['mean_completion_tokens']:>8.1f} {mm['total_tokens_gen']:>9,} "
              f"{mm['total_tokens_judge']:>9,} {mm['mean_elapsed_s']:>7.1f}")

    print("\nper-question judge verdicts:")
    for r in results["runs"]:
        mark = "PASS" if r["judge"]["pass"] else "fail"
        t = r["trace"]
        if r["condition"] == "agent_with_tool":
            calls = "; ".join(f"{c['query']!r}->{c.get('n_results', 0)} "
                              f"({','.join(c.get('docs_hit', []))})"
                              for c in t["tool_calls"]) or "(no tool calls)"
            extra = f"  {t['n_turns']} turn(s): {calls}"
        elif r["condition"] == "always_stuffed":
            extra = f"  stuffed: {', '.join(c['chunk_id'] for c in t['retrieved'])}"
        else:
            trunc = (f", DROPPED DOCS — kept: {', '.join(t['docs_included'])}"
                     if t.get("truncated") else "")
            extra = f"  corpus: {t['corpus_chars']:,} chars{trunc}"
        print(f"  {r['condition']:<24} {r['query_id']:<10} {mark}  {extra}")


# Categorical hues: reference palette slots, fixed entity order (validated
# CVD-safe).  Contrast WARN on green/amber -> direct value labels required,
# which both panels have.
COND_COLORS = {
    "agent_with_tool":       "#2a78d6",
    "always_stuffed":        "#1baf7a",
    "everything_in_context": "#eda100",
}
COND_LABELS = {
    "agent_with_tool":       "agent\n+tool",
    "always_stuffed":        "always\nstuffed",
    "everything_in_context": "everything\nin-context",
}


def plot_results(results: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = results["metrics"]
    conditions = [c for c in COND_COLORS if c in m]
    labels = [COND_LABELS[c] for c in conditions]
    colors = [COND_COLORS[c] for c in conditions]
    accs = [m[c]["accuracy"] for c in conditions]
    prompts = [m[c]["mean_prompt_tokens"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ink, grid = "#0b0b0b", "#d8d6d0"

    # (a) judge pass rate per condition — direct value labels, muted grid
    ax1.bar(labels, accs, width=0.62, color=colors)
    for x, v in zip(labels, accs):
        ax1.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=ink)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("judge pass rate")
    ax1.set_title(f"Answer quality by condition (n={results['n_questions']} questions)")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # (b) mean prompt tokens per question, log scale — everything-in-context
    # dwarfs the other two by ~100x, which is the cost half of the lesson
    ax2.bar(labels, prompts, width=0.62, color=colors)
    for x, v in zip(labels, prompts):
        ax2.text(x, v * 1.35, f"{v / 1000:.0f}K", ha="center", fontsize=9, color=ink)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean prompt tokens per question (log)")
    ax2.set_title(f"Prompt cost by condition (k={results['k']})")
    ax2.tick_params(axis="x", labelsize=8)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nplot -> {out_path}")


# ---------------------------------------------------------------------------
# Offline selftest — no API calls.  Proves the tool envelope, the prompts,
# and the rubrics before spending a token.
# ---------------------------------------------------------------------------

def selftest() -> None:
    checks: list[bool] = []

    def check(name: str, cond: bool) -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. tool schema is valid JSON with the required query parameter
    schema = json.dumps(SEARCH_TOOL)
    check("tool schema: parses as JSON", isinstance(json.loads(schema), dict))
    check("tool schema: name is search_knowledge",
          SEARCH_TOOL["function"]["name"] == "search_knowledge")
    check("tool schema: query is required",
          "query" in SEARCH_TOOL["function"]["parameters"]["required"])

    # 2. search envelope: week-3 shape, hits carried in order
    env = json.loads(_search_envelope("temperature sweep",
                                      [("week2.md", "alpha", 0.9),
                                       ("week5.md", "beta", 0.7)]))
    check("envelope: status=success, count=2",
          env["status"] == "success" and env["data"]["count"] == 2)
    check("envelope: hits ordered with doc/text/similarity",
          [h["doc"] for h in env["data"]["results"]] == ["week2.md", "week5.md"])
    check("envelope: similarity rounded to 4 places",
          env["data"]["results"][0]["similarity"] == 0.9)

    # 3. index math: query == corpus row -> that row is top-1
    rng = np.random.default_rng(3)
    E = rng.normal(size=(9, 16))
    E_hat = normalize_rows(E)
    q_hat = E_hat[4]
    idx, _ = top_k_similarity(E_hat, q_hat, 4)
    check("index: identical vector ranks top-1", int(idx[0]) == 4)

    # 4. everything-in-context prompt: corpus + question + doc headers present
    corpus, included = build_corpus({"week1.md": "alpha text",
                                     "week2.md": "beta text"})
    prompt = (f"NOTEBOOK (complete):\n\n{corpus}\n\n"
              f"Question: what?\n\nAnswer:")
    check("everything: question present", "what?" in prompt)
    check("everything: full corpus present",
          "alpha text" in prompt and "beta text" in prompt)
    check("everything: doc headers present",
          "### week1.md" in prompt and "### week2.md" in prompt)
    check("everything: no cap includes every doc",
          included == ["week1.md", "week2.md"])
    capped, capped_inc = build_corpus(
        {"week1.md": "x" * 60, "week2.md": "y" * 60, "week3.md": "z" * 60},
        cap_chars=100)
    check("everything: cap drops whole trailing docs, never cuts one",
          capped_inc == ["week1.md"] and len(capped) < 100
          and "y" not in capped)

    # 5. always-stuffed prompt reuses day-2 builder (question + all chunks)
    stuffed = [("week6.md", "chunk one"), ("week6.md", "chunk two")]
    p2 = build_user_prompt("what share?", stuffed)
    check("stuffed: question present", "what share?" in p2)
    check("stuffed: both chunks present", all(t in p2 for _, t in stuffed))

    # 6. rubrics carry their key facts (guards the gold numbers end-to-end)
    expected_facts = {
        "q_week1": ["4 of 5", "0 of 5"],
        "q_week2": ["97.5%", "100%"],
        "q_week3": ["30,000", "10%"],
        "q_week4": ["91%"],
        "q_week5": ["33.6%"],
    }
    for q in QUESTIONS:
        check(f"rubric {q['id']}: all key facts present",
              all(f in q["rubric"] for f in expected_facts[q["id"]]))

    # 7. chunker carries over: fixed_400 lossless reconstruction
    text = "0123456789abcdefghijklmnopqrstuvwxyz" * 3
    check("chunker: fixed_400 reconstruction is lossless",
          "".join(chunk_fixed(text, 400)) == text)

    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Week 7 Day 3: search_knowledge tool vs everything-in-context")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline checks only (no API calls)")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate each doc to 10k chars (quick sanity pass)")
    ap.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs per (condition, question) cell")
    ap.add_argument("--ctx-cap-chars", type=int, default=310_000,
                    help="fallback char cap (whole docs dropped from the end) "
                         "for everything-in-context if the full corpus is "
                         "rejected (default 310000: fits week1-week4)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    docs = {}
    for name in DOCS:
        path = DOCS_DIR / name
        if not path.exists():
            sys.exit(f"missing document: {path}")
        docs[name] = path.read_text()[:10_000 if args.smoke else None]
    total_chars = sum(len(t) for t in docs.values())
    print(f"docs: {len(docs)} files, {total_chars:,} chars (smoke={args.smoke}); "
          f"questions: {len(QUESTIONS)}; k={K}; chat={CHAT_MODEL}; "
          f"runs={args.runs}; ctx-cap={args.ctx_cap_chars}")

    embed_client = get_client()
    # Same client config as day 2 but with a 600s read timeout: the
    # everything-in-context call carries ~200K prompt tokens, which can take
    # longer than day 2's 300s cap to stream back.
    chat_client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=CHAT_BASE,
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(600.0, connect=15.0)),
    )
    t0 = time.time()

    print("building index (fixed_400) ...")
    flat, E_hat, index_stats = build_index(docs, embed_client)
    print(f"  {index_stats['n_chunks']} chunks, "
          f"{index_stats['embed_tokens']} embed tokens")

    print("embedding queries ...")
    Q, q_tokens = embed_texts(embed_client, [q["question"] for q in QUESTIONS])
    q_hat = normalize_rows(Q)

    search_fn = make_search(flat, E_hat, embed_client)
    corpus, all_docs = build_corpus(docs)
    print(f"corpus for everything-in-context: {len(corpus):,} chars "
          f"(~{len(corpus) // 2:,} chat tokens)")
    # attempts: full corpus first; if the API rejects it for context length,
    # fall back to the biggest whole-doc prefix that fits --ctx-cap-chars
    everything_attempts = [{"text": corpus, "docs": all_docs,
                            "truncated": False}]
    if args.ctx_cap_chars and len(corpus) > args.ctx_cap_chars:
        capped, capped_docs = build_corpus(docs, cap_chars=args.ctx_cap_chars)
        everything_attempts.append({"text": capped, "docs": capped_docs,
                                    "truncated": True})

    conditions = ["agent_with_tool", "always_stuffed", "everything_in_context"]
    runs: list[dict] = []
    for r_i in range(args.runs):
        for cond in conditions:
            for qi, question in enumerate(QUESTIONS):
                cell_t0 = time.time()
                if cond == "agent_with_tool":
                    res = run_agent_with_tool(chat_client, question["question"],
                                              search_fn)
                    trace = {"tool_calls": res["tool_calls"],
                             "n_turns": res["n_turns"]}
                elif cond == "always_stuffed":
                    res = run_always_stuffed(chat_client, question["question"],
                                             q_hat[qi], flat, E_hat)
                    trace = {"retrieved": res["retrieved"]}
                else:
                    res = run_everything_in_context(chat_client,
                                                    question["question"],
                                                    everything_attempts)
                    trace = {"corpus_chars": res["corpus_chars"],
                             "docs_included": res["docs_included"],
                             "truncated": res["truncated"]}
                elapsed = time.time() - cell_t0

                judge = llm_judge(chat_client, question["question"],
                                  question["rubric"], res["answer"])
                mark = "PASS" if judge["pass"] else "fail"
                n_extra = (f"  {res['n_turns']}t/{len(res['tool_calls'])} calls"
                           if cond == "agent_with_tool"
                           else f"  {res.get('corpus_chars', 0):,} chars"
                           + f" ({','.join(res.get('docs_included', []))})"
                           if cond == "everything_in_context" else "")
                print(f"  [{r_i}] {cond:<24} {question['id']:<10} {mark}"
                      f"{n_extra}  (prompt {res['usage']['prompt']:,} tok, "
                      f"{elapsed:.0f}s)")

                runs.append({
                    "condition": cond, "query_id": question["id"], "run": r_i,
                    "question": question["question"], "rubric": question["rubric"],
                    "answer": res["answer"], "judge": judge,
                    "trace": trace,
                    "usage": res["usage"],
                    "elapsed_s": round(elapsed, 1),
                    "error": res.get("error"),
                })

    print(f"\nelapsed {time.time() - t0:.0f}s  (query embed tokens: {q_tokens})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "week07_day3_agent_retrieval",
        "timestamp": ts,
        "config": {
            "docs": DOCS, "doc_chars": {d: len(t) for d, t in docs.items()},
            "chunker": f"fixed {CHUNK_SIZE} chars, step {CHUNK_SIZE}",
            "k": K, "runs": args.runs, "smoke": args.smoke,
            "ctx_cap_chars": args.ctx_cap_chars,
            "max_turns": MAX_TURNS,
            "chat": {"model": CHAT_MODEL, "base_url": CHAT_BASE,
                     "temperature": 0.0},
            "conditions": {
                "agent_with_tool": "agent + search_knowledge tool, decides when to retrieve",
                "always_stuffed": "day-2 pipeline: top-k chunks stuffed on every question, no choice",
                "everything_in_context": ("whole notebook stuffed into the user message; "
                                          "falls back to a whole-doc prefix if the API "
                                          "rejects the full corpus"),
            },
        },
        "index": index_stats,
        "runs": runs,
        **summarize(runs, conditions),
    }
    out_path = _THIS_DIR / f"day3_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"results -> {out_path}")

    print_report(results)

    if not args.no_plot:
        plot_path = _THIS_DIR / f"day3_plot_{ts}.png"
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
