"""Week 7 Day 2: mini RAG pipeline over real note documents.

Experiment question (PLAN.md week 7 day 2):
    Chunking: sizes, overlap, why chunking strategy dominates RAG quality.
    Build: chunk -> embed -> retrieve -> stuff into prompt.

Design:

    DOCS — reports/week1.md .. week6.md, this repo's own weekly notes
    (~370K chars, 6 documents). Real documents, not pre-cut texts: day 1's
    corpus was hand-harvested one-liners; day 2 must chunk raw markdown.

    PIPELINE — the four moves the plan names, nothing else:
        chunk   : four strategies (see CHUNKERS), the experiment variable
        embed   : DashScope text-embedding-v3 (day-1 client, reused unchanged)
        retrieve: top-k cosine via day-1 numpy matmul (k=4)
        stuff   : numbered chunks + question -> qwen3.7-plus chat answer

    CHUNKING SWEEP — the day's lesson is that chunking dominates RAG quality,
    so the experiment is a sweep with two clean single-variable contrasts:
        fixed_400       fixed 400 chars, step 400   } size: 400 vs 800
        fixed_800       fixed 800 chars, step 800   }
        fixed_800_o100  fixed 800 chars, step 700   } overlap: 0 vs 100
        para_800        paragraph-merge, target 800 } structure-aware
    plus a no_retrieval baseline (same questions, no context) to show
    retrieval actually helps — week 7's real goal is knowing when it does.

    QUERIES — one per document, each with a gold fact that lives in that
    report (exact numbers verified against the reports). Grading is evidence
    presence: every gold string must appear in the normalized answer. The
    facts are private to these notes, so the no-retrieval baseline should
    fail nearly everything — that failure IS the "retrieval helps" result.

    METRICS — per condition: accuracy (all gold facts present), retrieval
    quality (fraction of top-k chunks from the gold document — separates
    retrieval failures from generation failures), prompt tokens per question
    (what stuffing costs).

Run:
    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week07/day2_mini_rag.py --selftest   # offline checks, no API
    python experiments/week07/day2_mini_rag.py --smoke      # quick sanity pass, truncated docs
    python experiments/week07/day2_mini_rag.py              # full run (~240 embed + 24 chat calls)

Outputs (in experiments/week07/):
    day2_results_{ts}.json — config, chunk stats, every answer + grade + usage
    day2_plot_{ts}.png     — accuracy per condition + retrieval hit per strategy
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Day 1 handoff (its notes.md): the embedding client carries over unchanged.
from day1_embeddings import (  # noqa: E402
    embed_texts,
    get_client,
    normalize_rows,
    top_k_similarity,
)

CHAT_MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
CHAT_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

DOCS = ["week1.md", "week2.md", "week3.md", "week4.md", "week5.md", "week6.md"]
DOCS_DIR = _THIS_DIR.parent.parent / "reports"

K = 4  # top-k chunks stuffed per question

# ---------------------------------------------------------------------------
# Chunkers — the experiment variable. All take full text, return list[str].
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int, overlap: int = 0) -> list[str]:
    """Fixed-size sliding window. step = size - overlap; last chunk may be short."""
    if not (0 <= overlap < size):
        raise ValueError(f"need 0 <= overlap < size, got size={size} overlap={overlap}")
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)] if text else []


def chunk_paragraphs(text: str, target: int) -> list[str]:
    """Structure-aware: split on blank lines, merge paragraphs up to target.

    A single paragraph longer than target is hard-split (it must be cut
    somewhere; fixed-size is the honest fallback). Everything else stays
    paragraph-intact — the claim to test is that not slicing mid-thought
    beats fancier fixed windows.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > target:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(chunk_fixed(p, target))
            continue
        if buf and len(buf) + 1 + len(p) > target:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


CHUNKERS = {
    "fixed_400":      lambda t: chunk_fixed(t, 400),
    "fixed_800":      lambda t: chunk_fixed(t, 800),
    "fixed_800_o100": lambda t: chunk_fixed(t, 800, overlap=100),
    "para_800":       lambda t: chunk_paragraphs(t, 800),
}

# ---------------------------------------------------------------------------
# Questions — one per document, gold facts verified in the reports.
# Grading = evidence presence: ALL gold strings must survive normalization.
# ---------------------------------------------------------------------------

QUERIES = [
    {"id": "q_week1", "gold_doc": "week1.md",
     "question": ("In the week 1 day 6 experiment on the trap_binary_file "
                  "task, what were the wrong-tool rates of terse vs detailed "
                  "tool descriptions?"),
     "gold": ["4/5", "0/5"]},
    {"id": "q_week2", "gold_doc": "week2.md",
     "question": ("In the week 2 temperature sweep, what success rates did "
                  "temperatures 0.0, 0.7, and 1.2 achieve on the eval tasks?"),
     "gold": ["97.5", "100"]},
    {"id": "q_week3", "gold_doc": "week3.md",
     "question": ("Roughly how many tokens does a raw HTML page cost, and "
                  "what fraction of that is useful content?"),
     "gold": ["30000", "10"]},
    {"id": "q_week4", "gold_doc": "week4.md",
     "question": ("In the week 4 substring-vs-LLM-judge experiment, what "
                  "fraction of the disagreements were substring false "
                  "negatives?"),
     "gold": ["91"]},
    {"id": "q_week5", "gold_doc": "week5.md",
     "question": ("In week 5 day 1, how did rewriting the tool descriptions "
                  "with when-to-call guidance change token consumption?"),
     "gold": ["33.6"]},
    {"id": "q_week6", "gold_doc": "week6.md",
     "question": ("In the week 6 20-turn session, what share of input tokens "
                  "were tool results, and what was the input:output ratio?"),
     "gold": ["45.6", "37:1"]},
]

# ---------------------------------------------------------------------------
# Grading: evidence presence on a normalized answer.
# ---------------------------------------------------------------------------

_FULLWIDTH = str.maketrans(
    "，。．：；！？（）％"
    + "".join(chr(0xFF10 + i) for i in range(10))   # fullwidth digits ０-９
    + "".join(chr(0xFF21 + i) for i in range(26))   # fullwidth latin Ａ-Ｚ
    + "".join(chr(0xFF41 + i) for i in range(26)),  # fullwidth latin ａ-ｚ
    ",..:;!?()%" + "0123456789"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz",
)


def normalize(s: str) -> str:
    """Lowercase, fold fullwidth punctuation, drop whitespace/commas/markdown.

    "1,909" and "1909" become the same string; "45.6 %" matches "45.6%";
    "**97.5**" matches "97.5". Exact numbers are the gold facts here, so
    comma/percent formatting must not decide the grade.
    """
    s = s.lower().translate(_FULLWIDTH)
    s = re.sub(r"[*_`#\s,]", "", s)
    return s


def grade(answer: str, gold: list[str]) -> dict:
    """ALL gold strings present in the normalized answer -> pass."""
    n = normalize(answer)
    found = [g for g in gold if normalize(g) in n]
    return {"pass": len(found) == len(gold), "found": found,
            "missing": [g for g in gold if g not in found]}


# ---------------------------------------------------------------------------
# Stuff + chat.
# ---------------------------------------------------------------------------

SYSTEM_WITH_CONTEXT = (
    "You are answering questions from a personal notebook on AI agent "
    "engineering. Answer ONLY from the retrieved context below. Give "
    "concrete numbers when the context has them. If the context does not "
    "contain the answer, say exactly: I don't know from the context."
)
SYSTEM_NO_CONTEXT = (
    "You are a helpful assistant. Answer from your own knowledge. If you do "
    "not know, say so."
)


def get_chat_client() -> OpenAI:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        sys.exit("DASHSCOPE_API_KEY not set. Source it first:\n"
                 "  export $(grep DASHSCOPE_API_KEY "
                 "../../.env | xargs)")
    http = httpx.Client(trust_env=False,
                        timeout=httpx.Timeout(300.0, connect=15.0))
    return OpenAI(api_key=key, base_url=CHAT_BASE, http_client=http)


def build_user_prompt(question: str, chunks: list[tuple[str, str]] | None) -> str:
    """Stuff retrieved chunks (doc, text) into the user message."""
    if chunks is None:
        return f"Question: {question}\n\nAnswer:"
    ctx = "\n".join(f"[{i + 1}] {doc}: {text}" for i, (doc, text) in enumerate(chunks))
    return f"Retrieved context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def chat_answer(client: OpenAI, question: str,
                chunks: list[tuple[str, str]] | None) -> tuple[str, dict]:
    """One chat call with (or without) stuffed context -> (answer, usage)."""
    system = SYSTEM_WITH_CONTEXT if chunks is not None else SYSTEM_NO_CONTEXT
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": build_user_prompt(question, chunks)}],
        temperature=0.0,
    )
    u = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}
    return r.choices[0].message.content.strip(), u


# ---------------------------------------------------------------------------
# Experiment core: chunk every doc per strategy, embed once per strategy,
# retrieve, stuff, answer, grade.
# ---------------------------------------------------------------------------

def chunk_docs(docs: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """chunk_docs -> {strategy: [(doc, chunk_text), ...]} over all documents."""
    out = {}
    for name, fn in CHUNKERS.items():
        flat: list[tuple[str, str]] = []
        for doc, text in docs.items():
            flat += [(doc, c) for c in fn(text)]
        out[name] = flat
    return out


def run_condition(condition: str, question: dict, chat_client,
                  q_hat, qi: int, chunks: list[tuple[str, str]],
                  corpus_hat) -> dict:
    """One (condition, question) cell: retrieve -> stuff -> answer -> grade."""
    if condition == "no_retrieval":
        retrieved = []
        answer, usage = chat_answer(chat_client, question["question"], None)
    else:
        idx, sims = top_k_similarity(corpus_hat, q_hat[qi], K)
        retrieved = [{"chunk_id": f"{chunks[i][0]}#{i:04d}", "doc": chunks[i][0],
                      "sim": round(float(s), 4)} for i, s in zip(idx, sims)]
        stuffed = [chunks[i] for i in idx]
        answer, usage = chat_answer(chat_client, question["question"], stuffed)

    g = grade(answer, question["gold"])
    gold_hits = sum(1 for c in retrieved if c["doc"] == question["gold_doc"])
    return {
        "condition": condition, "query_id": question["id"],
        "question": question["question"], "gold": question["gold"],
        "gold_doc": question["gold_doc"],
        "retrieved": retrieved,
        "gold_doc_hits": gold_hits,
        "answer": answer,
        "grade": g,
        "usage": usage,
    }


def summarize(runs: list[dict], conditions: list[str],
              chunk_stats: dict) -> dict:
    """Per-condition metrics: accuracy, retrieval hit, token cost."""
    metrics = {}
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        acc = sum(r["grade"]["pass"] for r in rs) / len(rs)
        if cond == "no_retrieval":
            hit = None
        else:
            hit = sum(r["gold_doc_hits"] / K for r in rs) / len(rs)
        metrics[cond] = {
            "accuracy": round(acc, 3),
            "gold_doc_hit_rate": round(hit, 3) if hit is not None else None,
            "mean_prompt_tokens": round(sum(r["usage"]["prompt"] for r in rs) / len(rs), 1),
            "mean_completion_tokens": round(sum(r["usage"]["completion"] for r in rs) / len(rs), 1),
            "total_tokens": sum(r["usage"]["prompt"] + r["usage"]["completion"] for r in rs),
        }
    return {"metrics": metrics, "chunk_stats": chunk_stats,
            "k": K, "n_queries": len(QUERIES)}


def print_report(results: dict) -> None:
    m = results["metrics"]
    print(f"\n{'condition':<16} {'acc':>5} {'hit@doc':>8} "
          f"{'prompt/q':>9} {'compl/q':>8} {'tot_tok':>8}  {'chunks':>7}")
    for cond, mm in m.items():
        n_chunks = results["chunk_stats"].get(cond, {}).get("n_chunks", "—")
        hit = f"{mm['gold_doc_hit_rate']:.3f}" if mm["gold_doc_hit_rate"] is not None else "—"
        print(f"{cond:<16} {mm['accuracy']:>5.3f} {hit:>8} "
              f"{mm['mean_prompt_tokens']:>9.1f} {mm['mean_completion_tokens']:>8.1f} "
              f"{mm['total_tokens']:>8}  {str(n_chunks):>7}")

    print("\nper-question grades (gold facts in answer):")
    for r in results["runs"]:
        mark = "PASS" if r["grade"]["pass"] else "fail"
        miss = f"  missing: {r['grade']['missing']}" if r["grade"]["missing"] else ""
        retr = "  retrieved: " + ", ".join(c["chunk_id"] for c in r["retrieved"]) \
            if r["retrieved"] else "  retrieved: (none)"
        print(f"  {r['condition']:<16} {r['query_id']:<10} {mark}{miss}{retr}")


# ---------------------------------------------------------------------------
# Offline selftest: prove chunkers + grader before any API call.
# ---------------------------------------------------------------------------

def selftest() -> None:
    checks: list[bool] = []

    def check(name: str, cond: bool) -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. fixed chunker: lossless reconstruction, size caps
    text = "0123456789abcdefghij"
    chunks = chunk_fixed(text, 8)
    check("fixed: reconstruction is lossless", "".join(chunks) == text)
    check("fixed: no chunk exceeds size", all(len(c) <= 8 for c in chunks))
    check("fixed: no empty chunks", all(c for c in chunks))

    # 2. overlap invariant: chunk[i] tail == chunk[i+1] head for full chunks
    long_text = "x" * 700
    ov = chunk_fixed(long_text, 400, overlap=100)
    full_pairs = [(a, b) for a, b in zip(ov, ov[1:]) if len(a) == 400 and len(b) == 400]
    check("overlap: every full chunk pair shares exactly 100 chars",
          bool(full_pairs) and all(a[-100:] == b[:100] for a, b in full_pairs))
    check("overlap: step shrinks by overlap (more chunks than no-overlap)",
          len(ov) > len(chunk_fixed(long_text, 400)))

    # 3. paragraph chunker: paragraphs <= target stay intact
    paras = ["alpha beta gamma", "delta epsilon", "z" * 200, "eta theta"]
    doc = "\n\n".join(paras)
    pchunks = chunk_paragraphs(doc, 50)
    for p in paras[:2] + paras[3:]:
        check(f"para: paragraph intact in exactly one chunk ({p[:8]}...)",
              sum(1 for c in pchunks if p in c) == 1)
    check("para: every chunk within target (long paras hard-split)",
          all(len(c) <= 50 for c in pchunks))
    check("para: reconstruction covers every paragraph",
          all(any(p[:20] in c for c in pchunks) for p in paras))
    check("para: long paragraph is split (its pieces reconstruct it)",
          "".join(c for c in pchunks if c in paras[2]) == paras[2])

    # 4. normalize + grade: formatting must not decide the grade
    check("normalize: commas folded (1,909 == 1909)",
          normalize("1,909") == normalize("1909"))
    check("normalize: whitespace folded (45.6 % == 45.6%)",
          normalize("45.6 %") == normalize("45.6%"))
    check("normalize: fullwidth folded (９１％ == 91%)",
          normalize("９１％") == normalize("91%"))
    check("normalize: markdown folded (**97.5** == 97.5)",
          normalize("**97.5**") == normalize("97.5"))
    check("normalize: case folded", normalize("Tool Results") == normalize("tool results"))
    check("grade: all-gold AND semantics (one missing -> fail)",
          grade("input:output 37:1", ["45.6", "37:1"])["pass"] is False)
    check("grade: passes when every gold string is present",
          grade("45.6% tool results, 37:1 ratio", ["45.6", "37:1"])["pass"] is True)
    check("grade: gold not matched by its prefix (37 != 37:1)",
          grade("the ratio was 37", ["37:1"])["pass"] is False)

    # 5. stuff prompt: question + every chunk + its source must appear
    stuffed = [("week6.md", "alpha chunk text"), ("week6.md", "beta chunk text")]
    prompt = build_user_prompt("what share?", stuffed)
    check("stuff: question present", "what share?" in prompt)
    check("stuff: both chunks present", all(t in prompt for _, t in stuffed))
    check("stuff: sources present", all(d in prompt for d, _ in stuffed))

    # 6. retrieval integration: query == corpus row -> that row is top-1
    rng = np.random.default_rng(3)
    E = rng.normal(size=(7, 16))
    E_hat = normalize_rows(E)
    q_hat = E_hat[3]
    idx, _ = top_k_similarity(E_hat, q_hat, 4)
    check("retrieve: identical vector ranks top-1", int(idx[0]) == 3)

    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Plot: accuracy per condition (incl. baseline) + retrieval hit per strategy.
# ---------------------------------------------------------------------------

# Categorical hues: dataviz reference palette slots 1-4, fixed entity order;
# the baseline is a muted gray (identity: "no retrieval", not a strategy).
COND_COLORS = {
    "no_retrieval":   "#9CA3AF",
    "fixed_400":      "#2a78d6",
    "fixed_800":      "#eb6834",
    "fixed_800_o100": "#1baf7a",
    "para_800":       "#eda100",
}


def plot_results(results: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = results["metrics"]
    conditions = [c for c in COND_COLORS if c in m]
    accs = [m[c]["accuracy"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ink, grid = "#0b0b0b", "#d8d6d0"

    # (a) accuracy by condition; direct value labels (relief rule for the
    # low-contrast aqua/yellow bars), muted grid
    ax1.bar(conditions, accs, width=0.62,
            color=[COND_COLORS[c] for c in conditions])
    for x, v in zip(conditions, accs):
        ax1.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=ink)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("accuracy (all gold facts present)")
    ax1.set_title(f"Answer quality by condition (k={results['k']})")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # (b) retrieval quality per chunking strategy (no baseline — it has no
    # retrieval to measure)
    strat = [c for c in conditions if c != "no_retrieval"]
    hit_vals = [m[c]["gold_doc_hit_rate"] for c in strat]
    ax2.bar(strat, hit_vals, width=0.62, color=[COND_COLORS[c] for c in strat])
    for x, v in zip(strat, hit_vals):
        ax2.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=ink)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("fraction of top-k chunks from gold doc")
    ax2.set_title("Retrieval quality by chunking strategy")
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

def main() -> None:
    ap = argparse.ArgumentParser(description="Week 7 Day 2: mini RAG pipeline (chunk -> embed -> retrieve -> stuff)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline chunker/grader checks only (no API calls)")
    ap.add_argument("--smoke", action="store_true",
                    help="truncate each doc to 10k chars (quick sanity pass)")
    ap.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    ap.add_argument("--k", type=int, default=K, help="top-k chunks stuffed per question")
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
          f"strategies: {list(CHUNKERS)}; k={args.k}; chat={CHAT_MODEL}")

    embed_client = get_client()
    chat_client = get_chat_client()
    t0 = time.time()

    # chunk + embed, one pass per strategy
    by_strategy = chunk_docs(docs)
    chunk_stats: dict[str, dict] = {}
    strategy_hats: dict[str, object] = {}
    for name, flat in by_strategy.items():
        lens = [len(t) for _, t in flat]
        E, tokens = embed_texts(embed_client, [t for _, t in flat])
        strategy_hats[name] = normalize_rows(E)
        chunk_stats[name] = {"n_chunks": len(flat),
                             "mean_chars": round(sum(lens) / len(lens), 1),
                             "max_chars": max(lens),
                             "embed_tokens": tokens}
        print(f"  {name:<16} {len(flat):>5} chunks  mean {chunk_stats[name]['mean_chars']:>6.1f} chars  "
              f"{tokens:>6} embed tokens")

    # query embeddings shared across strategies
    Q, q_tokens = embed_texts(embed_client, [q["question"] for q in QUERIES])
    q_hat = normalize_rows(Q)

    conditions = ["no_retrieval"] + list(CHUNKERS)
    runs: list[dict] = []
    for cond in conditions:
        for qi, question in enumerate(QUERIES):
            if cond == "no_retrieval":
                run = run_condition(cond, question, chat_client, q_hat, qi,
                                    [], None)
            else:
                run = run_condition(cond, question, chat_client, q_hat, qi,
                                    by_strategy[cond], strategy_hats[cond])
            runs.append(run)
            mark = "PASS" if run["grade"]["pass"] else "fail"
            print(f"  {cond:<16} {question['id']:<10} {mark}"
                  f"  (prompt {run['usage']['prompt']} tok)")

    print(f"\nelapsed {time.time() - t0:.0f}s  "
          f"(query embed tokens: {q_tokens})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "week07_day2_mini_rag",
        "timestamp": ts,
        "config": {
            "docs": DOCS, "doc_chars": {d: len(t) for d, t in docs.items()},
            "chunkers": {
                "fixed_400": "fixed 400 chars, step 400 (no overlap)",
                "fixed_800": "fixed 800 chars, step 800 (no overlap)",
                "fixed_800_o100": "fixed 800 chars, step 700 (100-char overlap)",
                "para_800": "paragraph-merge, target 800 chars",
            },
            "k": args.k, "smoke": args.smoke,
            "chat": {"model": CHAT_MODEL, "base_url": CHAT_BASE, "temperature": 0.0},
        },
        "runs": runs,
        **summarize(runs, conditions, chunk_stats),
    }
    out_path = _THIS_DIR / f"day2_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"results -> {out_path}")

    print_report(results)

    if not args.no_plot:
        plot_path = _THIS_DIR / f"day2_plot_{ts}.png"
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
