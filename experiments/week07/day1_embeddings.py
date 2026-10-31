"""Week 7 Day 1: Embeddings + top-k similarity search with numpy. No vector DB.

Experiment question (PLAN.md week 7 day 1):
    What are embeddings, and how does top-k similarity search actually work?

Build: embed 50 short texts, implement cosine top-k search in pure numpy,
score it against a labeled corpus. Arrays are enough to learn the math —
a vector DB is an index over this same operation, nothing more.

Design:

    CORPUS — 50 short texts, 10 themes x 5 texts. Texts are harvested from
    this project's own week 1-6 notes/reports (context management, evals,
    tool design, ...), so week 7 day 2's RAG-over-my-notes pipeline can
    reuse them. Each text carries a theme label = ground truth.

    SEARCH — the core lesson, three numpy lines on the whole corpus at once:
        E_hat = E / norm(E, axis=1)      # L2-normalize rows
        sims  = E_hat @ q_hat            # (n,) cosine similarities, one matmul
        idx   = np.argsort(-sims)[:k]    # descending, take k best
    Cosine == dot product of L2-normalized vectors; the matrix multiply is
    O(n*d) over all n texts simultaneously. A vector DB earns its keep only
    when n outgrows one array (millions of vectors -> ANN indexes,
    persistence, sharding) — with n=50, numpy is the honest tool.

    QUERIES — 10 thematic queries (one per theme, gold set = the theme's 5
    texts) + 2 extras: a tricky "vector database" query (gold = memory_rag
    theme) and one Chinese query (gold = evals theme) to demo the
    multilingual embedding space.

    METRICS — per query: hit@k (fraction of the 5 gold texts inside top-k),
    MRR (1/rank of the first gold hit), precision@k. Global: mean intra-
    theme vs inter-theme cosine (do same-theme texts cluster?), and the
    fraction of texts whose nearest neighbor is a same-theme text.

Run:
    export $(grep DASHSCOPE_API_KEY .env | xargs)
    python experiments/week07/day1_embeddings.py --selftest   # offline checks, no API
    python experiments/week07/day1_embeddings.py              # full run, ~6 API calls

Outputs (in experiments/week07/):
    day1_results_{ts}.json  — corpus, queries, per-query top-k, metrics
    day1_embeddings_{ts}.npz — corpus/query embedding matrices (reloadable)
    day1_plot_{ts}.png      — similarity heatmap + per-query top-5 bars
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import httpx
import numpy as np
from openai import OpenAI

MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BATCH = 10  # DashScope embedding batches; chunk inputs to this size

# ---------------------------------------------------------------------------
# Corpus: 50 short texts, 10 themes x 5. Harvested from this repo's notes.
# ---------------------------------------------------------------------------

THEMES = [
    "agent_loop", "tools", "context", "evals", "temperature",
    "memory_rag", "cost_tokens", "mcp", "multiagent", "planning",
]

CORPUS = [
    # --- agent_loop ---------------------------------------------------------
    {"theme": "agent_loop", "text": "ReAct interleaves reasoning traces and task actions in the same loop: think, act, observe, then reason again from the observation."},
    {"theme": "agent_loop", "text": "The chat API is stateless: the entire message array is re-sent on every call. The loop is the agent, not the model."},
    {"theme": "agent_loop", "text": "When a model calls a tool it executes nothing: it emits JSON with a function name and arguments, then stops with finish_reason tool_calls."},
    {"theme": "agent_loop", "text": "ReAct reduces CoT hallucination because every answer has external evidence to check, and the reasoning trace is human-readable and diagnosable."},
    {"theme": "agent_loop", "text": "One loop iteration appends user -> assistant(tool_call) -> tool(result) to history, and the whole sequence becomes input to the next reasoning step."},
    # --- tools --------------------------------------------------------------
    {"theme": "tools", "text": "Tool descriptions are prompt engineering: the description is the triggering condition that decides when the model calls this tool."},
    {"theme": "tools", "text": "Blocklists always lose; allowlists win: run_command only permits executables on an explicit safe list."},
    {"theme": "tools", "text": "A model-supplied path is untrusted input: resolve it and verify it stays inside the sandbox root."},
    {"theme": "tools", "text": "Return errors to the model instead of crashing: a good error message lets the agent retry with corrected arguments."},
    {"theme": "tools", "text": "Tool-result design matters: structured, brief outputs with truncation markers beat dumping raw text into context."},
    # --- context ------------------------------------------------------------
    {"theme": "context", "text": "Growth comes from tool results (45.6% of all input tokens in the week-6 session) and messages (36.9%) - both re-enter every API call."},
    {"theme": "context", "text": "The size of your tool results, not the number of turns, sets the slope of context growth."},
    {"theme": "context", "text": "Pruning replaces old tool results with [cleared] stubs: drop the bulk, keep the decisions."},
    {"theme": "context", "text": "Compaction summarizes the older half of history when it crosses N tokens, then splices the summary back in."},
    {"theme": "context", "text": "Lost in the Middle: even below the token limit, recall degrades for information buried in the middle of long contexts."},
    # --- evals --------------------------------------------------------------
    {"theme": "evals", "text": "A substring checker gives false confidence: an answer containing the digit 3 can pass a task whose true answer is 28."},
    {"theme": "evals", "text": "Programmatic checkers verify artifacts on disk - the filesystem is ground truth that text-only judging is blind to."},
    {"theme": "evals", "text": "An LLM judge handles formatting variance and semantic equivalence, but it is not ground truth either: position bias and self-preference bias exist."},
    {"theme": "evals", "text": "One run means nothing: compute per-task pass rates across at least 5 runs to see variance."},
    {"theme": "evals", "text": "Eval ground truth silently decays against a mutable sandbox - pin fixtures or re-verify them before every run."},
    # --- temperature --------------------------------------------------------
    {"theme": "temperature", "text": "Temperature rescales logits before softmax: low temperature makes the distribution peakier and more greedy."},
    {"theme": "temperature", "text": "top-p truncates the candidate set: it cuts the long tail instead of reshaping the distribution."},
    {"theme": "temperature", "text": "temperature=0 means 'as deterministic as possible', not 'guaranteed deterministic' - providers may switch to argmax, but you cannot assume it."},
    {"theme": "temperature", "text": "The same prompt at three temperatures produces different output variance - and variance affects task success, not just text style."},
    {"theme": "temperature", "text": "Agent success rates vary run to run even at temperature 0, which is why experiments need multiple runs per condition."},
    # --- memory_rag ---------------------------------------------------------
    {"theme": "memory_rag", "text": "An embedding is a point in a vector space: texts with similar meaning land close together."},
    {"theme": "memory_rag", "text": "Cosine similarity between normalized vectors is a dot product - with numpy it is one matrix multiply."},
    {"theme": "memory_rag", "text": "Retrieval injects relevant chunks into the prompt, and chunking strategy dominates retrieval quality."},
    {"theme": "memory_rag", "text": "No vector DB is needed to learn similarity search: 50 vectors fit comfortably in one numpy array."},
    {"theme": "memory_rag", "text": "Retrieval helps when the answer lives in documents the model has never seen; it hurts when it stuffs noise into the context."},
    # --- cost_tokens --------------------------------------------------------
    {"theme": "cost_tokens", "text": "Input tokens dominate agent cost: the week-6 session spent 565k input vs 15k output tokens, a 37:1 ratio."},
    {"theme": "cost_tokens", "text": "Tool schemas are re-sent with every API call - 1,909 tokens per call for 6 tools, pure repeated static content."},
    {"theme": "cost_tokens", "text": "usage.prompt_tokens and usage.completion_tokens are ground truth; accumulate them across iterations for per-run cost."},
    {"theme": "cost_tokens", "text": "Streaming sends deltas as they are generated, so the UI can render tokens before the response finishes."},
    {"theme": "cost_tokens", "text": "Structured output via response_format beats asking politely: schema-constrained JSON validated with pydantic."},
    # --- mcp ----------------------------------------------------------------
    {"theme": "mcp", "text": "MCP separates hosts, clients, and servers, with tools, resources, and prompts as the three server primitives."},
    {"theme": "mcp", "text": "An MCP client just speaks protocol: initialize a session, list_tools, call_tool - no LLM required."},
    {"theme": "mcp", "text": "Bridging MCP to an agent means converting MCP tool listings into provider tool schemas and routing calls through the session."},
    {"theme": "mcp", "text": "stdio transport runs the server as a subprocess; HTTP transport makes it a network service."},
    {"theme": "mcp", "text": "The MCP indirection pays off when one tool set must be shared across many agents and apps."},
    # --- multiagent ---------------------------------------------------------
    {"theme": "multiagent", "text": "The orchestrator-workers pattern: one agent plans, workers execute subtasks, the orchestrator synthesizes their results."},
    {"theme": "multiagent", "text": "Context isolation is the real benefit of subagents - a worker's 50-tool-call search never pollutes the parent's history."},
    {"theme": "multiagent", "text": "Communication cost: what is lost at each handoff is exactly what the parent never sees."},
    {"theme": "multiagent", "text": "Parallel delegation cuts wall-clock when subtasks are independent; dependent subtasks make workers thrash."},
    {"theme": "multiagent", "text": "More agents is usually not the answer: interdependence is where a single agent glides and multi-agent fails."},
    # --- planning -----------------------------------------------------------
    {"theme": "planning", "text": "Plan-then-execute: one call produces a step list, and the executor works through the steps with tools."},
    {"theme": "planning", "text": "Reflexion feeds a failed run back to the agent for one retry - reflection turns failure into a prompt."},
    {"theme": "planning", "text": "Plans go stale mid-execution: the executor must be able to flag 'step impossible' and trigger a replan."},
    {"theme": "planning", "text": "ReAct's failure mode is myopia: without a plan, the agent chases the next local step and loses the goal."},
    {"theme": "planning", "text": "On short tasks planning is overhead; on long horizons it pays for itself."},
]

QUERIES = [
    {"id": "q_agent_loop", "gold": "agent_loop",
     "text": "How does the agent loop connect reasoning and tool use?"},
    {"id": "q_tools", "gold": "tools",
     "text": "How should tool descriptions and safety checks be designed?"},
    {"id": "q_context", "gold": "context",
     "text": "What makes the agent's context grow, and how do pruning and compaction work?"},
    {"id": "q_evals", "gold": "evals",
     "text": "Which checker types exist for grading agent answers, and what are their failure modes?"},
    {"id": "q_temperature", "gold": "temperature",
     "text": "What do temperature and top-p do to sampling?"},
    {"id": "q_memory_rag", "gold": "memory_rag",
     "text": "How do embeddings and cosine similarity power similarity search?"},
    {"id": "q_cost_tokens", "gold": "cost_tokens",
     "text": "Where do the tokens and cost in an agent session come from?"},
    {"id": "q_mcp", "gold": "mcp",
     "text": "What is the MCP protocol architecture and how does a client use it?"},
    {"id": "q_multiagent", "gold": "multiagent",
     "text": "When do orchestrator-worker multi-agent setups help or hurt?"},
    {"id": "q_planning", "gold": "planning",
     "text": "What planning patterns exist for agents, and when is planning worth it?"},
    # extras: tricky query (expects the memory_rag texts about vector DBs)
    # and a Chinese query (multilingual embedding demo)
    {"id": "q_vectordb", "gold": "memory_rag",
     "text": "When should I reach for a vector database instead of plain arrays?"},
    {"id": "q_evals_zh", "gold": "evals",
     "text": "如何衡量评估任务的通过率？"},
]

THEME_COUNT = {t: sum(1 for c in CORPUS if c["theme"] == t) for t in THEMES}


# ---------------------------------------------------------------------------
# Embedding client (week-6 lessons: explicit timeout, trust_env=False so the
# dead macOS system proxy at 127.0.0.1:7890 cannot break API calls).
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        sys.exit("DASHSCOPE_API_KEY not set. Source it first:\n"
                 "  export $(grep DASHSCOPE_API_KEY "
                 "../../.env | xargs)")
    http = httpx.Client(trust_env=False,
                        timeout=httpx.Timeout(300.0, connect=15.0))
    return OpenAI(api_key=key, base_url=BASE_URL, http_client=http)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of texts -> (len(texts), DIM) float array. Batched + retried."""
    rows, usage = [], 0
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        for attempt in range(3):
            try:
                r = client.embeddings.create(
                    model=MODEL, input=chunk, dimensions=DIM)
                rows += [d.embedding for d in r.data]
                usage += r.usage.prompt_tokens
                break
            except Exception as e:  # transient API error -> retry with backoff
                if attempt == 2:
                    raise
                print(f"  retry {attempt + 1} after: {type(e).__name__}: {str(e)[:80]}")
                time.sleep(2 ** attempt)
    return np.asarray(rows, dtype=np.float64), usage


# ---------------------------------------------------------------------------
# The day's lesson: cosine top-k search in pure numpy.
# ---------------------------------------------------------------------------

def normalize_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalize every row so that cos(a, b) == dot(a_hat, b_hat)."""
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def top_k_similarity(E_hat: np.ndarray, q_hat: np.ndarray, k: int):
    """Top-k by cosine similarity, pure numpy.

    E_hat: (n, d) L2-normalized corpus rows. q_hat: (d,) L2-normalized query.
    Returns (idx, sims): indices and cosine similarities of the k best texts.

    One matmul scores ALL n texts at once (O(n*d)); argsort finds the k best
    (O(n log n)). A vector DB exists because n=millions makes exactly this
    loop too slow to run per query — ANN indexes trade recall for speed.
    """
    sims = E_hat @ q_hat                # (n,) cosine similarities
    idx = np.argsort(-sims)[:k]         # descending order, take k best
    return idx, sims[idx]


def cosine_matrix(E_hat: np.ndarray) -> np.ndarray:
    """All pairwise cosine similarities: (n, n), symmetric, diagonal = 1."""
    return E_hat @ E_hat.T


# ---------------------------------------------------------------------------
# Evaluation against the theme labels (ground truth).
# ---------------------------------------------------------------------------

def evaluate(E_hat: np.ndarray, q_hat: np.ndarray, k: int,
             themes: list[str], queries: list[dict]) -> dict:
    """Score each query against its gold theme: hit@k, MRR, precision@k.

    Also the two global 'clumping' numbers: mean intra-theme vs inter-theme
    cosine, and the fraction of texts whose nearest neighbor (excluding
    itself) belongs to the same theme.
    """
    theme_of = themes
    per_query = []
    for qi, q in enumerate(queries):
        idx, sims = top_k_similarity(E_hat, q_hat[qi], k)
        top_ids = [f"{theme_of[i]}_{i:02d}" for i in idx]
        is_gold = np.array([theme_of[i] == q["gold"] for i in idx])
        ranks = [int(r) + 1 for r in np.where(
            np.array([theme_of[i] == q["gold"] for i in idx]))[0]]
        hit = int(is_gold.sum())
        per_query.append({
            "id": q["id"], "gold_theme": q["gold"],
            "top_ids": top_ids,
            "top_sims": [round(float(s), 4) for s in sims],
            "hit_at_k": hit,
            "precision_at_k": hit / k,
            "mrr": round(1.0 / ranks[0], 4) if ranks else 0.0,
        })

    # clumping: do same-theme texts actually sit close together?
    S = cosine_matrix(E_hat)
    theme_idx = np.array([sorted(set(themes)).index(t) for t in theme_of])
    same = theme_idx[:, None] == theme_idx[None, :]
    np.fill_diagonal(same, False)  # exclude self-similarity
    intra = float(S[same].mean()) if same.any() else float("nan")
    inter = float(S[~same].mean())
    # nearest neighbor per text (excluding itself): same theme?
    nn = np.argmax(S - np.eye(len(S)) * 1e9, axis=1)
    nn_same = float((theme_idx[nn] == theme_idx).mean())

    hits = [q["hit_at_k"] for q in per_query]
    return {
        "k": k,
        "per_query": per_query,
        "mean_hit_at_k": round(sum(hits) / len(hits), 3),
        "mean_mrr": round(sum(q["mrr"] for q in per_query) / len(per_query), 3),
        "mean_precision_at_k": round(sum(q["precision_at_k"] for q in per_query) / len(per_query), 3),
        "intra_theme_mean_cosine": round(intra, 4),
        "inter_theme_mean_cosine": round(inter, 4),
        "nn_same_theme_fraction": round(nn_same, 4),
    }


def print_report(metrics: dict) -> None:
    print(f"\ntop-k = {metrics['k']}")
    print(f"{'query':<16} {'gold':<14} {'hit@k':>6} {'MRR':>6}  top-5 retrieved (gold = *)")
    for q in metrics["per_query"]:
        tops = "  ".join(
            f"{'*' if t.startswith(q['gold_theme']) else ' '}{t}"
            for t in q["top_ids"])
        print(f"{q['id']:<16} {q['gold_theme']:<14} {q['hit_at_k']:>4}/{metrics['k']}"
              f" {q['mrr']:>6.3f}  {tops}")
    print(f"\nmean hit@{metrics['k']}: {metrics['mean_hit_at_k']}  "
          f"mean MRR: {metrics['mean_mrr']}  "
          f"mean P@{metrics['k']}: {metrics['mean_precision_at_k']}")
    print(f"intra-theme mean cosine: {metrics['intra_theme_mean_cosine']}  "
          f"inter-theme mean cosine: {metrics['inter_theme_mean_cosine']}  "
          f"(gap = {metrics['intra_theme_mean_cosine'] - metrics['inter_theme_mean_cosine']:+.3f})")
    print(f"nearest-neighbor same-theme fraction: {metrics['nn_same_theme_fraction']}")


# ---------------------------------------------------------------------------
# Plot: (a) 50x50 cosine heatmap with theme blocks, (b) per-query top-5 bars.
# ---------------------------------------------------------------------------

def plot_results(E_hat: np.ndarray, q_hat: np.ndarray, metrics: dict,
                 out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    S = cosine_matrix(E_hat)
    theme_of = np.array([c["theme"] for c in CORPUS])
    order = [i for t in THEMES for i in range(len(CORPUS)) if theme_of[i] == t]
    St = S[np.ix_(order, order)]
    bounds = np.cumsum([THEME_COUNT[t] for t in THEMES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) heatmap: single-hue sequential ramp (magnitude job -> one hue)
    im = ax1.imshow(St, cmap="Blues", vmin=-0.05, vmax=1.0, aspect="auto")
    for b in bounds[:-1]:
        ax1.axhline(b - 0.5, color="white", lw=1.2)
        ax1.axvline(b - 0.5, color="white", lw=1.2)
    centers = [sum(THEME_COUNT[t] for t in THEMES[:i]) + THEME_COUNT[t] / 2
               for i, t in enumerate(THEMES)]
    ax1.set_xticks(centers)
    ax1.set_xticklabels(THEMES, rotation=30, ha="right", fontsize=8)
    ax1.set_yticks(centers)
    ax1.set_yticklabels(THEMES, fontsize=8)
    ax1.set_title("Corpus x corpus cosine similarity (blocks = themes)")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="cosine similarity")

    # (b) top-5 bars per query: gold hits in accent, others muted gray
    gold_color, other_color = "#2563EB", "#9CA3AF"
    nq = len(metrics["per_query"])
    for row, q in enumerate(metrics["per_query"]):
        for rank, (tid, s) in enumerate(zip(q["top_ids"], q["top_sims"])):
            is_gold = tid.startswith(q["gold_theme"])
            ax2.barh(nq - row - 0.5 + (4 - rank) * 0.16, s, height=0.15,
                     color=gold_color if is_gold else other_color)
    ax2.set_yticks([nq - r - 1 + 0.32 for r in range(nq)])
    ax2.set_yticklabels([q["id"] for q in metrics["per_query"]], fontsize=8)
    ax2.set_xlabel("cosine similarity")
    ax2.set_xlim(0, 1.0)
    ax2.set_title(f"Top-5 per query (k={metrics['k']})")
    ax2.legend(handles=[
        Line2D([0], [0], color=gold_color, lw=6, label="gold theme hit"),
        Line2D([0], [0], color=other_color, lw=6, label="other theme"),
    ], loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nplot -> {out_path}")


# ---------------------------------------------------------------------------
# Offline selftest: prove the numpy math against a naive loop, no API.
# ---------------------------------------------------------------------------

def selftest() -> None:
    rng = np.random.default_rng(7)
    checks = []

    def check(name: str, cond: bool) -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. normalize_rows -> unit-norm rows
    E = rng.normal(size=(30, 64))
    E_hat = normalize_rows(E)
    check("normalize_rows: all rows unit norm",
          np.allclose(np.linalg.norm(E_hat, axis=1), 1.0, atol=1e-12))

    # 2. top_k_similarity matches a naive loop over the cosine formula
    q = normalize_rows(rng.normal(size=(1, 64)))[0]
    idx, sims = top_k_similarity(E_hat, q, k=5)
    naive = np.array([np.dot(E_hat[i], q) /
                      (np.linalg.norm(E_hat[i]) * np.linalg.norm(q))
                      for i in range(len(E_hat))])
    check("top_k: indices match naive cosine loop",
          np.array_equal(idx, np.argsort(-naive)[:5]))
    check("top_k: scores match naive cosine loop",
          np.allclose(sims, naive[idx], atol=1e-12))
    check("top_k: returns exactly k distinct indices",
          len(idx) == 5 and len(set(idx.tolist())) == 5)
    check("top_k: sorted descending", np.all(np.diff(sims) <= 0))

    # 3. cosine identity: dot of normalized == cosine formula
    a, b = normalize_rows(rng.normal(size=(2, 64)))
    check("cos(a,b) == dot(a_hat, b_hat)",
          np.isclose(np.dot(a, b), np.dot(a, b) /
                     (np.linalg.norm(a) * np.linalg.norm(b)), atol=1e-12))

    # 4. evaluate() on a synthetic corpus where themes are near-duplicates
    n_themes, per = 3, 5
    synth = []
    for t in range(n_themes):
        center = normalize_rows(rng.normal(size=(1, 64)))[0]
        for _ in range(per):
            synth.append(center + 0.02 * rng.normal(size=64))
    synth_hat = normalize_rows(np.array(synth))
    synth_themes = [f"t{t}" for t in range(n_themes) for _ in range(per)]
    synth_queries = [{"id": f"s{t}", "gold": f"t{t}", "text": ""}
                     for t in range(n_themes)]
    # one query per theme: the first text of each theme block
    m = evaluate(synth_hat, synth_hat[[t * per for t in range(n_themes)]],
                 5, synth_themes, synth_queries)
    check("evaluate: perfect synthetic corpus -> hit@5 = 1.0 everywhere",
          all(q["hit_at_k"] == 5 for q in m["per_query"]))
    check("evaluate: intra-theme cosine > inter-theme cosine",
          m["intra_theme_mean_cosine"] > m["inter_theme_mean_cosine"])
    check("evaluate: nn-same-theme fraction = 1.0 on synthetic corpus",
          m["nn_same_theme_fraction"] == 1.0)

    # 5. edge: k larger than corpus -> returns everything, no crash
    idx_all, _ = top_k_similarity(E_hat, q, k=100)
    check("top_k: k > n returns all n indices", len(idx_all) == len(E_hat))

    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Week 7 Day 1: embeddings + numpy top-k search")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline numpy checks only (no API calls)")
    ap.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    ap.add_argument("--k", type=int, default=5, help="top-k for search and scoring")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    print(f"corpus: {len(CORPUS)} texts, {len(THEMES)} themes x "
          f"{len(CORPUS) // len(THEMES)};  queries: {len(QUERIES)} "
          f"(k={args.k});  model: {MODEL} dim={DIM} base={BASE_URL}")

    client = get_client()
    t0 = time.time()
    print("embedding corpus ...")
    E, corpus_tokens = embed_texts(client, [c["text"] for c in CORPUS])
    print("embedding queries ...")
    Q, query_tokens = embed_texts(client, [q["text"] for q in QUERIES])
    print(f"embedded in {time.time() - t0:.0f}s; "
          f"tokens: {corpus_tokens} corpus + {query_tokens} queries "
          f"= {corpus_tokens + query_tokens} total")
    E_hat, Q_hat = normalize_rows(E), normalize_rows(Q)

    metrics = evaluate(E_hat, Q_hat, k=args.k,
                       themes=[c["theme"] for c in CORPUS], queries=QUERIES)
    print_report(metrics)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(out_dir, f"day1_results_{ts}.json")
    npz_path = os.path.join(out_dir, f"day1_embeddings_{ts}.npz")

    results = {
        "experiment": "week07_day1_embeddings",
        "timestamp": ts,
        "embedding": {"model": MODEL, "dim": DIM, "base_url": BASE_URL,
                      "embedding_tokens": corpus_tokens + query_tokens},
        "corpus": [{"id": f"{c['theme']}_{i:02d}", **c}
                   for i, c in enumerate(CORPUS)],
        "queries": QUERIES,
        "metrics": metrics,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.savez(npz_path, E=E, Q=Q,
             corpus_ids=[f"{c['theme']}_{i:02d}" for i, c in enumerate(CORPUS)],
             query_ids=[q["id"] for q in QUERIES])
    print(f"results -> {results_path}")
    print(f"embeddings -> {npz_path}")

    if not args.no_plot:
        plot_path = os.path.join(out_dir, f"day1_plot_{ts}.png")
        plot_results(E_hat, Q_hat, metrics, plot_path)


if __name__ == "__main__":
    main()
