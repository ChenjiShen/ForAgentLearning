"""Week 6 Day 1: Anatomy of context growth — instrument a 20-turn session.

Experiment question:
    Across a 20-turn agent session, how fast does context grow, and where
    does the growth come from — messages or tool results?

Method:
    1. Run ONE long session: 20 sequential user turns, history never reset.
       The prompts form a coherent mini research project (explore sandbox →
       write data files → compute → web research → save notes → recall) so the
       agent actually uses its tools and tool results accumulate.
    2. Instrument every API call:
       - API ground truth: prompt_tokens / completion_tokens from `usage`.
       - Local decomposition of the input prompt into four components:
           system prompt, conversation messages, tool results, tool schemas.
         Component sizes are estimated with a chars/4 token heuristic
         (qwen's tokenizer is not public), then CALIBRATED once on the first
         API call: factor = api_prompt_tokens / estimated_total. All later
         estimates are scaled by that same factor. The per-call residual
         (api_total - calibrated_sum) measures how well the estimator holds
         as content mix changes (JSON vs prose vs code).
    3. Roll up per user turn (a turn = 1..k API calls + tool executions):
       input/output tokens, tool calls, component breakdown, cumulative input.
    4. Plot cumulative context growth stacked by component + per-turn output,
       and print a per-turn table.

    Expected result (the week's lesson): growth is roughly linear per turn,
    and tool results dominate the context — messages are second, schemas are
    a small constant.

Run:
    export DASHSCOPE_API_KEY=sk-...
    python experiments/week06/day1_context_growth.py
    python experiments/week06/day1_context_growth.py --no-plot   # table only

Reuses week05/day1_tool_descriptions.py for the 6-tool schema (condition
"after" descriptions — the current best) and the week03 tool implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WEEK05_DIR = _THIS_DIR.parent / "week05"
if str(_WEEK05_DIR) not in sys.path:
    sys.path.insert(0, str(_WEEK05_DIR))

from day1_tool_descriptions import (  # noqa: E402
    TOOL_FUNCTIONS,
    _estimate_cost,
    build_tools,
)
from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "qwen3.7-plus")
API_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to sandboxed tools. "
    "All file paths and shell commands run inside the sandbox workspace. "
    "Complete the user's task using the available tools.\n\n"
    "IMPORTANT — Tool selection rules:\n"
    "- Read each tool's description carefully before calling it.\n"
    "- Only call tools that are actually needed for the task.\n"
    "- If a task can be done without any tool, answer directly.\n"
    "- When you have the answer, state it clearly and stop calling tools.\n"
    "- This is a long working session: answers should be brief and factual.\n"
    "- When the user says 'no tools needed', answer from the conversation "
    "history without calling any tool."
)

# ============================================================================
# The 20-turn session — a coherent mini research project.
# Turns marked (recall) reference earlier turns, so history genuinely matters.
# ============================================================================

SESSION_PROMPTS: list[dict] = [
    {"id": "T01", "kind": "tool",
     "prompt": "Explore the sandbox: list what files and directories exist."},
    {"id": "T02", "kind": "tool",
     "prompt": ("Create the file research/data1.md containing these numbers, "
                "one per line: 12, 45, 7, 89, 23, 56, 91, 34.")},
    {"id": "T03", "kind": "tool",
     "prompt": "Read research/data1.md back and confirm all 8 numbers are there."},
    {"id": "T04", "kind": "tool",
     "prompt": ("Use a python one-liner to compute the sum and mean of the "
                "numbers in research/data1.md.")},
    {"id": "T05", "kind": "tool",
     "prompt": ("Search the web for 'context window token limits in LLM agents' "
                "and save the top results to research/notes.md.")},
    {"id": "T06", "kind": "tool",
     "prompt": ("Fetch https://en.wikipedia.org/wiki/Context_window and "
                "summarize it in three bullet points.")},
    {"id": "T07", "kind": "tool",
     "prompt": "Append the three bullets from the previous step to research/notes.md."},
    {"id": "T08", "kind": "recall",
     "prompt": ("No tools needed — what was the mean of our numbers, and what "
                "topic did we research first?")},
    {"id": "T09", "kind": "tool",
     "prompt": ("Create research/data2.md with these numbers, one per line: "
                "101, 202, 303, 404, 505.")},
    {"id": "T10", "kind": "tool",
     "prompt": "Use grep to find every line mentioning 'mean' in the research directory."},
    {"id": "T11", "kind": "tool",
     "prompt": ("Write a python script research/analyze.py that reads both data "
                "files and prints their combined count, sum, and mean.")},
    {"id": "T12", "kind": "tool",
     "prompt": "Run research/analyze.py and report what it prints."},
    {"id": "T13", "kind": "tool",
     "prompt": ("Search the web for 'prompt caching LLM API cost savings' and "
                "add the key facts to research/notes.md.")},
    {"id": "T14", "kind": "recall",
     "prompt": ("No tools needed — which two topics have we researched so far, "
                "and what was the combined mean of our two data files?")},
    {"id": "T15", "kind": "tool",
     "prompt": "List the files in the research directory with their sizes."},
    {"id": "T16", "kind": "tool",
     "prompt": ("Fetch https://en.wikipedia.org/wiki/Prompt_engineering and add "
                "two bullet points to research/notes.md.")},
    {"id": "T17", "kind": "tool",
     "prompt": "Use grep (case-insensitive) to find 'cache' mentions in the research directory."},
    {"id": "T18", "kind": "recall",
     "prompt": "No tools needed — in one paragraph, summarize everything we have done in this session."},
    {"id": "T19", "kind": "tool",
     "prompt": "Count the words in research/notes.md."},
    {"id": "T20", "kind": "tool",
     "prompt": ("Write a final summary of our findings to "
                "research/final_summary.md, then read it back to me.")},
]

# ============================================================================
# Token accounting — local component estimation + calibration
# ============================================================================


def est_tokens(text: str) -> int:
    """chars/4 heuristic: the standard pre-tokenizer estimate for English."""
    return max(1, math.ceil(len(text) / 4))


def measure_components(messages: list[dict], tools: list[dict]) -> dict:
    """Estimate input-token components of a request (uncalibrated).

    Returns raw estimates for: system, messages (user+assistant, incl.
    tool_call payloads), tool_results (role=tool), schemas (the tools param).
    """
    comp = {"system": est_tokens(SYSTEM_PROMPT),
            "messages": 0, "tool_results": 0, "schemas": 0}
    for m in messages:
        key = "tool_results" if m["role"] == "tool" else "messages"
        comp[key] += est_tokens(json.dumps(m, ensure_ascii=False))
    comp["schemas"] = est_tokens(json.dumps(tools, ensure_ascii=False))
    return comp


def execute_tool(name: str, args: dict) -> str:
    try:
        return str(TOOL_FUNCTIONS[name](args))
    except Exception as e:  # tool errors go back to the model as content
        return f"[tool error] {name}: {e}"


# ============================================================================
# The instrumented session
# ============================================================================


def run_session(prompts: list[dict], tools: list[dict],
                max_iter_per_turn: int = 10) -> dict:
    # trust_env=False: ignore the macOS system proxy (127.0.0.1:7890 — a local
    # proxy app that may not be running). The Beijing DashScope endpoint is
    # reachable directly; going through a dead proxy breaks every API call.
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=API_BASE,
        http_client=httpx.Client(trust_env=False),
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    calib_factor: float | None = None
    factors_seen: list[float] = []
    calls: list[dict] = []
    turns: list[dict] = []

    for pi, p in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": p["prompt"]})
        turn_calls, tool_names, t0_turn = [], [], time.monotonic()

        for _ in range(max_iter_per_turn):
            t0 = time.monotonic()
            comp = measure_components(messages, tools)
            response = client.chat.completions.create(
                model=MODEL, tools=tools, messages=messages)
            elapsed = time.monotonic() - t0

            usage = response.usage
            api_in = usage.prompt_tokens if usage else None
            api_out = usage.completion_tokens if usage else None

            est_total = sum(comp.values())
            if calib_factor is None and api_in:
                # Calibrate the estimator on the very first API call
                # (system + one user message + schemas — no tool results yet).
                calib_factor = api_in / est_total
            if api_in and est_total:
                factors_seen.append(api_in / est_total)
            f = calib_factor if calib_factor else 1.0
            calib = {k: round(v * f) for k, v in comp.items()}
            residual = (api_in - sum(calib.values())) if api_in else None

            choice = response.choices[0]
            msg = choice.message

            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            call = {
                "call_index": len(calls) + 1,
                "turn": pi,
                "api_prompt_tokens": api_in,
                "api_completion_tokens": api_out,
                "n_tool_calls": len(msg.tool_calls or []),
                "finish_reason": choice.finish_reason,
                "components_est": calib,
                "residual_tokens": residual,
                "elapsed_s": round(elapsed, 2),
            }
            calls.append(call)
            turn_calls.append(call)

            if not msg.tool_calls:
                break  # final answer — turn over

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    out = f"[tool error] invalid JSON arguments: {e}"
                else:
                    out = execute_tool(name, args)
                tool_names.append(name)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out})

        # ---- roll up the turn ------------------------------------------
        in_tok = sum(c["api_prompt_tokens"] or 0 for c in turn_calls)
        out_tok = sum(c["api_completion_tokens"] or 0 for c in turn_calls)
        comp_sum = {k: sum(c["components_est"][k] for c in turn_calls)
                    for k in ("system", "messages", "tool_results", "schemas")}
        turn = {
            "turn": pi,
            "prompt_id": p["id"],
            "kind": p["kind"],
            "prompt": p["prompt"],
            "n_api_calls": len(turn_calls),
            "n_tool_calls": len(tool_names),
            "tools_called": tool_names,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "components": comp_sum,
            "cumulative_input": in_tok + (turns[-1]["cumulative_input"] if turns else 0),
            "elapsed_s": round(time.monotonic() - t0_turn, 2),
            "agent_stopped": bool(p["kind"] == "tool" and not tool_names),
        }
        turns.append(turn)

    # ---- session rollup -----------------------------------------------
    total_in = sum(t["input_tokens"] for t in turns)
    total_out = sum(t["output_tokens"] for t in turns)
    comp_total = {k: sum(t["components"][k] for t in turns)
                  for k in ("system", "messages", "tool_results", "schemas")}
    factors = [f for f in factors_seen if f]
    factor_mean = sum(factors) / len(factors) if factors else None
    factor_sd = (sum((f - factor_mean) ** 2 for f in factors) / len(factors)) ** 0.5 \
        if factors else None
    residuals = [abs(c["residual_tokens"]) for c in calls
                 if c["residual_tokens"] is not None]
    resid_pcts = [abs(c["residual_tokens"]) / c["api_prompt_tokens"] * 100
                  for c in calls
                  if c["residual_tokens"] is not None and c["api_prompt_tokens"]]
    mape = sum(resid_pcts) / len(resid_pcts) if resid_pcts else 0.0

    return {
        "model": MODEL,
        "n_turns": len(turns),
        "n_api_calls": len(calls),
        "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": round(_estimate_cost(total_in, total_out), 4),
        "components_total": comp_total,
        "components_share": {k: round(v / max(sum(comp_total.values()), 1), 4)
                             for k, v in comp_total.items()},
        "calibration": {
            "factor_first_call": round(calib_factor, 4) if calib_factor else None,
            "factor_mean": round(factor_mean, 4) if factor_mean else None,
            "factor_sd": round(factor_sd, 4) if factor_sd else None,
            "mean_abs_residual_percent": round(mape, 3),
        },
        "turns": turns,
        "calls": calls,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# Reporting — table + plot
# ============================================================================


def print_table(s: dict) -> None:
    print("\n=== 20-turn session: token growth per turn ===")
    print(f"model={s['model']}  api_calls={s['n_api_calls']}  "
          f"tool_calls={s['n_tool_calls']}")
    print(f"total input={s['total_input_tokens']:,}  "
          f"output={s['total_output_tokens']:,}  "
          f"est. cost=${s['estimated_cost_usd']:.4f}")
    cal = s["calibration"]
    print(f"estimator calibration: first-call factor={cal['factor_first_call']} "
          f"(mean={cal['factor_mean']} sd={cal['factor_sd']}, "
          f"mean |residual| {cal['mean_abs_residual_percent']}%)")
    print("\n turn  calls tools  in_tok  out_tok  cum_in  "
          "| msgs   tools   schemas (est shares of in_tok)")
    for t in s["turns"]:
        c = t["components"]
        tot = t["input_tokens"] or 1
        print(f" {t['turn']:>4} {t['n_api_calls']:>5} {t['n_tool_calls']:>5} "
              f"{t['input_tokens']:>7,} {t['output_tokens']:>7,} "
              f"{t['cumulative_input']:>7,}  | "
              f"{c['messages']/tot*100:>4.0f}%  {c['tool_results']/tot*100:>4.0f}%  "
              f"{c['schemas']/tot*100:>4.0f}%   [{t['prompt_id']} "
              f"{','.join(t['tools_called']) or 'none'}]")
    ct = s["components_total"]
    print("\nWhere the context came from (cumulative input tokens):")
    print(f"  tool results : {ct['tool_results']:>8,} tokens "
          f"({ct['tool_results']/max(sum(ct.values()),1)*100:.1f}%)")
    print(f"  messages     : {ct['messages']:>8,} tokens "
          f"({ct['messages']/max(sum(ct.values()),1)*100:.1f}%)")
    print(f"  schemas      : {ct['schemas']:>8,} tokens "
          f"({ct['schemas']/max(sum(ct.values()),1)*100:.1f}%)  [constant per call]")
    print(f"  system       : {ct['system']:>8,} tokens "
          f"({ct['system']/max(sum(ct.values()),1)*100:.1f}%)  [constant per call]")


def make_plot(s: dict, out_png: Path) -> None:
    from itertools import accumulate

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # -- dataviz palette (light mode), validated slots 1-3 + sequential blue
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    SECONDARY = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    C_TOOL = "#2a78d6"    # slot 1 blue   — tool results (the headline series)
    C_MSG = "#eb6834"     # slot 2 orange — messages
    C_SCHEMA = "#1baf7a"  # slot 3 aqua   — tool schemas
    C_SYSTEM = "#eda100"  # slot 4 yellow — system prompt
    C_OUT = "#3987e5"     # sequential blue 400 — output bars

    turns = s["turns"]
    x = [t["turn"] for t in turns]
    comps = [t["components"] for t in turns]
    # t["components"] is PER-TURN. The growth chart needs CUMULATIVE sums —
    # context compounds because every API call re-sends the whole history.
    y_system = list(accumulate(c["system"] for c in comps))
    y_schema = list(accumulate(c["schemas"] for c in comps))
    y_msg = list(accumulate(c["messages"] for c in comps))
    y_tool = list(accumulate(c["tool_results"] for c in comps))
    y_out = [t["output_tokens"] for t in turns]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 2.2, 1.0],
                          hspace=0.42, left=0.08, right=0.97,
                          top=0.95, bottom=0.06)

    # ---- KPI strip -----------------------------------------------------
    ax_k = fig.add_subplot(gs[0])
    ax_k.axis("off")
    ct = s["components_total"]
    tool_share = ct["tool_results"] / max(sum(ct.values()), 1) * 100
    kpis = [
        (f"{s['total_input_tokens']:,}", "total input tokens (20 turns)"),
        (f"{tool_share:.0f}%", "of context = tool results"),
        (f"{s['total_output_tokens']:,}", "total output tokens"),
        (f"${s['estimated_cost_usd']:.3f}", "est. API cost"),
    ]
    for i, (val, label) in enumerate(kpis):
        col_x = 0.03 + i * 0.26
        ax_k.text(col_x, 0.55, val, transform=ax_k.transAxes, fontsize=26,
                  fontweight="bold", color=INK, va="center")
        ax_k.text(col_x, 0.02, label, transform=ax_k.transAxes, fontsize=10.5,
                  color=SECONDARY, va="center")

    # ---- Panel A: cumulative context, stacked by component -------------
    ax_a = fig.add_subplot(gs[1])
    # Draw order bottom→top; system and schemas are per-call constants,
    # messages and tool results are what accumulate with history.
    stacks = [y_system, y_schema, y_msg, y_tool]
    labels = ["system", "tool schemas", "messages", "tool results"]
    colors = [C_SYSTEM, C_SCHEMA, C_MSG, C_TOOL]
    base = [0.0] * len(x)
    for vals, col in zip(stacks, colors):
        ax_a.fill_between(x, base, [b + v for b, v in zip(base, vals)],
                          step="mid", color=col, alpha=1.0,
                          edgecolor=SURFACE, linewidth=2, zorder=2)
        base = [b + v for b, v in zip(base, vals)]

    # Direct labels at the end of each band (aqua/yellow need the relief
    # rule). Nudge apart if two band tops land too close at turn 20.
    band_tops = []
    running = 0.0
    for vals in stacks:
        running += vals[-1]
        band_tops.append(running)
    ylim_top = band_tops[-1] * 1.12
    min_gap = 0.085 * ylim_top
    placed = []
    for top in band_tops:
        if placed and top - placed[-1] < min_gap:
            top = placed[-1] + min_gap
        placed.append(top)
    for label, top in zip(labels, placed):
        ax_a.annotate(label, xy=(x[-1], top), xytext=(x[-1] + 0.4, top),
                      fontsize=10.5, color=SECONDARY, va="center",
                      annotation_clip=False)

    ax_a.set_ylabel("cumulative input tokens", fontsize=11)
    ax_a.set_title("Cumulative context growth over a 20-turn session — "
                   "stacked by where the input tokens come from",
                   fontsize=13, loc="left", pad=10)
    ax_a.set_xticks(range(1, 21))
    ax_a.set_xlim(0.6, 21.6)
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax_a.grid(axis="y", color=GRID, linewidth=1)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    for s_ in ax_a.spines.values():
        s_.set_color("#c3c2b7")
    ax_a.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=C_TOOL, label="tool results"),
        plt.Rectangle((0, 0), 1, 1, color=C_MSG, label="messages"),
        plt.Rectangle((0, 0), 1, 1, color=C_SCHEMA, label="tool schemas"),
        plt.Rectangle((0, 0), 1, 1, color=C_SYSTEM, label="system prompt"),
    ], loc="upper left", frameon=False, fontsize=10.5, labelcolor=SECONDARY)

    # ---- Panel B: output tokens per turn -------------------------------
    ax_b = fig.add_subplot(gs[2])
    ax_b.bar(x, y_out, color=C_OUT, width=0.72, zorder=3)
    ax_b.set_ylabel("output tokens", fontsize=11)
    ax_b.set_xlabel("user turn", fontsize=11)
    ax_b.set_title("Output tokens per turn — flat while input compounds",
                   fontsize=13, loc="left", pad=10)
    ax_b.set_xticks(range(1, 21))
    ax_b.set_xlim(0.6, 21.6)
    ax_b.grid(axis="y", color=GRID, linewidth=1)
    ax_b.set_axisbelow(True)
    ax_b.spines[["top", "right"]].set_visible(False)
    for s_ in ax_b.spines.values():
        s_.set_color("#c3c2b7")

    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nPlot saved to {out_png}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 6 Day 1: context growth")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the matplotlib chart (table only)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="max API iterations per user turn")
    parser.add_argument("--plot-only", metavar="RESULTS_JSON",
                        help="regenerate the plot from a saved results JSON "
                             "(no API calls); overwrites the matching PNG name")
    args = parser.parse_args()

    if args.plot_only:
        rp = Path(args.plot_only)
        s = json.loads(rp.read_text())
        out = rp.with_name(rp.stem.replace("results", "context_growth") + ".png")
        make_plot(s, out)
        print_table(s)
        return

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set — export it first "
                 "(e.g. `set -a; source ../../.env; set +a`).")

    print(f"Starting 20-turn instrumented session (model={MODEL}) ...")
    t_start = time.monotonic()
    summary = run_session(SESSION_PROMPTS, build_tools("after"),
                          max_iter_per_turn=args.max_iter)
    wall = time.monotonic() - t_start

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = _THIS_DIR / f"day1_results_{ts}.json"
    summary["wall_seconds"] = round(wall, 1)
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Raw trace saved to {results_path}")

    print_table(summary)
    if not args.no_plot:
        png_path = _THIS_DIR / f"day1_context_growth_{ts}.png"
        make_plot(summary, png_path)
    print(f"\nWall time: {wall:.0f}s")


if __name__ == "__main__":
    main()
