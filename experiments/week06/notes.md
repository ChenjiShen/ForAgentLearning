# Week 6 Day 1 — Anatomy of context growth (20-turn session)

Run: `python experiments/week06/day1_context_growth.py`
Results: `day1_results_20260817_205740.json` (clean run, raw per-call trace)
Plot: `day1_context_growth_20260817_205740.png` (cumulative stacked area —
regenerate from a saved JSON with
`python day1_context_growth.py --plot-only <results.json>`)
Prior (degraded) run kept for comparison: `day1_results_20260817_084524.json` + `_084524` plot/log.

Note (2026-08-18): v1 of the plot accidentally stacked the PER-TURN component
values while the axis said "cumulative" — the bands wiggled instead of rising.
Fixed to cumulative sums (`itertools.accumulate`) and re-rendered from the
saved JSONs; per-turn values are still in the JSONs.

## Method

One long session, 20 user turns, history never reset. Prompts form a mini
research project (sandbox exploration → data files → python compute → web
research → notes → history recall), reusing the week05 6-tool agent
(condition-B descriptions) and week03 tool implementations.

Instrumentation, per API call:

- **Ground truth** from `usage`: `prompt_tokens` / `completion_tokens`.
- **Component decomposition** (where the input tokens live):
  `system` | `messages` (user + assistant, incl. tool_call payloads) |
  `tool_results` (role=tool) | `schemas` (the tools param).
  Component sizes use a chars/4 token estimate, calibrated ONCE on the first
  API call: factor = api_prompt / estimated_total ≈ **0.99**.

Estimator quality (clean run): mean per-call residual 2.2% of prompt tokens;
component sum ≈ 552,763 vs API ground truth 565,076. For a private
tokenizer (qwen), chars/4 + one calibration point is a solid cheap
instrumentation strategy.

## Results (clean run — proxy up, web tools working)

47 API calls, 30 tool calls, 322s wall time, $0.24 est. cost
(565,076 input + 14,929 output tokens).

| Source | Cumulative input tokens | Share |
|---|---|---|
| tool results | 251,890 | 45.6% |
| messages | 203,771 | 36.9% |
| tool schemas | 89,723 | 16.2% |
| system prompt | 7,379 | 1.3% |

- **Growth is linear**: ~29,050 tokens added per user turn.
- **Answer to the day's question**: growth comes from **tool results**
  (45.6%, the single largest component) and **messages** (36.9%) — both
  re-enter every API call's prompt. Schemas are a constant 1,909 tokens ×
  call (16.2% and falling as a share); system prompt is negligible (1.3%).
- Output tokens stay small and flat relative to input (avg ~318/turn).
- The agent completed all 20 turns correctly this time, including every
  multi-tool turn (search+save, fetch+append). Only quirk: turn 1 called
  `list_dir` 4× redundantly — 4× schema + message cost for zero new info.

## The degraded run as an accidental ablation

Run `_084524` (proxy app off → all web tools returned connection errors):

| Metric | Degraded | Clean | Δ |
|---|---|---|---|
| total input tokens | 319,827 | 565,076 | +77% |
| tool-result share | 35.9% | 45.6% | +9.7pp |
| schema share | 27.1% | 16.2% | −10.9pp |
| growth rate (tokens/turn) | 16,149 | 29,050 | +80% |
| web_search retries on T05 | 4 | 1 | — |

Same agent, same prompts — the only difference is whether tool results are
rich content or error strings. With thin tool results, history compounds
slower and the constant schema cost looms larger (27% of all input). With
healthy tool results, tool results dominate and the session costs 77% more
context. Lesson: **the size of your tool results, not the number of turns,
sets the slope of context growth** — the exact lever day 2 (pruning) and
day 3 (compaction) attack.

---

# Week 6 Day 2 — clear-old-tool-results (pruning)

Run: `python experiments/week06/day2_clear_old_tool_results.py --selftest`
(first), then `--n 3` / `--n 6` for the live 20-turn session.
Results: `day2_results_n3_20260818_210039.json`,
`day2_results_n6_20260818_210653.json` (raw per-call traces).
Same 20-turn session as day 1 (clean-regime: proxy app running), so the
day-1 clean run (`day1_results_20260817_205740.json`, 565,076 input tokens)
is the direct unpruned baseline.

## Implementation

`clear_old_tool_results(messages, msg_turns, current_turn, n)` — a ~20-line
pure function. At each turn boundary (before any API call), every
`role="tool"` message from turn t with `current_turn - t >= n` gets
`content = "[cleared]"`. Nothing else is touched:

- assistant/user messages (the **decisions**) survive verbatim — that's the
  point: drop the bulk, keep the conclusions;
- cleared messages keep `role` and `tool_call_id`, so requests stay
  API-valid — the model sees "[cleared]" stubs, not malformed history;
- results from the last n−1 turns plus the current turn are never pruned.

`--selftest` covers the contract with 7 deterministic checks (no API):
recent results kept / old ones cleared, decisions untouched, structure
preserved, idempotency, n=1 edge, n > session length, n < 1 rejected.

## Verification — does the agent still function?

Both conditions pass every check:

| Check | n=3 | n=6 |
|---|---|---|
| all 20 turns completed | PASS | PASS |
| every tool turn used tools | PASS | PASS |
| fresh (current-turn) results never pruned | PASS | PASS |
| T08 recall (mean 44.62, first topic) | PASS | PASS |
| T14 recall (2 topics, combined mean 144.00) | PASS | PASS |
| T18 recall (session summary) | PASS | PASS |

The recall turns were answered with correct numbers and topics — their facts
live in assistant messages, which pruning keeps. At the end of the n=3 run
the model was looking at **29 "[cleared]" stubs** in its prompt and still
worked normally. Nothing "died" in this session because nothing was asked
about raw tool-result content after its window expired — day 4 will test
that failure mode deliberately.

## Token savings vs unpruned baseline

| Metric | baseline | n=3 | n=6 |
|---|---|---|---|
| input tokens | 565,076 | 437,658 | 552,774 |
| savings | — | **127,418 (22.6%)** | 12,302 (2.2%) |
| est. cost | $0.2439 | **$0.1945** | $0.2412 |
| tool calls | 30 | 32 | 36 |
| cleared events (chars) | — | 29 (43,745) | 27 (42,089) |

- **n=3 works as advertised**: tool behavior nearly identical to baseline
  (same turns, same tools), so the 22.6% is almost pure pruning. Savings
  grow steadily once the window fills (turn 7+), reaching ~127k tokens by
  turn 20.
- **n=6 barely saved anything** — and not because pruning failed. Two
  effects: (a) the wider window keeps 5 turns of results in flight, so at
  session end results from turns 15–20 were never cleared; (b) run-to-run
  behavior variance — the n=6 agent made 6 extra tool calls (turn 4 ran the
  python compute 4× vs baseline's 2×, turn 13 fetched an extra page) *before
  any pruning had even kicked in*, and each extra call re-sends the whole
  history plus schemas.
- **Lesson for day 6**: pruning savings are real but not monotonic in N, and
  a single run's savings can be swamped by stochastic tool-call variance.
  The honest experiment needs ≥5 runs per condition; and the *mechanism*
  (kept decisions, cleared bulk) is better isolated with per-turn
  instrumentation than with a single total.
- Trivial detail: the stubs themselves cost ~3 tokens each (~29 stubs ≈
  <100 tokens in the final prompt) — the replacement string is negligible.

---

# Week 6 Day 3 — LLM-summary compaction (history > N → summarize older half)

Run: `python experiments/week06/day3_compaction.py --selftest` (first), then
`--max-tokens 12000` for the live 20-turn session.
Results: `day3_results_n12000_20260819_215923.json` (raw per-call traces +
compaction events + summary texts).
Plot: `day3_compaction_20260819_215923.png` (regenerate from a saved JSON
with `--plot-only <results.json>`).
Same 20-turn session as days 1-2 (clean-regime: proxy app running, web tools
working), so the day-1 clean run (565,076 input tokens) is the baseline.

## Implementation

`compact_history(messages, msg_turns, summarizer, max_tokens, calib)` — the
summarizer is injected, so the whole mechanism is selftest-able without an
API (9 deterministic checks). Mechanics:

- **Trigger**: calibrated estimate of system+messages > max_tokens, checked
  at every turn boundary BEFORE any API call.
- **Split**: the older half is messages[1:split] by message count; the split
  walks forward past any role="tool" messages so it never cuts between an
  assistant tool_call and its results (orphan tool results are API-invalid).
- **Guards**: older half needs ≥4 messages; compaction is skipped when the
  kept half alone already exceeds N (one summary cannot fix that — it would
  thrash every turn).
- **Summarizer**: the older half is serialized to text (roles + content +
  tool-call args flattened) and sent as ONE plain chat completion with a
  system prompt that insists on preserving numbers, filenames, facts,
  decisions ("any detail you drop is lost forever"). Output is spliced in as
  a single role="user" message with a `[Earlier conversation summary]`
  prefix. Its input/output tokens count against the session totals and cost.
- `--selftest` covers: below/above threshold, exact half-splitting with the
  kept half untouched, tool-pair straddling the midpoint (walk-forward),
  thrash guard, MIN_OLD_MSGS guard, max_tokens<1 rejected, and
  summary-of-summary (a later compaction absorbs the previous summary).

## Results (N = 12,000 tokens)

| Metric | baseline | compaction N=12k | Δ |
|---|---|---|---|
| agent input tokens | 565,076 | 509,167 | −55,909 (−9.9%) |
| summarizer tokens | — | 18,244 in + 9,781 out | +28,025 |
| **NET input tokens** | 565,076 | 527,411 | **−37,665 (−6.7%)** |
| est. cost | $0.2439 | $0.2430 | −$0.0009 |
| API calls | 47 | 50 | +3 |
| tool calls | 30 | 32 | +2 |
| wall time | 322s | 524s | +202s |

3 compactions fired:

| # | turn | est → est | absorbed | chars | summarizer tok |
|---|---|---|---|---|---|
| 1 | 14 | 16,164 → 9,402 | turns 1-6 (37 msgs) | 29,543 → 2,175 | 7,593 + 3,261 |
| 2 | 17 | 15,672 → 11,566 | turns 6-12 (27 msgs) | 18,277 → 1,631 | 4,676 + 2,947 |
| 3 | 18 | 12,242 → 7,380 | turns 12-15 (15 msgs) | 22,088 → 2,372 | 5,975 + 3,573 |

The boundary-history sawtooth is exactly the advertised mechanism: history
grows ~linear → crosses N → the older half collapses into a summary (~93%
smaller in chars) → grows again. By turn 20 the ENTIRE turns-1-15 history
had been folded into ONE summary message (2,372 chars) in the prompt —
compactions #1 and #2 were themselves absorbed by #3 (summary-of-summary
happens automatically and the API stayed valid throughout).

## Verification — does the agent still function, and what survived?

All function checks PASS:

| Check | Result |
|---|---|
| all 20 turns completed | PASS |
| every tool turn used tools | PASS |
| T08 recall (mean 44.62, first topic) | PASS |
| T14 recall (2 topics, combined mean 144.0) | PASS |
| T18 recall (session summary) | PASS |
| every compaction ok | PASS |

The honest test is T14 and T18: **both were answered AFTER compactions had
already absorbed the turns their facts came from** (T14 answered with turns
1-6 summarized; T18 with turns 1-15 summarized). All 8 key facts — the two
means (44.6, 144.0), both research topics, data1/data2 filenames, sums 357
and 1872 — are present verbatim in the summary texts. With the
preserve-everything summarizer prompt, *numbers, filenames and topics
survive*; what the summaries visibly dropped is mostly prose and
environment noise.

## Honest caveats

- **Savings arrived late.** Cumulative input was ABOVE baseline until turn
  18 (negative saved_cum through turn 18) — this run's agent made 3 extra
  API calls / 2 extra tool calls before any compaction fired (turn 4: 5
  calls, turn 13: 4 calls vs baseline's 2), and compaction only triggers
  once history crosses 12k. Same lesson as day 2's n=6: run-to-run variance
  can swamp a single run's savings. Day 6's ≥5-runs-per-condition
  experiment is where this gets settled.
- **Cost is flat at this N.** The summarizer's 28k tokens eat most of the
  56k input savings. Smaller N would save more but risk recall failures —
  exactly what day 4 measures (tasks that REQUIRE early-context details).
- Compaction is turn-boundary only: a single turn's huge tool results
  (e.g. turn 13's 63k-token turn) still re-enter every call of that turn.

---

# Week 6 Day 4 — Compaction failure modes: 3 probes on early-context details

Run: `python experiments/week06/day4_compaction_failure_modes.py --selftest`
(first), then `--runs 5` for both conditions.
Results: `day4_results_20260820_195140.json` (5 compact + 5 control runs,
per-run traces + probe scoring). Smoke/validation run:
`day4_results_20260820_184949.json` (control, used to validate the plants).
NOTE: the first batch launch hung ~1h on a stalled DashScope request — the
client had no timeout. Fixed with `httpx.Timeout(300s, connect=15s)`; the
endpoint itself was healthy 5 min later. Lesson logged for day 6.

## Design — 12-turn session, plants → filler → probes

- **Plants (T1-3)**, each detail kept OUT of the user prompts so only the
  agent's actions/history carry it:
  - T01 plants the CONSTRAINT ("every new file → research/, name starts with
    d4_") and generates 30 numbers via a python script: 29 formula values +
    ONE final line with a RANDOM 5-digit number (value exists only in tool
    results/files).
  - T02 reads the file and echoes the sorted list → all 30 numbers in history.
  - T03 creates a log file named d4_log_<YYYY_MM_DD_HH_MM_SS> — the
    timestamp is INVENTED BY THE MODEL, so the exact filename exists only in
    tool-call args/results.
- **Filler (T4-9)**: list, web_search (bulk), wikipedia fetch (bulk), a
  compute script, grep, session notes — history crosses N=8,000 around T7.
- **Probes (T10-12)**: T10 NUMBER ("final line of d4_numbers.md?", no
  tools), T11 FILENAME ("exact name incl. timestamp?", no tools), T12
  CONSTRAINT ("create a NEW file in research/ with your final notes" — no
  prefix hint). Ground truth comes from the SANDBOX FILESYSTEM, not from
  planted text. A probe is void if the agent used tools on a no-tools turn;
  in the compact condition it is *genuine* only if a compaction absorbed
  turns ≥3 before the probe fired.

## Results — 5 runs per condition

| Probe | compact (N=8k) | control | verdict |
|---|---|---|---|
| number (random 5-digit) | **5/5** | 5/5 | survives every summary verbatim |
| filename (invented timestamp) | 4/5 raw, **4/4 plant-success** | 3/5 raw, **3/3 plant-success** | survives; the 3 raw misses were plant failures, not memory |
| constraint (d4_ prefix) | **5/5** | 5/5 | every run created research/d4_final_notes.md |

14/15 probes in the genuine set (plants absorbed by compaction before the
probe) passed. The ONE genuine miss is the plant-failed filename run: the
log file was never created (sandbox rejected the `.txt`-less name), and the
filesystem ground truth refuted the agent's false "Done. Created" claim —
which compaction had preserved faithfully. Verified for the number probe:
the random value appears verbatim in all 10 compaction summaries, never
leaked into the kept half via later turns, and T10 never used tools — the
answers genuinely came from the summaries.

## What actually failed — and it wasn't the summaries

The 3 filename "misses" (1 compact, 2 control) are the same story: at T03
the agent tried to write the log file WITHOUT the `.txt` extension, the
sandbox `write_file` tool REJECTED it (extension whitelist), and the agent
replied "Done. Created research/d4_log_..." anyway. The file never existed
(truth=None from the filesystem) — yet at T11 all 3 runs confidently
recalled the nonexistent name, and compaction faithfully preserved the
FALSE claim (compact run 1: plants absorbed, and the model reproduced the
fictional filename exactly).

This is the day-3 taxonomy in action, but on the AGENT side, not the
compressor side: 「结论比依据容易保存」 — the conclusion ("I created X")
survived in the assistant message; the evidence (the tool rejection) died —
in BOTH conditions, because the agent's own message outlived the tool
result. Compaction preserved the lie verbatim.

## Cost/savings vs control

| Metric | compact (N=8k) | control | Δ |
|---|---|---|---|
| agent input, mean/run | 222,252 | 277,922 | **−20.0%** |
| summarizer tokens (total) | 42,946 in + 29,739 out | — | +72,685 |
| total tokens (5 runs) | 1,183,949 | 1,430,025 | **−17.2%** |
| est. cost (5 runs) | $0.545 | $0.604 | −9.8% |
| wall time (5 runs) | 25 min | 27 min | −2 min |

Compaction saves real money even including its own cost, and at this N the
probes show it loses nothing we could measure. (Caveat: T12's constraint is
partly reinforced by the d4_ pattern in the FILLER prompts' user messages —
T5/T6/T7/T9 all name d4_ files — so "constraint survived the summary"
cannot be separated from "pattern inferred from recent context" in this
design. Both conditions share the confound, so the comparison stays fair.)

## Lessons

1. **Zero measured compaction deaths with a preserve-everything summarizer.**
   Numbers buried in tool results, model-invented filenames, and early user
   constraints all survived 9 compactions across 5 runs. The failure modes
   the taxonomy predicts (critical-detail omission, constraint forgetting)
   need either a weaker summarizer prompt or deliberately worse designs
   (longer lists, more exotic details) to materialize — or a different
   summarizer model. Day 6's 3-way comparison should keep this probe
   battery and add a LAZY summarizer prompt variant if it wants deaths.
2. **The agent's false-success echo is the real information hazard.** Tool
   rejection + "Done." reply = a false fact that verbatim history AND
   summaries both preserve perfectly. Checking tool-result status before
   believing assistant claims is the fix (a checker, not a compression
   technique).
3. **Consecutive-turn compaction churn exists.** compact run 2 compacted at
   T7, T8 and T9 (est 9.3k → 8.0k → 8.1k → 3.2k) — each boundary re-crossed
   N because the kept half plus fresh turns stayed near the threshold. Not
   a correctness bug (the thrash guard only skips when ONE summary cannot
   fix it), but it's 2 extra summarizer calls for little gain — a min-interval
   (e.g. don't re-compact within K turns) would fix it.
4. **Environment:** the DashScope endpoint can stall a request indefinitely
   (~1h observed). Every production-ish harness needs a client timeout;
   with 300s/15s the whole 10-run batch completed without a single retry
   stall.

---

# Day 5 — System prompt restructure: role / tool guidance / constraints / output format

## What changed

The agent's system prompt (a flat "helpful assistant + tool rules" block that
had accreted session-specific lines — "this is a long working session…",
"when the user says 'no tools needed'…") was restructured into four explicit
sections: `# Role`, `# Tool guidance` (incl. a direct-tool mapping),
`# Constraints` (incl. "report only what the tools actually returned" — a
response to day 4's false-success-echo finding), `# Output format`. The two
session-specific lines moved OUT of the system layer (they belong in user
messages of the turns that need them). Old 159 est. tokens -> new 271
(Δ +112/request). Script: `day5_prompt_ab.py`; the two prompts are stored
verbatim inside every results JSON.

## Design

Two conditions (old vs new), ONE thing varied. Same loop, same 6 tools, same
sandbox, same checkers. 240 cells (task x condition x run), order shuffled
with a fixed seed so time-of-day effects can't confound (week-5 day-6
lesson). 5 runs per cell. Two suites:

- **Suite A** — the compounding `evals/tasks.json` (14 tasks, root harness
  substring checkers).
- **Suite B** — week-5's 10-prompt tool-selection set (should-call /
  wrong-tool / exact-match vs. expected tool sets).

## Results — a clean null

| | old | new | Fisher p |
|---|---|---|---|
| Suite A, full 14 tasks | 34/66 (51.5%) | 33/66 (50.0%) | 1.00 |
| Suite A, valid subset (6 tasks) | 30/30 (100%) | 30/30 (100%) | 1.00 |
| Suite B exact-match | 35/50 (70.0%) | 33/49 (67.3%) | 0.83 |
| Suite B should-call / wrong-tool | 100% / 15.3% | 100% / 16.7% | — |
| median answer length | 107 chars | 118 chars | — |
| median tool calls | 2 | 2 | — |
| total input tokens | 1,537,988 | 1,483,913 | — |
| est. cost (all 240 runs) | $0.68 | $0.66 | — |

Errors symmetric: 4 old + 4 new max_turns_exceeded (all on the
MOCK_ORDERS-drifted orders tasks), 1 new APITimeout (recorded, excluded
from rates). Prompt structure moved NOTHING: both prompts are perfect on
every task whose environment still matches, and equally wrong-tooled
(~16%) on the tool-selection suite. The "keep replies brief" output-format
line did not bind — new answers were slightly LONGER, not shorter.

## What the suite itself says (more important than the prompt)

1. **8 of 14 suite-A tasks have drifted ground truth** — requirements.txt
   and MOCK_ORDERS don't exist in the week03 sandbox, notes.txt was
   rewritten by a later experiment (2 lines now, first line changed),
   sandbox root holds 28 .py files (task expects 3). Both conditions see
   the same environment, so the A/B stays fair — but the compounding
   harness's ground truth silently decays against a mutable sandbox. Fix
   for day 7: pin per-week ground-truth snapshots or protect the sandbox
   fixtures. (Selftest re-verifies each drift claim against the live
   filesystem before every run.)
2. **sandbox-list-py "passes" are substring artifacts**: agents answered
   30/35 (wrong counts containing the digit "3") — 7 of 10 runs "passed"
   a task whose true answer is 28. Another argument for exact checkers.
3. **P5-fetch-url is now environment-blocked**: web_fetch rejects
   example.com because it resolves to a private/internal IP (week-3 SSRF
   guard). The agent falls back to web_search (scored "extra") — old 3/5,
   new 1/5, n=5 noise either way, but the failure signature is new since
   week 5.

## Lessons

1. **Structure alone is not content.** The old prompt's tool rules were
   already good enough (week 5 showed descriptions dominate tool
   selection, not the system block around them); re-shelving the same
   facts into sections changes nothing measurable. The one genuinely new
   instruction (report-only-what-tools-returned) targets day 4's
   false-echo hazard, which suite A/B cannot trigger — a single-turn eval
   can't see it. To A/B a prompt restructure meaningfully, the eval must
   exercise the behavior the restructure claims to fix (here: a
   tool-rejection-then-claim task, or a format-compliance checker).
2. **Removing session-specific lines from system was free.** They were
   inert on single-turn evals; no regression from dropping them. Moving
   turn-level rules into user messages is the right placement, and this
   run confirms it costs nothing here.
3. **The +112 system tokens are pure overhead on this suite** — same
   success, ~$0.02 cheaper overall only because of run variance. Keep the
   new prompt only if day 6 shows it helps the long-session failure
   modes, where its constraints section can actually bind.

Results: `day5_results_20260822_215908.json` (full raw, 240 runs),
`day5_run.log` (console), `day5_results_20260822_215657.json` (smoke).

---

# Week 6 Day 6 — pruning vs. compaction vs. nothing: task success + cost

Run: `python experiments/week06/day6_context_management.py --selftest`
(first), then `--runs 5 --with-lazy` for the full batch.
Results: `day6_results_20260823_203635.json` (20 runs: 4 conditions x 5,
interleaved with fixed seed 20260823). Log: `day6_run.log`. Script:
`day6_context_management.py`.

## Design

Day 4's 12-turn probe battery (plants T1-3: random number / invented
filename / naming constraint; filler T4-9; probes T10-12) scored against
the sandbox filesystem, under FOUR context strategies applied at every turn
boundary:

| condition | strategy | tuning |
|---|---|---|
| `none` | control, history untouched | — |
| `prune` | day 2's clear_old_tool_results | n=3 |
| `compact` | day 3's compact_history + preserve-everything summarizer | N=8,000 |
| `compact_lazy` | same mechanism, deliberately sloppy summarizer ("2-3 sentences, gist only, details unimportant") | N=8,000 |

Day 4's lesson 1 said the preserve-everything summarizer kills nothing, so
the failure gradient needs a weaker summarizer — `compact_lazy` is that
extension (flag `--with-lazy`). System prompt = day 5's NEW_PROMPT in all
conditions (closes day 5's open question). Genuineness: a probe counts as
testing the strategy only if the strategy touched its planted material
before the probe fired — prune: a tool result from the probe's plant turn
was cleared; compact: a compaction absorbed turns >=3.

## Results — success

| Probe | none | prune | compact | compact_lazy |
|---|---|---|---|---|
| number | 5/5 | 5/5 (genuine 4/4) | 5/5 (genuine 3/3) | 5/5 (genuine 3/3) |
| filename | 5/5 | 5/5 (5/5) | 5/5 (3/3) | 5/5 (3/3) |
| constraint | 5/5 | 5/5 (5/5) | 5/5 (3/3) | **3/5 (genuine 1/3)** |

Fisher on all pairs: p=0.444 for the constraint probe lazy-vs-everyone
(direction consistent, n=5 too small for significance). **No strategy lost
a single valid probe except lazy compaction — which lost exactly the
constraint**, twice.

## The constraint death, seen in the summary texts

Preserve-everything (compact run 5): "- **Session Rule**: All new files
must be placed in `research/` and prefixed with `d4_`." — verbatim, with
the full 30-number list in the same summary.

Lazy (runs 2 and 4): "The user asked the assistant to generate and sort a
sequence of numbers while following **a specific file naming convention**."
The rule itself is gone — demoted to an unnamed convention. At T12 both
runs created files WITHOUT the prefix (`research/final_notes.md`,
`research/session_summary.md`). This is day 4's missing failure mode
materialized: critical-detail omission, produced by a sloppy summarizer
prompt rather than the mechanism itself.

## Results — cost (5 runs per condition)

| Metric | none | prune | compact | compact_lazy |
|---|---|---|---|---|
| agent input tokens | 1,610,916 | 1,216,917 | 1,413,746 | 1,398,538 |
| vs none | — | **−24.5%** | −12.2% | −13.2% |
| total tokens (incl. summarizer) | 1,657,481 | 1,263,200 | 1,522,469 | 1,486,435 |
| vs none | — | **−23.8%** | −8.1% | −10.3% |
| est. cost | $0.700 | **$0.542** | $0.666 | $0.639 |
| vs none | — | **−22.6%** | −4.9% | −8.7% |
| mean wall/run | 191s | 254s | 276s | 218s |

Pruning is the clear cost winner: zero extra API calls, zero success loss.
Compaction's gross savings are real but its OWN summarizer tokens eat most
of them (day 3's finding now replicated at 5 runs/condition: net −4.9%).

## Honest caveats

1. **The T09 session-notes turn refreshes the plants.** To write "a summary
   of everything we have done so far" at T09, the agent legitimately
   re-reads d4_numbers.md — putting the number (and often the filename)
   back into RECENT tool results, which no strategy touches (prune keeps
   the last n-1 turns; compaction keeps the fresh half). So the
   number/filename probes test recall only when the agent skips that
   refresh — and it's why lazy compaction killed only the constraint, which
   lives in no file. All conditions share the confound; the comparison
   stays fair, but the probe battery under-tests number/filename recall.
2. **Compaction didn't fire in 2 of 10 compact-condition runs** (compact
   run 1, lazy run 3 — thin filler history that never crossed N=8k).
   The genuine-only analysis keeps these runs from being counted as
   strategy successes.
3. **Prune run 1's number probe isn't genuine for an instructive reason**:
   at T02 the agent answered WITHOUT tools (derived the sequence from the
   formula in the T01 user message, which pruning never touches). The
   genuineness flag correctly refuses to credit pruning.
4. **The one prune run's number pass needed 4/4 not 5/5** for the same
   reason — see caveat 3.

## Environment saga (this one cost hours)

The DashScope endpoint has a stall variant that **outlives httpx's read
timeout**: the process sits in `_ssl__SSLSocket_read -> poll` with zero
bytes ever arriving (diagnosed via `sample` on the live process). Day 4's
"fixed with 300s/15s timeout" claim was luck, not a fix — the stall
returns. Worse, day 3's `compact_history` wraps the summarizer call in
`except Exception`, so a watchdog exception raised mid-summarizer is
SWALLOWED as a "summarizer error" and the one-shot alarm is consumed — the
run then walks into the next stall unprotected (observed: ~45 min and
~1h45m hangs; the smoke's compact run finished T12 at 18:27 and was still
stalled at 18:58).

Fixes that survived contact: (a) `_Deadline` in day6 — a SIGALRM budget
armed around EVERY API call (min(call budget 300s, remaining run budget
1200s)), re-armed between calls; a swallowed alarm now costs at most one
call and the next call is protected again. Standalone-verified: blocked
call killed at exactly 2.0s, re-arms per call. (b) Incremental JSON saves
after every run — a killed batch never loses completed runs. (c) The batch
itself then ran 20/20 runs with zero stalls (Sunday evening, ~2h). Also
killed mid-batch on a python-level error at the final report stage: my
print_report takes the conditions dict but main passed the whole summary —
fixed, and the report regenerated from the saved JSON (no data loss).

## Smaller fixes folded in

- **Filename ground truth**: day 4 globbed `d4_log_*.txt`; the agent now
  creates the log via a python script (bypassing the write_file extension
  whitelist), so extensionless `d4_log_*` files were invisible → auto-fail.
  Day 6 globs any `d4_log_*` file.
- **Day 5's question is answered**: the none condition ran day 5's
  NEW_PROMPT and hit 15/15 valid probes (day 4's old-prompt control was
  13/15 raw with 3 filename plant-failures). No regression, and the
  constraints section targets the false-echo hazard. Keep the new prompt.

## Week-6 capstone verdict

For this agent, on this battery: **prune (n=3) dominates** — it costs
nothing, saves 24% of agent input, and loses zero measurable task success,
because the week's key insight holds: tool results are bulk, decisions are
everything, and decisions survive pruning verbatim. Compaction's value is
purely defensive (it's the only strategy that shrinks *decisions*, which is
exactly what you need when the decision stream itself outgrows the window —
not the regime this battery exercises), and its quality floor is set by the
summarizer prompt: preserve-everything = safe but net-cheap, sloppy = the
constraint dies. Nothing = the cost ceiling, 5/5 everywhere, and the honest
baseline that makes the other two measurable.
