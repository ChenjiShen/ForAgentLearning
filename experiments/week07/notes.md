# Week 7 Day 1 — Embeddings + top-k similarity search with numpy

Run: `python experiments/week07/day1_embeddings.py --selftest` (offline checks,
10/10), then the full run (no flags).
Results: `day1_results_20260824_201520.json` (corpus, queries, per-query top-5,
metrics), `day1_embeddings_20260824_201520.npz` (E 50×1024, Q 12×1024),
`day1_plot_20260824_201520.png` (similarity heatmap + per-query bars).
Log: `day1_run.log`. Model: DashScope `text-embedding-v3`, dim 1024, via
`dashscope.aliyuncs.com/compatible-mode/v1` (also verified working: v4, and
the maas endpoint). 1,487 embedding tokens total, ~3s.

## What was built

- **Corpus**: 50 short texts, 10 themes × 5, harvested from this repo's own
  week 1–6 notes (agent_loop, tools, context, evals, temperature,
  memory_rag, cost_tokens, mcp, multiagent, planning). Reusable by day 2's
  RAG pipeline.
- **The search** — the whole lesson in three numpy lines:

  ```python
  E_hat = E / np.linalg.norm(E, axis=1, keepdims=True)  # L2-normalize
  sims  = E_hat @ q_hat                                  # all cosines at once
  idx   = np.argsort(-sims)[:k]                          # k best
  ```

  Cosine similarity == dot product of L2-normalized vectors, so scoring all
  n texts against one query is a single (n×d)·(d) matmul — O(n·d). A vector
  DB is an *index over this exact operation*: it earns its keep when n is
  millions (ANN indexes trade recall for speed, plus persistence/sharding).
  With n=50, numpy is the honest tool — that is the point the plan wants
  internalized.
- **Scoring** against theme labels: hit@k, MRR, precision@k per query; plus
  two global "clumping" numbers (intra- vs inter-theme mean cosine,
  nearest-neighbor-same-theme fraction).

## Results

| Metric | Value |
|---|---|
| mean MRR | **1.000** — the top-1 hit was the correct theme for all 12 queries |
| mean hit@5 | 2.75 / 5 |
| mean P@5 | 0.55 |
| intra vs inter theme cosine | 0.529 vs 0.470 (gap +0.059) |
| nn-same-theme fraction | 0.52 |

Per query: temperature 5/5, memory_rag 5/5, mcp 4/5, agent_loop 3/5,
context 3/5, multiagent 3/5, tools 2/5, evals 2/5, cost_tokens 1/5,
planning 1/5. The two extras: `q_vectordb` 2/5 (gold memory_rag — got the
two "no vector DB needed" texts), Chinese `q_evals_zh` 2/5 (cross-lingual
retrieval works: 如何衡量评估任务的通过率 → evals texts).

## What this teaches (honest read)

1. **Top-1 is trustworthy here; top-5 is polluted by adjacent themes.**
   MRR 1.0 says the query lands in the right neighborhood every time — the
   embedding space has real structure. But hit@5 = 2.75/5 and the tiny
   intra/inter gap (+0.059, inter-theme cosine 0.47!) say the neighborhoods
   are NOT cleanly separable. All 50 texts are about AI agents, so
   cost_tokens ↔ context ↔ mcp and planning ↔ multiagent bleed into each
   other — e.g. `q_planning`'s #1 hit was correct but slots 2–3 went to
   multiagent texts, and `q_cost_tokens` filled with context/mcp texts.
   This is normal embedding behavior on a homogeneous corpus, and it is
   exactly why day 2's plan says *chunking strategy dominates RAG quality*:
   if the retrieved neighbors are fuzzy, what you stuff into the prompt is
   what decides usefulness.
2. **The corpus itself is the strongest lever on metrics.** These themes
   were deliberately drawn from one project's notes (continuity with day 2),
   which guarantees soft boundaries. A benchmark-quality number would need
   harder, more separated topics — the 2.75 is a property of this corpus,
   not of text-embedding-v3.
3. **Multilingual retrieval just works.** The Chinese query retrieved the
   English evals texts (2/5, top-1 correct) with no special handling.
4. **The selftest is the real proof of understanding.** 10 offline checks
   verify the numpy math against a naive loop, a synthetic near-duplicate
   corpus (hit@5 = 1.0), and the cosine == dot(unit vectors) identity —
   before a single API call. The two bugs it caught were mine (a
   keyword-argument syntax error and queries sampled from one theme), not
   the API's.

## Day 2 handoff

Reuse `CORPUS` (and the npz if wanted) from `day1_embeddings.py`; day 2
needs real documents + a chunker (sizes, overlap) rather than pre-cut
texts. The embedding client (`get_client` + `embed_texts` with trust_env,
timeout, retry) carries over unchanged.

---

# Week 7 Day 2 — Mini RAG pipeline: chunk → embed → retrieve → stuff

Run: `python experiments/week07/day2_mini_rag.py --selftest` (23/23 offline
checks), `--smoke` (truncated docs, sanity), then the full run (no flags).
Results: `day2_results_20260825_223024.json` (config, chunk stats, every
answer + grade + usage + retrieved chunks), `day2_plot_20260825_223024.png`.
Smoke pass: `day2_results_20260825_221134.json`, `day2_smoke.log`.
Embeddings: `text-embedding-v3` (day-1 client, reused unchanged). Chat:
`qwen3.7-plus`, temp 0. Full run: 944s, ~693K tokens total.

## What was built

- **Documents**: `reports/week1.md`–`week6.md`, this repo's own notes —
  6 real markdown files, 370,288 chars. Not pre-cut texts: day 2 must chunk
  raw notes.
- **The pipeline**, four moves only:
  chunk → embed (batched, day-1 client) → retrieve (numpy top-k, k=4) →
  stuff (numbered chunks with sources + question) → chat answer.
- **The experiment variable — chunking**, a sweep with two clean
  single-variable contrasts + a structure-aware strategy:

  | strategy | rule | chunks | embed tokens |
  |---|---|---|---|
  | `fixed_400` | 400 chars, step 400 | 929 | 153,492 |
  | `fixed_800` | 800 chars, step 800 | 466 | 153,511 |
  | `fixed_800_o100` | 800 chars, step 700 (100 overlap) | 533 | 175,347 |
  | `para_800` | paragraph-merge, target 800 | 602 | 153,034 |

  plus a `no_retrieval` baseline (same questions, empty context).
- **Questions**: 6, one per document, each gold fact verified in the report
  (4/5 vs 0/5 wrong-tool, 97.5%/100% temperature rates, 30,000-token HTML /
  10% useful, 91% false negatives, 33.6% token increase, 45.6% / 37:1).
  Grader = evidence presence (all gold strings in normalized answer).

## Results

| condition | accuracy | hit@doc | prompt tok/q | verdict |
|---|---|---|---|---|
| no_retrieval | 0.000 | — | 68 | 0/6 — model refuses or guesses |
| fixed_400 | **0.833** | **0.708** | 775 | 5/6 |
| fixed_800 | 0.333 | 0.625 | 1,340 | 2/6 |
| fixed_800_o100 | **0.833** | 0.667 | 1,253 | 5/6 |
| para_800 | 0.500 (0.667 adj.) | 0.625 | 1,178 | 3/6, one grader miss |

Adjudicated: para_800's q_week4 answer was correct ("10 out of the 11…
fraction 10/11") but the grader wanted "91" — a false negative, adjudicated
pass by reading the answer (same policy as the week-4 report).

## What this teaches (honest read)

1. **Chunking strategy dominates RAG quality — demonstrated, not assumed.**
   Same documents, same embedder, same retriever, same model, same k=4.
   Only the chunker changed, and accuracy spanned 2/6 → 5/6. fixed_400 and
   fixed_800_o100 both hit 5/6; fixed_800 hit 2/6. The plan's day-2 claim
   is now a measured fact on this corpus.
2. **Small chunks beat big chunks for fact lookup** (5/6 vs 2/6 at the same
   cost). An 800-char chunk blends many topics into one embedding, so its
   vector matches the query worse and its top-4 ranking drops (hit@doc
   0.625 vs 0.708). Small chunks also cost *less* per question (775 vs
   1,340 prompt tokens at k=4).
3. **Overlap rescued the big chunks** (2/6 → 5/6) for +14% embedding
   tokens. Each fact appears in ~2 overlapping chunks, giving retrieval
   more views of the same text; boundary placement stops mattering.
4. **Failure taxonomy — 9 retrieval-condition failures, 3 kinds:**
   - *Retrieval* (model honestly said "I don't know from the context"):
     q_week6 everywhere (the 45.6%/37:1 facts sit at the start of week6.md,
     but the query's wording matches week5's many cost passages better —
     query phrasing + corpus redundancy decides), fixed_800 q_week1
     (3 of 4 chunks from the wrong doc poisoned the context), and the
     800-size q_week5 cells (right doc, wrong region).
   - *Generation*: fixed_800 q_week4 **hallucinated** "100% (11 out of
     11)" from partial context — confident, wrong, and cited the chunks.
     Noise stuffing can manufacture authority (day-1 corpus: retrieval
     "hurts when it stuffs noise into the context").
   - *Grader*: para_800 q_week4 answered correctly as "10/11"; the
     evidence grader wanted "91". Week 4's lesson reproduced live.
5. **Parametric knowledge is fuzzy and non-deterministic.** no_retrieval
   q_week3 passed the smoke run ("15,000 to 30,000 tokens") and failed the
   full run ("10,000 to 50,000") — same question, same temp 0, different
   guess from the reasoning model. Without retrieval the model knows the
   *shape* of well-known rules of thumb but not the exact figures; for the
   5 private facts it simply refused.
6. **Caveats (why day 6 exists):** single run per condition; n=6 questions;
   the evidence grader has false negatives. Directions are strong (0→5/6,
   2/6↔5/6) but unmeasured variance — day 6 re-measures with ≥5 runs.

## Day 3 handoff

Convert this pipeline into a `search_knowledge` agent tool: reuse
`CHUNKERS`, the retrieval code, and the evidence grader, but grade with the
week-4 LLM judge instead of evidence strings (the false negative above).
Best defaults so far: `fixed_400` or `fixed_800_o100`. Day 3 compares
agent-with-retrieval-tool vs everything-in-context on 5 questions.

---

# Week 7 Day 3 — search_knowledge tool vs everything-in-context

Run: `python experiments/week07/day3_agent_retrieval.py --selftest` (20/20
offline checks), `--smoke` (mechanism pass). **Full run pending** — command
below. Smoke audit: `day3_results_20260826_134500.json`,
`day3_plot_20260826_134500.png`; `day3_smoke_first_attempt_crashed.log` is
the first smoke attempt, which crashed on a results-format bug (now fixed)
before saving anything — kept as the bug's audit trail.

## What was built

- **The tool**: `search_knowledge(query, k)` — day 2's embed → numpy top-k →
  return, behind a function boundary with the week-3 result envelope. One
  tool schema, described in when-to-call terms. The index (fixed_400 chunks
  over week1–6 reports, embedded once) is a closure — nothing about the
  search changed, only *who decides to call it*.
- **Corpus note**: the reports have grown since day 2 (370K → ~518K chars),
  so the index and the everything-in-context corpus are bigger than day 2's.
- **Conditions** — 5 questions (day 2's wording, q_week6 dropped: it failed
  in ALL day-2 conditions, its phrasing matches the wrong passages, so it
  would only add retrieval noise):
  - `agent_with_tool` — agent + one tool, max 6 turns, system prompt budgets
    "at most 3 searches".
  - `always_stuffed` — day 2's pipeline verbatim (top-4 stuffed every time,
    no choice).
  - `everything_in_context` — the whole notebook (~518K chars ≈ 207K chat
    tokens) in one user message. If the API rejects it for context length,
    the script falls back to a whole-doc prefix (`--ctx-cap-chars`, default
    310,000 = week1–4) and records which docs were actually included.
- **Grading**: the week-4 LLM judge (rubric-first, PASS/FAIL first token,
  temp 0), with the week-7 transport fix (trust_env=False). Rubrics carry
  the exact facts, verified against the reports (4/5 vs 0/5; 97.5/100/97.5;
  30,000/10%; 91%; +33.6%).

## Smoke (mechanism check — docs truncated to 10K chars each)

All 15 cells ran, results saved, judge functioning. q_week5 passed in all
three conditions (its fact is at week5.md line 132, inside the truncation);
the other four facts sit beyond the 10K cut, so those failures are the
expected truncation artifact — smoke validates the mechanism, not quality.

Two real findings surfaced by smoke:
1. **Unbudgeted search thrashes.** The first smoke had the agent make 11
   search calls (26K prompt tokens) when the answer did not exist in the
   truncated docs. Fix: "at most 3 searches" in the system prompt. It still
   overshoots (6 calls) when the fact is truly absent — MAX_TURNS=6 is the
   hard bound. The budget wording is a *soft* instruction; that gap is
   itself a day-3 lesson about tool-using agents.
2. **The agent's retrieval was hitting the right docs** (q_week4 searches
   consistently returned week4.md) — failures were facts-absent, not
   retrieval-broken.

## Full run (2026-08-26 19:17)

`day3_results_20260826_191726.json`, `day3_plot_20260826_191726.png`,
`day3_run.log`. 907s. Corpus at run time: 370,288 chars — byte-identical to
day 2's full run (929 chunks, 153,492 embed tokens), so day 2 ↔ day 3 are
directly comparable. (The on-disk reports have since grown; a re-run would
use a different corpus.)

| condition | acc | turns | searches/q | prompt/q | tot_gen |
|---|---|---|---|---|---|
| agent_with_tool | **5/5** | 3.4 | 2.4 | 7,437 | 40,480 |
| always_stuffed | **5/5** | 1 | 0 | 806 | 9,158 |
| everything_in_context | **5/5** | 1 | 0 | 136,998 | 686,990 |

## Honest read

1. **15/15 all pass — the judge swap fixed day 2's false negative** (q_week4's
   "10/11" answer now grades correctly everywhere). Quality can't separate the
   conditions on these questions; cost can: everything-in-context is 17× the
   agent and 75× the stuffed pipeline.
2. **The 137K-token prompt was accepted** (window ≥ 137K) and answered in ~10s
   (~14K tok/s) — no fallback, no lost-in-the-middle at this scale (n=5).
3. **The search budget prompt is soft.** q_week4: 5 searches / 6 turns
   (17,235 prompt tokens) despite "at most 3 searches" — and searches 4–5 are
   what found the answer. MAX_TURNS is the real bound. Smoke's facts-absent
   case hit the same wall (6 searches).
4. **The agent adapts retrieval**: k raised from 4 to 6–8; q_week1's third
   query embedded the guessed answer ("4/5 0/5") as confirmation; q_week4's
   queries converged from "disagreements" to "taxonomy table".
5. **Retrieval results re-enter context every turn** — the 7,437 prompt/q is
   mostly loop tax (week 6's growth law via tool results). The facts
   themselves cost ~800 (stuffed's price).
6. Caveats: runs=1 (variance unmeasured — day-6-style re-measurement if it
   matters), n=5 questions, judge = same model family (self-preference bias
   possible, as in week 4).

---

# Week 7 Day 4 — File-based memory: read_memory/write_memory tools

Run: `python experiments/week07/day4_memory_tools.py --selftest` (21/21
offline checks), `--smoke` (mechanism pass), full run (no flags).
Results: `day4_results_20260829_210043.json` (full), `day4_run.log`;
smoke: `day4_results_20260829_205848.json`, `day4_smoke.log`. 92s full run,
41,512 gen tokens. Corpus again 370,288 chars — byte-identical to the
day 2/day 3 full runs (929 chunks / 153,492 embed tokens), so week 7's
experiments share one corpus.

## What was built

- **Two memory tools over `experiments/week07/memory/`**: `read_memory`
  (8,000-char cap, week-3 truncation marker) and `write_memory`
  (overwrite/append — append is what makes a file a real scratchpad). Both
  use the week-3 envelope and the week-3 sandbox discipline: model-supplied
  paths are realpath-resolved and must stay inside the memory root.
- **The plan's system-prompt note**: memory is a scratchpad for intermediate
  results and decisions; files persist, write don't-just-remember; read the
  notes back before the final answer. Plus day 3's search budget (≤6).
- **Agent = day 3's agent + 2 tools**: search_knowledge + read/write_memory,
  non-interactive loop, MAX_TURNS=12.
- **3 tasks graded by filesystem checks** (week 4: the filesystem is ground
  truth text judging is blind to) + final-answer checks.

## Results

| task | file checks | answer checks | turns | calls | prompt tok |
|---|---|---|---|---|---|
| t1_write_read_prefs | 4/4 | 2/2 | 3 | 1 write + 1 read | 3,384 |
| t2_scratchpad_notes | 3/4 | 2/3 | 7 | 4 search + 3 write + 1 read | 27,569 |
| t3_arithmetic_scratchpad | 5/5 | 1/1 | 6 | 4 write + 1 read | 8,082 |

t3's trace is the textbook frontier-agent pattern: overwrite once, then
append × 3, then read the scratchpad back and answer — 16,191 correct.

## Honest read

1. **Smoke caught a real path bug: the model wrote `memory/x.md`.** Files
   landed in a nested `memory/memory/` dir — tools worked, paths were the
   bug. Fix: canonicalize model paths (strip `./` and `memory/` prefixes)
   + explicit "no memory/ prefix" descriptions. Week 3's "model-supplied
   paths are untrusted input" reproduced on the memory tools.
2. **Memory amplifies, it doesn't verify.** t2's Q3: one search returned
   chunks without the exact number; the agent overgeneralized "dominant
   disagreement pattern" into "all 11 (100%)", wrote that to scratchpad.md,
   read it back, and repeated it in the final answer. The scratchpad worked
   flawlessly — it faithfully persisted a wrong conclusion. Notes-to-self
   do not protect against a bad reading of the source; they make errors
   durable. This is day 5's memory-poisoning topic in embryonic form.
3. **t2's miss was a generation error, not retrieval or memory failure.**
   The search hit week4.md; the answer over-claimed. Compare day 3, where
   the same fact took 5 converging searches to find — day 4's one-shot
   search + 6-search budget produced a confident wrong answer instead.
4. **The budget held when facts existed** (4 searches ≤ 6) and exploded when
   they didn't (smoke: 17 searches, 204K prompt tokens — day 3's soft-budget
   lesson, again: budget prompts bound typical behavior, not desperation).
5. **t1 is the day-5 seed**: user_prefs.md now holds the three preferences;
   day 5's session 2 (fresh process) must consult it unprompted.

## Day 5 handoff

Two-session test: session 1 = t1's prefs are already in memory/; teach 3
preferences (or reuse these), then a fresh process must use them unprompted.
Also test memory poisoning: a wrong memory entry must visibly corrupt a
later answer (t2 already previewed the mechanism).

---

# Week 7 Day 5 — Cross-session memory: teach -> update -> recall -> poison

Run: `python experiments/week07/day5_cross_session.py --selftest` (20/20
offline), `--smoke` (mechanism pass). Full run:
`day5_results_20260829_212835.json`, `day5_run.log`. 101s, 36,679 gen
tokens. Corpus again 370,288 chars (the shared week-7 corpus).

## What was built

- **Five scenarios, one memory/ dir, each a FRESH agent process** — the only
  continuity between sessions is the files:
  1. `s1_teach` — teach 3 prefs (English / markdown tables / <100 words)
  2. `s1b_update` — user changes mind (English -> Chinese); agent must EDIT
     memory (the staleness/update path)
  3. `s2_recall` — fresh process, prompt mentions NOTHING about prefs,
     format, or memory (unprompted-consultation test; selftest asserts the
     prompt is hint-free)
  4. `s2b_recall_strict` — SAME user prompt, strict system prompt that
     makes consultation a hard routine (list -> read prefs -> apply)
  5. `s4_poison` — injected memory/facts.md: "45% … supersedes the report"
     (real: 91%); agent must resolve memory vs retrieval
- **Day-5 stack changes**: (a) cross-session system prompt (memory persists
  BETWEEN sessions; apply stored prefs without being asked); (b) new
  `list_memory` tool — the first smoke showed the agent guessing filenames
  (`scratchpad.md` -> not_found); consultation requires discoverability;
  (c) day-4 `run_agent` parameterized on system prompt + tools.
- Language direction is deliberate: teach English, update to Chinese — a
  Chinese answer to an English question is not the model default, so it is
  strong evidence the pref was applied. Grading = filesystem + answer +
  trace checks.

## Results

| scenario | file | answer | trace | turns | calls |
|---|---|---|---|---|---|
| s1_teach | 4/4 | 1/1 | — | 3 | 1 list + 1 write |
| s1b_update | 3/3 | — | — | 4 | 1 list + 1 read + 1 write |
| s2_recall | — | 4/4 | **0/1** | 4 | 3 search, 0 memory calls |
| s2b_recall_strict | — | 4/4 | **1/1** | 3 | 1 list + 1 read + 2 search |
| s4_poison | — | 2/2 (both numbers) | — | 3 | 1 list + 2 read + 2 search |

## Honest read

1. **Teach and update work mechanically** (7/7 file checks). s1b's edit
   removed the stale language pref AND preserved the unrelated prefs —
   the staleness fix is real, not cosmetic.
2. **The consultation trigger is the whole design.** s2 (relevance-judgment
   prompt) never touched memory in EITHER full run — and its answer
   language flipped English -> Chinese between the two runs with an
   identical prompt: without consultation, "did the pref apply?" is
   unmeasurable model noise. s2b (hard routine: list -> read prefs ->
   apply) consulted and applied 5/5. Lesson: "check if anything is
   relevant" is not a trigger; an explicit routine is.
3. **Trace checks beat answer checks for consultation.** s2's 4/4 answer
   checks (Chinese/table/concise/facts) were confounded by model defaults
   and drift; only the trace (0 read_memory) exposed that memory was never
   consulted.
4. **Memory poisoning did not silently win — conflict detection emerged.**
   s4's agent read the poisoned facts.md (45%), searched the notebook
   (91%), produced a source-comparison table, quoted the report's exact
   sentence, and refused to commit ("无法确认哪个是最终正确值") — in the
   stored pref language, Chinese. Poison bites when retrieval CANNOT
   refute it (smoke: truncated corpus -> the agent answered 45%); when
   contradicting evidence is retrievable, the agent flags the conflict
   without being told to.
5. **Discoverability is a prerequisite.** Before list_memory, the agent
   guessed `scratchpad.md` and gave up (smoke #1). Listing the dir fixed
   the guesswork — frontier agents list before reading.
6. Small warts: one API response contained a tool call with an empty name
   (recorded, harmless — the dispatch returned an error envelope and the
   agent continued); MAX_TURNS=12 was hit only in smoke when facts were
   absent.

## Day 6 handoff

The controlled experiment: does the memory scratchpad improve multi-step
task success, or just add tokens? Two conditions (with vs without
read/write_memory), same multi-step tasks, >=5 runs each, success + tokens
+ latency. Day 5's lesson to reuse: measure consultation from the TRACE,
not from answer style.

---

# Week 7 Day 6 — Experiment: does the memory scratchpad help, or just add tokens?

Run: `python experiments/week07/day6_memory_experiment.py --selftest`
(11/11 offline), `--smoke` (mechanism; also caught the metrics-key plot
bug), full run: `day6_results_20260829_215541.json`, `day6_run.log`,
`day6_plot_20260829_215541.png`. 731s, 30 cells (3 tasks x 2 conditions x
5 runs), ~830K gen tokens. Corpus: the shared 370,288-char week-7 corpus.

## Design

One variable — the memory tools. `with_memory` = day-5 stack (search +
read/write/list memory, scratchpad system prompt); `no_memory` = search
only, system prompt with no memory section (selftest asserts the two
prompts differ ONLY there). Same tasks, same model/temp 0, same budget.
Three multi-step tasks, all answerable without memory: t1 = 3 notebook
retrievals + full-number summary; t2 = 5-step chained arithmetic; t3 =
retrieve + RE-verify with a second search. Grading = normalized evidence
presence over gold groups. Memory use measured from the TRACE (day 5's
lesson).

## Results — 30/30 cells pass; the scratchpad added tokens, not success

| task | condition | acc | turns | calls | mem | tok/cell | sec |
|---|---|---|---|---|---|---|---|
| t1_three_facts_summary | with_memory | 1.000 | 7.80 | 7.80 | 1.40 | 47,224 | 41.3 |
| t1_three_facts_summary | no_memory | 1.000 | 7.40 | 6.40 | 0.00 | 34,342 | 37.0 |
| t2_arithmetic_chain | with_memory | 1.000 | 1.00 | 0.00 | 0.00 | 1,668 | 11.2 |
| t2_arithmetic_chain | no_memory | 1.000 | 1.00 | 0.00 | 0.00 | 1,097 | 10.1 |
| t3_verify_peak_temp | with_memory | 1.000 | 3.00 | 2.20 | 0.00 | 8,543 | 20.1 |
| t3_verify_peak_temp | no_memory | 1.000 | 3.20 | 2.20 | 0.00 | 7,270 | 19.2 |

Memory was used in 5/15 with_memory cells (mean 0.47 calls/cell) — all five
in t1; t2 and t3 never touched it.

## Honest read — "just adds tokens" is the measured answer, with a boundary

1. **Zero headroom, zero lift.** Both conditions hit 100% on all three
   tasks, so the scratchpad had nothing to improve. This is the ceiling
   problem: memory's benefit is invisible when the baseline is perfect.
   The honest answer to the plan's question on THIS task set is "just adds
   tokens" — t1 cost +37% more (47.2K vs 34.3K tokens/cell) when memory
   was available, driven mostly by extra turns and search thrash, not by
   useful notes.
2. **The scratchpad is mostly ignored when offered.** Only 5/15 cells used
   it, never more than 2 calls/cell, and only on the multi-fact task. The
   model prefers holding state in context — which works fine because these
   tasks never push context anywhere near the window (worst cell ~94K
   prompt tokens).
3. **t2 is the clean boundary case.** The 5-step chain was solved in ONE
   turn with ZERO tool calls: the model computed it in its head. A
   scratchpad cannot help a task that needs no tools at all — memory pays
   only when intermediate state is produced by tool calls and is too bulky
   or fragile to keep in context.
4. **Variance swamps the condition effect.** t1 with_memory cells spanned
   10.7K-94.1K tokens (3-12 turns) — run-to-run behavior dominates any
   with/without difference at this scale. The two heaviest cells were both
   with_memory thrash (71.7K, 94.1K), which is variance, not a memory
   effect.
5. **Where the scratchpad SHOULD pay (not tested here):** tasks with real
   context pressure — 10+ tool calls whose results push early intermediate
   state toward the middle/out of the window (week 6's growth law). That is
   exactly week 9's day-1 task spec (multi-step tasks where myopia hurts);
   re-running this A/B on those tasks is the honest follow-up.

## Day 7 handoff

Report week 7. Phase 2 closes with week 8 (MCP): the day-6 finding ("memory
pays only under context pressure") is the seed for week 9's myopia tasks.
