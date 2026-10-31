# Daily Learning Plan — AI Agent Engineering

Day-by-day plan for the 14-week roadmap in [README.md](README.md).

## How to use this plan

- **Every day = Learn (20–30 min) + Build (40–50 min) + Log (5–10 min).** The Log step is always the same — fill in `logs/YYYY-MM-DD.md` — so it isn't repeated below.
- **Week structure:** Days 1–5 learn & build, Day 6 run the experiment, Day 7 write the report and add the week's tasks to `evals/tasks.json`.
- **Falling behind is fine; skipping the Build is not.** If time is short, cut the reading, never the coding. A day's "Build" that takes two days is normal — just shift the plan, don't skip.
- **Done-when checks** are listed for the days where "done" is easy to fake. Be honest with them.

---

# PHASE 1 — FOUNDATIONS

## Week 1 — The agent loop from scratch

Goal: understand every line of `week01_minimal_agent/agent.py` and be able to rewrite it from memory.

| Day | Learn (20–30 min) | Build (40–50 min) |
|---|---|---|
| 1 | How chat LLM APIs actually work: messages array, roles (system/user/assistant/tool), why the API is stateless, what `usage` reports. Read your provider's chat-completions docs. | Make a raw API call **without tools**. Print the entire response object. Identify every field. Add a second turn manually (append the assistant reply, ask a follow-up). |
| 2 | Tool calling fundamentals: how the model "calls" a tool (it doesn't — it emits JSON and stops), `finish_reason`, `tool_call_id`. Read the OpenAI function-calling guide. | Run `agent.py` with the debug prints. Trace one full loop iteration on paper: what exactly is in `messages` before and after each API call? (Exercise 1) |
| 3 | Tool schemas: JSON Schema basics, `required`, `enum`, why descriptions are prompt engineering. | Exercise 2: add a `read_file` tool (with path-safety check). Then write two versions of its description — vague vs. precise — and observe when the model picks the wrong tool. |
| 4 | Error handling in agents: why you return errors to the model instead of crashing, what the model does with them. | Exercise 3: make `calculator` raise randomly 50% of the time. Watch the model retry. Then make the error message useless (`"Error"`) vs. informative — compare recovery. |
| 5 | Read the **ReAct paper** (Yao et al. 2022) — just sections 1–3. Map its Thought/Action/Observation onto your loop. | Exercise 4: ask a question needing both tools; observe parallel tool calls (or your provider's lack of them). Then rewrite the agent loop **from memory** in a new file. Done when: it runs without peeking. |
| 6 | — | **Experiment:** terse vs. detailed tool descriptions → wrong-tool-call rate. 2 conditions × 5 runs × 3 tasks. Save to `experiments/week01_tool_descriptions/`. |
| 7 | — | Write `reports/week01.md`. Add 2–3 new tasks to `evals/tasks.json`. Run `evals/run_evals.py 5` for your baseline. |

## Week 2 — API mechanics that production code needs

Goal: streaming, structured output, token accounting — the plumbing every real agent app has.

| Day | Learn | Build |
|---|---|---|
| 1 | Streaming: SSE, delta events, why UIs stream. Read your provider's streaming docs. | Add `stream=True` to the agent's final answer; print tokens as they arrive. Note: tool-call deltas are fiddly — streaming just the final text turn is enough today. |
| 2 | Structured outputs: JSON mode / `response_format`, schema-constrained output, why "please reply in JSON" isn't reliable. | Build a small extractor: given a messy text, return validated JSON (use `pydantic` to validate). Feed it 5 tricky inputs; count validation failures. |
| 3 | Tokens: what a token is, why counts differ per model, input vs. output pricing. | Add token & cost tracking to the agent loop — accumulate `usage` across iterations, print total tokens + estimated cost per run. Keep this in every future agent. |
| 4 | Context windows: what happens at the limit, why long tool outputs are dangerous. | Deliberately blow up the context (a tool that returns 50K characters). Observe the failure. Then add naive protection: truncate tool results over N chars. |
| 5 | Sampling: temperature, top_p, why temperature=0 ≠ deterministic. | Run the same prompt 10× at temperature 0 / 0.7 / 1.2. Measure: identical outputs? Same tool calls? This is your first taste of variance measurement. |
| 6 | — | **Experiment:** effect of temperature on agent task success rate (not just text variance). 3 temps × 5 runs on your eval tasks. |
| 7 | — | Report + add tasks to evals. |

## Week 3 — Real tools

Goal: an agent with file, shell, and web tools — the standard trio — plus the safety thinking they require.

| Day | Learn | Build |
|---|---|---|
| 1 | Path traversal attacks; why model-supplied paths are untrusted input. | File tools: `read_file`, `write_file`, `list_dir`, confined to a sandbox root (resolve + verify the path stays inside). Try to escape your own sandbox. |
| 2 | Command injection; allowlist vs. blocklist (blocklists always lose). | A `run_command` tool with an executable allowlist, timeout, and output cap. Ask the agent to do something the allowlist forbids; watch how it copes. |
| 3 | How web tools work in agents: search APIs vs. scraping, why raw HTML wrecks the context window. | A `web_fetch` tool: `requests` + HTML-to-text extraction + length cap. Optionally a `web_search` tool via a search API. |
| 4 | Read **Anthropic's "Building Effective Agents"** — the single best practical read so far. Map its workflow patterns onto what you've built. | Combine all tools into one agent. Give it a real multi-step task: "find X online, save a summary to notes.md". Watch the full trace. |
| 5 | Tool-result design: what the model needs back (structure, brevity, errors it can act on). | Refactor your tool outputs: consistent format, truncation with a "truncated at N chars" marker, actionable errors. Re-run Day 4's task; compare traces. |
| 6 | — | **Experiment:** task success vs. number of available tools — same task with 4 tools vs. those 4 + 8 irrelevant ones. |
| 7 | — | Report + evals. |

---

# PHASE 2 — CORE AGENT ENGINEERING

## Week 4 — Evals (the frontier skill)

Goal: a harness you trust. Everything after this week is measured, not vibed.

| Day | Learn | Build |
|---|---|---|
| 1 | Why evals are hard: contamination, flakiness, overfitting to your own test set. | Write 10 real tasks for your agent in `evals/tasks.json` — mix easy/medium/hard, tool-requiring and not. |
| 2 | Grader types: exact match, substring, programmatic checkers, LLM-as-judge. | Extend `run_evals.py`: support a `checker` field per task — `substring`, `exact`, and `python` (a code snippet that validates the answer or a produced file). |
| 3 | LLM-as-judge: rubrics, position bias, self-preference bias, why judge ≠ ground truth. | Add an `llm_judge` checker (a second API call with a rubric). Validate it: hand-label 10 answers yourself, measure judge agreement with you. |
| 4 | pass@k and variance: why one run means nothing. | Run the full suite 5×. Compute per-task pass rate. Identify your flakiest task and figure out *why* it's flaky. |
| 5 | Regression tracking. | Make the harness append run summaries to `evals/history.jsonl` (date, agent version, pass rate). Print a comparison with the previous run. |
| 6 | — | **Experiment:** substring checker vs. LLM judge — how often do they disagree, and which one is wrong? |
| 7 | — | Report + evals. |

## Week 5 — Tool design deep dive

Goal: internalize that tool surface design is the highest-leverage agent decision.

| Day | Learn | Build |
|---|---|---|
| 1 | Reread tool-design guidance (Anthropic's tool-use docs); descriptions as triggering conditions ("call this when…"). | Rewrite all your tool descriptions with explicit *when-to-call* conditions. Measure should-call rate before/after on 10 prompts. |
| 2 | Granularity: one powerful tool vs. many narrow ones; bash-vs-dedicated-tools tradeoff. | Build the same capability twice: one `file_op(action, ...)` mega-tool vs. three narrow tools. Compare error rates on the same tasks. |
| 3 | Tool-count effects and context cost of schemas. | Measure: tokens consumed by tool schemas alone at 5/15/30 tools. Then measure wrong-tool rate at each count. |
| 4 | Tool-use examples & few-shot in descriptions. | Add worked examples to your two most-misused tools. Re-measure. |
| 5 | Forced tool use (`tool_choice`), disabling parallel calls — when to constrain the model. | Build a router: first call forces a `classify_request` tool, then dispatches to a specialist prompt. Compare vs. one big agent. |
| 6 | — | **Experiment:** pick the most surprising effect from Days 1–5 and measure it properly. |
| 7 | — | Report + evals. |

## Week 6 — Context management

Goal: an agent that survives long sessions without drowning in its own history.

| Day | Learn | Build |
|---|---|---|
| 1 | Anatomy of context growth: measure it. | Instrument your agent: log tokens per turn across a 20-turn session. Plot (or table) the growth. Where does it come from — messages or tool results? |
| 2 | Pruning: dropping old tool results (they're the bulk), keeping decisions. | Implement `clear-old-tool-results`: replace tool results older than N turns with `"[cleared]"`. Verify the agent still functions. |
| 3 | Summarization/compaction: replacing history with a summary; what information dies in summaries. | Implement compaction: when history > N tokens, summarize the older half with an LLM call and splice it in. |
| 4 | Compaction failure modes. | Design 3 tasks that *require* details from early context (a number, a filename, a constraint). Measure success with/without your compaction. This is a real result — keep it for the report. |
| 5 | System prompt design: what belongs in system vs. user; instruction placement and stability. | Restructure your agent's system prompt: role, tool guidance, constraints, output format. A/B the old vs. new on your evals. |
| 6 | — | **Experiment:** pruning vs. compaction vs. nothing on long-session task success + cost. |
| 7 | — | Report + evals. |

## Week 7 — Memory & RAG

Goal: know when retrieval helps an agent and when it hurts.

| Day | Learn | Build |
|---|---|---|
| 1 | Embeddings: what they are, cosine similarity, embedding models. | Embed 50 short texts, implement top-k similarity search with numpy. No vector DB — arrays are enough to learn. |
| 2 | Chunking: sizes, overlap, why chunking strategy dominates RAG quality. | Build a mini RAG pipeline over 5–10 documents (your own notes work): chunk → embed → retrieve → stuff into prompt. |
| 3 | RAG *for agents*: retrieval as a tool the agent calls vs. always-stuffed context. | Convert your RAG pipeline into a `search_knowledge` tool. Compare: agent-with-retrieval-tool vs. everything-in-context on 5 questions. |
| 4 | File-based memory: scratchpads, notes-to-self; why frontier agents write markdown files. | Give your agent `read_memory`/`write_memory` tools over a `memory/` dir + a system-prompt note about them. Run tasks that benefit from remembering. |
| 5 | Cross-session memory design: what's worth persisting, staleness, memory poisoning. | Two-session test: session 1 teaches the agent 3 preferences; session 2 (fresh process) must use them. Verify it consults its memory unprompted. |
| 6 | — | **Experiment:** does the memory scratchpad improve multi-step task success, or just add tokens? |
| 7 | — | Report + evals. |

## Week 8 — MCP (Model Context Protocol)

Goal: build both sides of MCP — it's becoming the standard integration layer.

| Day | Learn | Build |
|---|---|---|
| 1 | MCP architecture: hosts, clients, servers; tools/resources/prompts; stdio vs. HTTP transports. Read modelcontextprotocol.io intro. | Install an existing MCP server (e.g. the filesystem server) and poke at it with the MCP inspector tool. |
| 2 | The MCP Python SDK. | Build your own MCP server exposing 2 of your week-3 tools (stdio transport). Test with the inspector. |
| 3 | MCP client side: session lifecycle, `list_tools`, `call_tool`. | Write an MCP client that connects to your server, lists tools, and calls one — no LLM yet, just the protocol. |
| 4 | Bridging MCP to LLM tool calling. | Connect your agent to your MCP server: convert MCP tool listings into OpenAI tool schemas, route tool calls through the MCP session. Your agent now uses MCP tools transparently. |
| 5 | MCP vs. plain function calling: when the indirection pays (sharing tools across agents/apps) and when it's overhead. | Connect your agent to a *third-party* MCP server you didn't write. Note every friction point. |
| 6 | — | **Experiment:** same tools called natively vs. via MCP — latency and reliability overhead. |
| 7 | — | Report + evals. **Phase 2 done — you now have: an agent, a tool library, a memory system, an MCP integration, and an eval harness.** |

