# AI Agent Learning Lab

A structured self-study project for going from AI application engineer to frontier-adjacent agent engineer.

**Core principle:** build everything from scratch once before using a framework, and measure everything. The weekly experiment habit *is* the training.

## Directory layout

```
ForLearning/
├── README.md              ← this file (roadmap + how to work)
├── PLAN.md                ← the day-by-day plan (what to learn/build each day)
├── requirements.txt
├── logs/                  ← daily logs (one file per day: 2026-07-12.md)
│   └── TEMPLATE.md
├── reports/               ← weekly reports (one per week: week01.md)
│   └── TEMPLATE.md
├── experiments/           ← one folder per weekly experiment
│   └── README.md          ← how to structure an experiment
├── evals/                 ← the compounding eval harness (tasks accumulate here)
│   ├── tasks.json
│   └── run_evals.py
└── week01_minimal_agent/  ← week 1 build: agent loop from scratch
    └── agent.py
```

> **Day-by-day version of this roadmap: [PLAN.md](PLAN.md)** — what to learn and build every single day.

## Roadmap

### Phase 1 — Foundations (weeks 1–3)
Work directly with the raw API, no frameworks.
- **Week 1:** minimal agent loop from scratch (`week01_minimal_agent/agent.py` is your starting point — read it, run it, extend it)
- **Week 2:** streaming, structured outputs, prompt caching, token counting
- **Week 3:** add 3–4 real tools; understand parallel tool calls and error feedback

### Phase 2 — Core agent engineering (weeks 4–8)
- **Week 4:** build the eval harness (`evals/`) — 10–20 fixed tasks, pass/fail scoring. Reused every week after.
- **Week 5:** tool design — descriptions, error messages, granularity
- **Week 6:** context management — compaction, what to keep/drop in long sessions
- **Week 7:** memory and RAG for agents
- **Week 8:** MCP — build one server and one client

## Daily rhythm (60–90 min)

1. **20–30 min reading** — blog post, paper section, or framework source code
2. **40–50 min hands-on** — advance the current week's build; end by writing down tomorrow's first step
3. **5–10 min log** — copy `logs/TEMPLATE.md` to `logs/YYYY-MM-DD.md`, fill it in

## Weekly report

Every week: copy `reports/TEMPLATE.md` to `reports/weekNN.md` and run one small experiment
(see `experiments/README.md` for structure and starter ideas). Add the week's tasks to
`evals/tasks.json` so later agents are tested against earlier tasks automatically.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                                  # then put your real DASHSCOPE_API_KEY in .env
export $(grep DASHSCOPE_API_KEY .env | xargs)         # or: export DASHSCOPE_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-provider.com/v1   # any OpenAI-compatible endpoint; omit for openai.com
export MODEL_NAME=your-model-name                     # e.g. deepseek-chat, glm-4, qwen-plus
python week01_minimal_agent/agent.py
```
