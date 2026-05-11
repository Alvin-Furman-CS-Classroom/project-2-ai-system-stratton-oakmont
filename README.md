# Intelligent Trading Agent: Strategy Discovery Through Search and Adaptive Risk Management

## Overview

This repository implements a five-stage intelligent trading system. The unifying theme is moving from **explainable rule-based decisions** to **searched and evolved strategies**, then conditioning those strategies on **market sentiment** and sizing risk with **reinforcement learning**.

In **Phase 1 (Modules 1–3)**, Module 1 evaluates propositional trading rules in CNF form over standard indicators (RSI, MACD, moving averages, volume, volatility) and returns a traceable BUY/SELL/HOLD decision with an inference chain. Module 2 treats strategy parameters as a search space and ranks candidates with informed search (beam search and A*), using backtested Sharpe and related metrics. Module 3 merges Module 2’s top candidates with strategies produced by a genetic algorithm started from random seeds, applies constraints and preferences, and selects a single final strategy with an explicit reason.

In **Phase 2 (Modules 4–5)**, Module 4 ingests news sentiment (Alpha Vantage), engineers features, fits a logistic-regression-style regime classifier, and recommends which discovered strategy fits the current regime. Module 5 models position sizing as an MDP and uses Q-learning over discrete position percentages, consuming sentiment, strategy quality, volatility, and capital to output a recommended allocation plus reasoning.

Shared types and market helpers in `src/shared/` keep handoffs between stages consistent. Staged integration tests under `integration_tests/module_2` through `module_5` document and enforce the pipeline contracts without always requiring live APIs (mocks where appropriate).

## Team

- Kyler Bailey
- Collin Riddle
- Casen Shoemake


## Proposal

Approved design and rationale are in [Project 2 Proposal.md](Project%202%20Proposal.md) (overview, AI techniques per module, and original milestone table). A practical layout, handoffs, and timeline are expanded in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Module Plan

| Module | Topic(s) | Inputs | Outputs | Depends On | Evidence |
| ------ | -------- | ------ | ------- | ---------- | -------- |
| 1 — Trading rule knowledge base | Propositional logic, CNF rules, forward chaining | Market indicators (RSI, MACD, MA20, MA50, volume, volatility), rule set | BUY/SELL/HOLD, fired rules, inference trace | — | Report: `checkpoints/checkpoint_1/` |
| 2 — Strategy parameter search | Informed search (A*, beam), heuristics, backtest | Parameter ranges, OHLCV history | Top candidate strategies ranked by Sharpe (and related backtest stats) | 1 | Report: `checkpoints/checkpoint_2/` |
| 3 — Evolution and unified selection | Genetic algorithm, constrained selection | Module 2 pool, full history, GA config, constraints/preferences | One selected strategy, metrics, selection reason (M2 vs GA origin) | 1, 2 | Report: `checkpoints/checkpoint_3/` |
| 4 — Market sentiment | Supervised classification (e.g. logistic regression), API client | News/sentiment feed (Alpha Vantage), candidate strategies from 3 | Regime label, confidence, headlines, strategy recommendation + rationale | 1–3 | Report: `checkpoints/checkpoint_4/` |
| 5 — Adaptive position sizing | MDP, Q-learning | Regime and confidence from 4, strategy metrics, volatility, capital | Discrete position % (see `POSITION_PCTS` in code), Q-values, risk-style reasoning | 1–4 | Report: `checkpoints/checkpoint_5/` |

## Repository Layout

```
project-2-ai-system-stratton-oakmont/
├── src/
│   ├── module_1_knowledge_base/       # rules, inference engine, demos
│   ├── module_2_strategy_search/      # search, backtest, evaluation, demo
│   ├── module_3_evolution/            # GA, evolution demos
│   ├── module_3_strategy_selection/   # unified pools, selection, reporting
│   ├── module_4_sentiment/            # features, classifier, pipeline, Alpha Vantage client
│   ├── module_5_position_sizing/      # MDP, agent, sizing API, demos
│   └── shared/                        # types, market data helpers
├── unit_tests/                        # mirrors modules under src/
├── integration_tests/
│   ├── module_2/                      # M1 + M2
│   ├── module_3/                      # M1 + M2 + M3
│   ├── module_4/                      # M1–M4
│   └── module_5/                    # M1–M5 (mocked APIs where needed)
├── checkpoints/                     # checkpoint reports and elegance reviews
├── data/                            # optional outputs (e.g. pipeline HTML/PNG reports)
├── run_pipeline.py                  # interactive demo: ticker → M1 → M2 → M3 → M4
├── requirements.txt
├── .claude/skills/code-review/SKILL.md
├── AGENTS.md
├── PROJECT_STRUCTURE.md
└── README.md
```

## Setup

1. **Python:** 3.10 or newer recommended.
2. **Virtual environment (recommended):**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment variables (optional but needed for live Module 4):**
   - `ALPHA_VANTAGE_API_KEY` — set in the shell or in a `.env` file at the repo root (`python-dotenv` loads `.env` when present). Without it, Module 4 network features and `run_pipeline.py` sentiment step will fail gracefully with a message; tests use mocks.

4. **Market data:** Historical OHLCV for demos uses Yahoo Finance via `yfinance` (network required for live fetch; synthetic fallback exists in `run_pipeline.py` if the fetch fails).

## Running

| What | Command |
| ---- | ------- |
| Interactive pipeline (prompts for ticker; runs M1–M4; writes report under `data/` when M4 succeeds) | `python run_pipeline.py` |
| Module 1 demo | `python -m src.module_1_knowledge_base.demo` |
| Module 2 demo | `python -m src.module_2_strategy_search.demo` |
| Module 3 evolution / selection demos | `python -m src.module_3_evolution.demo_evolution`, `python -m src.module_3_strategy_selection.demo_selection`, etc. |
| Module 4 demo (live API key recommended) | `python -m src.module_4_sentiment.demo` |
| Module 5 demo / charts | `python -m src.module_5_position_sizing.demo_visualization` |

**Note:** `run_pipeline.py` currently wires through **Module 4**; Module **5** is integrated in tests and module demos. For a full M4→M5 path in code, see `integration_tests/module_5/test_full_pipeline.py`.

## Testing

From the repository root:

```bash
python -m pytest unit_tests integration_tests -q
```

**Focused suites** (from `AGENTS.md`):

```bash
python -m pytest unit_tests/module_1_knowledge_base unit_tests/module_2_strategy_search integration_tests/module_2 -q
python -m pytest unit_tests/module_3_evolution unit_tests/module_3_strategy_selection integration_tests/module_3 -q
python -m pytest unit_tests/module_4_sentiment integration_tests/module_4 -q
python -m pytest unit_tests/module_5_position_sizing integration_tests/module_5 -q
```

**Test data:** Most tests use synthetic or bundled-style data. Module 4 integration tests mock HTTP; no API key is required for the default test run. Search/backtest-heavy tests can be slower; see `unit_tests/README.md` for layout notes.

## Checkpoint Log

Each row matches a deliverable folder under `checkpoints/` (module rubric and/or elegance notes inside).

| Checkpoint | Date (as in report) | Modules Included | Status | Evidence |
| ---------- | ------------------- | ---------------- | ------ | -------- |
| 1 | Feb 10, 2026 | Module 1 | Complete | `checkpoints/checkpoint_1/` |
| 2 | Feb 26, 2026 | Module 2 | Complete | `checkpoints/checkpoint_2/` |
| 3 | — | Module 3 | Complete | `checkpoints/checkpoint_3/` |
| 4 | — | Module 4 | Complete | `checkpoints/checkpoint_4/` |
| 5 | — | Module 5 + overall system | Complete | `checkpoints/checkpoint_5/` |

Target due dates for the course appear in `PROJECT_STRUCTURE.md` (e.g. CP1 ≈ Feb 11). Fill in the **Date** column for checkpoints 3–5 when you finalize those reports, if your instructor wants explicit submission dates.

## Required Workflow (Agent-Guided)

Before each module:

1. Write or update the short module spec in this README (inputs, outputs, dependencies, tests).
2. Ask the agent to propose a plan in Plan mode.
3. Review and approve the approach.
4. Implement in `src/`.
5. Add unit tests under `unit_tests/` (parallel to `src/`).
6. For modules after the first, add `integration_tests/module_<n>/`.
7. Run a rubric review using `.claude/skills/code-review/SKILL.md`.

Keep `AGENTS.md` updated with pipeline order, constraints, and external links.

## References

**Course / rubric**

- [Project 2 instructions (AI system)](https://csc-343.path.app/projects/project-2-ai-system/ai-system.project.md)
- [Project rubric](https://csc-343.path.app/projects/project-2-ai-system/ai-system.rubric.md)
- [Code elegance rubric](https://csc-343.path.app/rubrics/code-elegance.rubric.md)
- [Course schedule](https://csc-343.path.app/resources/course.schedule.md)

**APIs and libraries**

- [Alpha Vantage](https://www.alphavantage.co/) — news and sentiment endpoint (Module 4)
- **yfinance** — OHLCV for demos and backtests
- **NumPy, pandas, scikit-learn, matplotlib** — numerics, data, classifier, optional plots
- **pytest** — unit and integration tests
