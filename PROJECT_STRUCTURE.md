# Project Structure & Group Guide

**Intelligent Trading Agent: Strategy Discovery Through Search and Adaptive Risk Management**

This document gives your group a clear structure to complete the project: module breakdown, milestones, suggested ownership, handoffs, and workflow.

---

## 1. Overview

| Item | Details |
|------|---------|
| **System** | Intelligent Trading Agent (5 modules) |
| **Phase 1** | Module 1–3: Discover and optimize trading rules |
| **Phase 2** | Module 4–5: Classify sentiment, size positions, execute |
| **Checkpoints** | CP1 (Feb 11), CP2 (Feb 26), CP3 (Mar 19), CP4 (Apr 2) |

---

## 2. Repository Layout

```
project-2-ai-system-stratton-oakmont/
├── src/
│   ├── module_1_knowledge_base/     # Propositional logic, CNF rules, inference
│   ├── module_2_strategy_search/    # A*, Beam Search, parameter search
│   ├── module_3_evolution/          # Genetic algorithm, fitness, backtest
│   ├── module_4_sentiment/          # Logistic regression, Alpha Vantage API
│   ├── module_5_position_sizing/    # RL (MDP, Q-learning), position %
│   ├── shared/                      # Data types, indicators, market data helpers
│   └── run.py                       # Optional: end-to-end demo
├── unit_tests/
│   ├── module_1_knowledge_base/
│   ├── module_2_strategy_search/
│   ├── module_3_evolution/
│   ├── module_4_sentiment/
│   └── module_5_position_sizing/
├── integration_tests/
│   ├── module_2/                    # 1 + 2
│   ├── module_3/                    # 1 + 2 + 3
│   ├── module_4/                    # 1–4
│   └── module_5/                    # 1–5 full pipeline
├── data/                            # Optional: sample market data, cached API responses
├── .claude/skills/code-review/
├── AGENTS.md
├── README.md
├── Project 2 Proposal.md
└── PROJECT_STRUCTURE.md             # this file
```

---

## 3. Module Plan (from Proposal)

| Module | Topic(s) | Inputs | Outputs | Depends On | Checkpoint |
|--------|----------|--------|---------|------------|------------|
| **1** | Propositional Logic (KB, inference, CNF) | Market indicators (RSI, MACD, MA20, MA50, Volume), CNF rules | BUY/SELL/HOLD, fired rules, inference chain | — | CP1 (Feb 11) |
| **2** | Informed Search (A*, Beam, heuristics) | Parameter ranges, historical data | Top 10 candidates by Sharpe, params + explanation | 1 | CP1 (Feb 11) |
| **3** | Genetic Algorithms | Top 10 from M2, full history, GA params | Top 5 evolved strategies, metrics, evolution summary | 1, 2 | CP2 (Feb 26) |
| **4** | Supervised Learning (logistic regression) | Alpha Vantage sentiment API data | Regime (Bullish/Bearish/Neutral), confidence, top headlines, strategy recommendation | 1, 2, 3 | CP3 (Mar 19) |
| **5** | Reinforcement Learning (MDP, Q-learning) | Sentiment + confidence, strategy metrics, volatility, capital | Position % (1/5/10/15), Q-values, reasoning, risk assessment | 1–4 | CP4 (Apr 2) |

---

## 4. Module-by-Module Breakdown

### Module 1: Trading Rule Knowledge Base  
**Owner:** _Assign one person_  
**Checkpoint:** CP1 (Feb 11)

| Item | Description |
|------|-------------|
| **Input** | Current indicators (RSI, MACD, MA20, MA50, Volume), CNF rules |
| **Output** | Action (BUY/SELL/HOLD), fired rules, inference chain |
| **Deliverables** | `src/module_1_knowledge_base/` (KB, forward chaining, rule application), `unit_tests/module_1_*` |
| **Handoff** | Defines rule structure and interfaces that M2/M3 will optimize. Expose a clean API for “evaluate rules on indicators → action + trace.” |

---

### Module 2: Strategy Parameter Search  
**Owner:** _Assign one person_  
**Checkpoint:** CP1 (Feb 11)

| Item | Description |
|------|-------------|
| **Input** | Parameter ranges (e.g. RSI 20–40), historical market data |
| **Output** | Top 10 parameter configs ranked by Sharpe; each with params, score, explanation |
| **Deliverables** | `src/module_2_strategy_search/` (A*, Beam Search, heuristic, evaluation), `unit_tests/`, `integration_tests/module_2/` (M1 + M2) |
| **Handoff** | Top 10 candidates (config + Sharpe) → Module 3 as seed population. Agree on a **data format** (e.g. JSON/dict) for “candidate strategy.” |

---

### Module 3: Strategy Evolution Engine  
**Owner:** _Assign one person_  
**Checkpoint:** CP2 (Feb 26)

| Item | Description |
|------|-------------|
| **Input** | Top 10 from M2, full history, GA params (pop=20, gen=100, mutation=0.1) |
| **Output** | Top 5 evolved strategies + metrics (Sharpe, return, win rate, max drawdown), evolution summary |
| **Deliverables** | `src/module_3_evolution/` (GA, fitness=backtest), `unit_tests/`, `integration_tests/module_3/` (M1+M2+M3) |
| **Handoff** | Top 5 strategies + full metrics → Module 4. Define **strategy schema** (params, metrics, ID) so M4 can “recommend best strategy for regime.” |

---

### Module 4: Market Sentiment Classifier  
**Owner:** _Assign one person_  
**Checkpoint:** CP3 (Mar 19)

| Item | Description |
|------|-------------|
| **Input** | Alpha Vantage Market News & Sentiment API (scores, labels, volume, topics, trend) |
| **Output** | Regime (Bullish/Bearish/Neutral), confidence, top headlines, recommended strategy from M3 |
| **Deliverables** | `src/module_4_sentiment/` (feature engineering, logistic regression, API client), `unit_tests/`, `integration_tests/module_4/` |
| **Handoff** | Regime + confidence + selected strategy → Module 5. Document how M4 maps “regime + history” → one of the top 5 strategies. |

---

### Module 5: Adaptive Position Sizing Agent  
**Owner:** _Assign one person_  
**Checkpoint:** CP4 (Apr 2)

| Item | Description |
|------|-------------|
| **Input** | Sentiment regime & confidence (M4), selected strategy metrics (M3), volatility, capital (e.g. $10k) |
| **Output** | Position % (1/5/10/15), Q-values, reasoning, risk assessment |
| **Deliverables** | `src/module_5_position_sizing/` (MDP, Q-learning, state/action/reward), `unit_tests/`, `integration_tests/module_5/` (full pipeline) |
| **Handoff** | Final output is the system’s recommendation; no further modules. |

---

## 5. Suggested Division of Labor (3-Person Team)

| Person | Primary modules | Supporting role |
|--------|-----------------|-----------------|
| **A** | Module 1, Module 4 | Shared code (`shared/`), data/API helpers |
| **B** | Module 2, Module 5 | Integration tests, CI/test runs |
| **C** | Module 3 | Docs, checkpoint log, rubric reviews |

**2-person team:** Pair M1+M2 (both CP1), one does M3, one does M4+M5, and share `shared/` and integration work.

---

## 6. Milestone Timeline

| Milestone | Date | Modules | Goals |
|-----------|------|---------|-------|
| **CP1** | Feb 11 | 1, 2 | M1 KB + inference working; M2 search producing top 10; `integration_tests/module_2` passing |
| **CP2** | Feb 26 | 3 | GA evolving strategies; top 5 with metrics; `integration_tests/module_3` passing |
| **CP3** | Mar 19 | 4 | Sentiment classifier trained; regime + strategy recommendation; `integration_tests/module_4` passing |
| **CP4** | Apr 2 | 5 | RL position sizing; full pipeline run; `integration_tests/module_5` passing; rubric review done |

---

## 7. Workflow (Per Module)

1. **Spec** – Write a short spec in `README.md` (inputs, outputs, dependencies, tests).
2. **Plan** – Use the agent to propose a plan; review and approve.
3. **Implement** – Code in `src/<module>/`; use `shared/` for common types and helpers.
4. **Unit test** – Add tests in `unit_tests/<module>/`; mirror `src/` structure.
5. **Integrate** – For M2–M5, add `integration_tests/module_<n>/` and get them passing.
6. **Review** – Run the code-review skill (`.claude/skills/code-review/SKILL.md`), fix gaps, update `AGENTS.md` if needed.

---

## 8. Shared Components (`src/shared/`)

Define these early so all modules use the same formats:

- **Market indicators:** RSI, MACD, MA20, MA50, Volume, Volatility (computation or loading).
- **Data types:** `TradingAction`, `MarketIndicators`, `CandidateStrategy`, `EvolvedStrategy`, `SentimentRegime`, etc.
- **Market data:** Helpers to load historical OHLCV (Yahoo Finance / Alpha Vantage) for backtesting.
- **Config:** Parameter ranges, API keys (env vars, never committed), GA defaults.

---

## 9. Data & APIs

| Source | Use | Notes |
|--------|-----|-------|
| **Yahoo Finance / Alpha Vantage** | OHLCV, indicators | Free; use for backtesting and indicator computation |
| **Alpha Vantage Market News & Sentiment** | M4 features | Free tier ≈25 req/day; cache responses under `data/` for tests |

---

## 10. Handoff Contract (Summary)

| From → To | Artifact | Format |
|-----------|----------|--------|
| M1 → M2, M3 | Rule structure, “evaluate rules on indicators” API | Code + types in `shared/` |
| M2 → M3 | Top 10 candidates | List of `{params, sharpe, explanation}` |
| M3 → M4 | Top 5 strategies | List of `{params, metrics, id}`; M4 picks by regime |
| M4 → M5 | Regime, confidence, selected strategy | Same as M4 output; M5 consumes directly |

---

## 11. Quick Reference

- **Proposal:** `Project 2 Proposal.md`
- **Module plan & checkpoints:** `README.md` (and table above)
- **Agent workflow & rubric:** `AGENTS.md`, `.claude/skills/code-review/SKILL.md`
- **Course links:** See `AGENTS.md` (project instructions, rubric, schedule).

Use this structure to assign modules, track checkpoints, and keep handoffs clear. Update `README.md` and `AGENTS.md` as you lock in specs and ownership.
