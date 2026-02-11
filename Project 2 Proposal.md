# Intelligent Trading Agent: Strategy Discovery Through Search and Adaptive Risk Management

## Overview

This system helps users find profitable trading strategies for stocks and ETFs and manages risk when executing trades. It uses five AI modules that work together, each solving a specific part of the problem.

**Phase 1: Finding Good Strategies** — The system needs rules for when to buy and sell. Module 1 represents these rules using propositional logic—simple "if-then" statements like "if price momentum is high AND volatility is low, then buy." Module 2 searches through thousands of possible rule variations to find ones that look promising, using A* and Beam Search to avoid testing every single combination. Module 3 takes the best candidates and improves them further using genetic algorithms—a technique that mimics evolution by combining successful strategies and introducing small changes over many generations.

**Phase 2: Trading Smartly** — Once good strategies exist, the system needs to apply them wisely. Module 4 uses supervised learning to classify market sentiment from news data, helping select the right strategy for current conditions. Module 5 uses reinforcement learning to decide position sizes, learning which allocations lead to the best long-term results.

Together, these modules form a complete pipeline: discover rules, optimize them, understand market sentiment, and manage risk.

## Team

- Casen Shoemake
- Kyler Bailey
- Collin Riddle

## Proposal

The trading strategy problem naturally decomposes into five AI challenges, each addressed by a different technique:

1. **Propositional Logic** — Trading rules are inherently logical: "IF condition THEN action." Representing them formally enables explainable, traceable decisions.
2. **Informed Search** — Finding good strategy parameters is a search problem. With thousands of possible configurations, exhaustive testing is impractical. A* and Beam Search efficiently explore the space using heuristics.
3. **Genetic Algorithms** — Strategies can be improved through evolution. GA excels at optimization problems where the search space is large and the fitness function (backtested returns) is well-defined.
4. **Supervised Learning** — Market sentiment varies, and different strategies suit different conditions. A logistic regression classifier learns to recognize sentiment regimes from news features, predicting which market conditions will follow.
5. **Reinforcement Learning** — Position sizing is a sequential decision problem with delayed rewards. RL learns optimal risk management through experience, adapting to context rather than following fixed rules.

The modules follow both logical dependency (rules → parameters → evolution → classification → execution) and course schedule (early topics first). This ensures the system can be built incrementally, with each module tested before the next checkpoint.

## Module Plan

| Module | Topic(s) | Inputs | Outputs | Depends On | Checkpoint |
| ------ | -------- | ------ | ------- | ---------- | ---------- |
| 1: Trading Rule Knowledge Base | Propositional Logic (Knowledge Bases, Inference, Forward Chaining) | Market indicators (RSI, MACD, MA20, MA50, Volume) and trading rules in CNF format | Trading action (BUY/SELL/HOLD), fired rules, inference chain | None | CP1 (Feb 11) |
| 2: Strategy Parameter Search | Informed Search (A*, Beam Search, Heuristics) | Parameter ranges defining search space, historical market data | Top 10 candidate parameter configurations ranked by Sharpe ratio | Module 1 | CP1 (Feb 11) |
| 3: Strategy Evolution Engine | Advanced Search (Genetic Algorithms) | Top 10 candidates from Module 2, historical data, GA parameters | Top 5 evolved strategies with performance metrics (Sharpe, return, win rate, max drawdown) | Module 2 | CP2 (Feb 26) |
| 4: Market Sentiment Classifier | Supervised Learning (Logistic Regression, Classification) | News sentiment data from Alpha Vantage API (scores, volume, topics, trend) | Sentiment regime (Bullish/Bearish/Neutral), confidence, recommended strategy | Module 3 | CP3 (Mar 19) |
| 5: Adaptive Position Sizing Agent | Reinforcement Learning (MDP, Q-Learning) | Sentiment regime from Module 4, strategy metrics, volatility, capital | Position size (1-15%), Q-values, reasoning, risk assessment | Module 4 | CP4 (Apr 2) |

## Repository Layout

```
project-2-ai-system-stratton-oakmont/
├── src/                              # main system source code
│   ├── module_1_knowledge_base/      # propositional logic trading rules
│   ├── module_2_strategy_search/     # A* and Beam Search optimization
│   ├── module_3_evolution/           # genetic algorithm strategy evolution
│   ├── module_4_sentiment/           # sentiment classification
│   ├── module_5_position_sizing/     # RL position sizing agent
│   └── shared/                       # shared utilities
├── unit_tests/                       # unit tests (parallel structure to src/)
├── integration_tests/                # integration tests (new folder for each module)
├── data/                             # market data and datasets
├── .claude/skills/code-review/SKILL.md  # rubric-based agent review
├── AGENTS.md                         # instructions for your LLM agent
└── README.md                         # system overview and checkpoints
```

## Setup

List dependencies, setup steps, and any environment variables required to run the system.

## Running

Provide commands or scripts for running modules and demos.

## Testing

**Unit Tests** (`unit_tests/`): Mirror the structure of `src/`. Each module should have corresponding unit tests.

**Integration Tests** (`integration_tests/`): Create a new subfolder for each module beyond the first, demonstrating how modules work together.

Provide commands to run tests and describe any test data needed.

## Checkpoint Log

| Checkpoint | Date | Modules Included | Status | Evidence |
| ---------- | ---- | ---------------- | ------ | -------- |
| 1 | Feb 11 | Module 1, Module 2 |  |  |
| 2 | Feb 26 | Module 3 |  |  |
| 3 | Mar 19 | Module 4 |  |  |
| 4 | Apr 2 | Module 5 |  |  |

## Required Workflow (Agent-Guided)

Before each module:

1. Write a short module spec in this README (inputs, outputs, dependencies, tests).
2. Ask the agent to propose a plan in "Plan" mode.
3. Review and edit the plan. You must understand and approve the approach.
4. Implement the module in `src/`.
5. Unit test the module, placing tests in `unit_tests/` (parallel structure to `src/`).
6. For modules beyond the first, add integration tests in `integration_tests/` (new subfolder per module).
7. Run a rubric review using the code-review skill at `.claude/skills/code-review/SKILL.md`.

Keep `AGENTS.md` updated with your module plan, constraints, and links to APIs/data sources.

## References

**Market Indicators:**
- **RSI (Relative Strength Index):** A 0-100 score measuring if an asset is overbought (>70) or oversold (<30)
- **MACD (Moving Average Convergence Divergence):** A momentum indicator showing trend direction and strength
- **MA20/MA50 (Moving Averages):** Average prices over the last 20 or 50 days, used to identify trends
- **Volume:** How much of the asset was traded, indicating market activity
- **Volatility:** How much prices fluctuate—high volatility means bigger price swings

**Sentiment Features (from API):**
- **Sentiment Score:** Numerical value from -1 (very bearish) to 1 (very bullish)
- **Article Volume:** Number of news articles in a time period—high volume often signals important events
- **Sentiment Trend:** Whether sentiment is improving or declining over recent periods
- **Topic Distribution:** What subjects articles cover (earnings, economy, mergers, etc.)

**Performance Metrics:**
- **Sharpe ratio:** Return per unit of risk (higher = better risk-adjusted performance)
- **Total return:** Overall profit percentage
- **Win rate:** Percentage of trades that were profitable
- **Max drawdown:** Largest peak-to-trough loss (measures worst-case scenario)

**APIs & Libraries:**
- Alpha Vantage API (news sentiment data)
