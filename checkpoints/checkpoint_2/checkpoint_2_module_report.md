# Checkpoint 2: Module Rubric Report

**Module:** Module 2 – Strategy Parameter Search  
**Date:** February 26, 2026  
**Topics:** Informed Search, A*, Beam Search, Heuristics, Backtesting

---

## Summary

Module 2 is **fully implemented, well-tested, and cleanly integrated** with the rest of the system. It performs parameter search over trading-rule thresholds using Beam Search and A* Search, backtests each configuration using Module 1’s knowledge base, and returns ranked `CandidateStrategy` objects suitable as seed inputs for Module 3. There are **43 unit tests** (`unit_tests/module_2_strategy_search/test_strategy_search.py`) and **5 integration tests** (`integration_tests/module_2/test_m1_m2_integration.py`), all passing, which collectively validate search behavior, backtesting, error handling, and the handoff contract to later modules. Overall, Module 2 is ready for Checkpoint 2 submission.

---

## Findings

### 1. Functionality (Score: 8/8)

**Assessment:** All required features behave correctly and match the intended design.

**Evidence:**
- **Backtesting and indicators:** `backtest()` in `backtest.py` converts OHLCV data into indicators via `indicators_from_ohlcv()` (RSI, MACD line, MA20, MA50, volatility) and simulates positions and returns; `WARMUP_BARS` ensures indicators are valid before trading.
- **Sharpe ratio:** `sharpe_ratio()` computes annualized Sharpe for daily returns, handling empty or zero-variance returns by safely returning 0.0.
- **Search algorithms:** `beam_search()` and `astar_search()` in `search.py` explore parameter space using successors from `_get_successors()` and diverse seeds from `_diverse_starting_points()`, returning top strategies respecting the `top_k` budget.
- **Heuristic:** `_heuristic()` provides a nonnegative, range-aware estimate of remaining improvement potential; it is designed to be conservative and is tested for key properties.
- **Diversity:** `_diversity_filter()` enforces diversity among strategies by bucketing Sharpe values and deduplicating parameter sets, preventing the search from collapsing to nearly identical configurations.
- **Main entrypoint:** `search_top_strategies()` supports `"beam"` and `"astar"` methods, defaults to well-chosen `DEFAULT_PARAM_RANGES`, and raises a clear `ValueError` for unknown methods.
- **Integration with Module 1:** `backtest()` calls `evaluate_rules_on_indicators()` from Module 1 and respects its `TradingAction` outputs (`BUY`, `SELL`, `HOLD`), ensuring a coherent cross-module pipeline.
- **Output contract:** `evaluate_candidate()` in `evaluation.py` produces `CandidateStrategy` objects with `params`, `sharpe`, `total_return`, `win_rate`, `max_drawdown`, `num_trades`, and a concise `explanation` string.

---

### 2. Code Elegance and Quality (Score: 8/8)

**Assessment:** Code quality is excellent across naming, structure, style, and error handling.

**Evidence:**
- See the detailed [Checkpoint 2 Code Elegance Report](checkpoint_2_elegance_report.md), which reviews Module 2 and its shared support code against the Code Elegance rubric.
- In summary, the module earns **4.0 / 4.0** on elegance (mapped to **4/4** in this rubric), with strong scores in naming, modularity, control flow, and Pythonic idioms, and no outstanding issues identified for this checkpoint.

---

### 3. Testing (Score: 8/8)

**Assessment:** Testing is comprehensive and exercises both normal behavior and edge cases.

**Evidence (43 unit tests + 5 integration tests):**

| Category                  | Tests | Coverage                                                                        |
|---------------------------|-------|---------------------------------------------------------------------------------|
| Sharpe Ratio              | 3     | Empty input, constant returns, positive-return scenarios                        |
| Indicators                | 2     | Warmup handling and required indicator fields on valid rows                     |
| Backtest                  | 5     | Length of returns/actions, non-DataFrame input, missing columns, row count, minimal valid OHLCV |
| Evaluate Candidate        | 3     | `CandidateStrategy` structure, all-HOLD scenario, finiteness of metrics         |
| Beam Search               | 1     | Returns at most `top_k` strategies                                              |
| Search Top Strategies     | 4     | Output type/ordering, A* routing, invalid method handling                       |
| A* Heuristic              | 4     | Hashable/deterministic keys, nonnegativity, center vs. boundary behavior, empty ranges |
| A* Search                 | 4     | Top-k cap, ordering by Sharpe, parameter validity, bounded expansion behavior   |
| Beam vs A* Comparison     | 1     | Both search methods produce non-empty results on the same data                  |
| `_clamp_params`           | 3     | In-range values unchanged, clamping to bounds, ignoring extra keys              |
| `_get_successors`         | 3     | Neighbor generation, range-clamping, two-parameter perturbations                |
| `_diverse_starting_points`| 5     | Count, center point, bounds, determinism, structure of sampled points          |
| `_diversity_filter`       | 6     | Empty input, `max_keep` enforcement, Sharpe bucket caps, diversity, ordering, deduplication |
| Backtest Validation       | 4     | Type, schema, row-count validation and minimal valid OHLCV behavior             |
| Integration (M1 + M2)     | 5     | Valid actions, usable `CandidateStrategy` for M3, finite returns, params within ranges |

**Strengths:**
- Tests emphasize behavior and contracts rather than implementation details.
- Synthetic OHLCV data via `generate_synthetic_ohlcv()` makes tests deterministic and fast.
- Negative tests cover invalid input types, missing columns, too-few rows, and unsupported search methods.
- Integration tests validate the end-to-end pipeline and the handoff format for Module 3.

---

### 4. Individual Participation (Score: TBD/6)

**Assessment:** This criterion depends on commit history and division of work across team members.

**Notes:**
- The code and tests themselves reflect thoughtful design, but participation scoring requires instructor review of Git history and is not inferred here.

---

### 5. Documentation (Score: 5/5)

**Assessment:** Documentation is clear and sufficient for both users and maintainers.

**Evidence:**
- Module-level docstrings (e.g., in `__init__.py`, `backtest.py`, `search.py`, `evaluation.py`) describe each file’s role and its public API.
- Public functions (`search_top_strategies`, `backtest`, `evaluate_candidate`, `sharpe_ratio`) include docstrings with Args/Returns and short behavioral descriptions.
- `_heuristic()` explains the rationale for its design, including range-based “room to move” intuition and conservativeness.
- `DEFAULT_PARAM_RANGES` is annotated with inline comments tying ranges to standard technical-analysis conventions (e.g., RSI oversold/overbought bands).
- `demo.py` shows end-to-end usage with real or synthetic data, including a buy-and-hold benchmark and train/test split, which is useful for demonstration and debugging.

---

### 6. I/O Clarity (Score: 5/5)

**Assessment:** Inputs and outputs are clearly defined, strongly typed, and easy to verify.

**Primary Inputs:**

```python
ohlcv: pd.DataFrame         # Columns: Open, High, Low, Close, Volume
param_ranges: ParamRanges   # Dict[str, tuple[float, float]] (optional; defaults provided)
rules: Optional[Sequence[HornRule]] = None  # Defaults to Module 1's default_trading_rules()
top_k: int = 10
method: str = "beam"        # "beam" or "astar"
```

**Primary Outputs:**

```python
CandidateStrategy(
    params: Dict[str, float],     # Tuned parameter configuration
    sharpe: float,                # Backtest Sharpe ratio
    total_return: float,
    win_rate: float,
    max_drawdown: float,
    num_trades: int,
    explanation: str,             # Compact human-readable summary
)
```

**Notes:**
- `_validate_ohlcv()` makes the expected OHLCV schema explicit and enforces it with informative error messages.
- Integration tests confirm that `CandidateStrategy` includes exactly the fields expected by Module 3’s genetic algorithm.

---

### 7. Topic Engagement (Score: 6/6)

**Assessment:** Module 2 shows strong engagement with informed search and AI concepts.

**AI Concepts Demonstrated:**

| Concept             | Implementation                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| A* Search           | `astar_search()` uses a priority queue on `-(sharpe + heuristic)` with open/closed sets. |
| Heuristic Design    | `_heuristic()` estimates remaining improvement based on distance to parameter bounds, scaled by Sharpe magnitude, and is tested for key properties. |
| Beam Search         | `beam_search()` keeps a beam of top configurations per iteration and applies diversity filtering. |
| Successor Generation| `_get_successors()` produces single- and two-parameter perturbations within valid ranges. |
| Backtesting         | `backtest()` transforms OHLCV into indicators, actions, positions, and returns, then `sharpe_ratio()` summarizes performance. |
| Diversity Heuristics| `_diverse_starting_points()` and `_diversity_filter()` encourage exploration of multiple high-quality regions in parameter space. |

**Rationale:**
- The parameter search space is large and continuous; exhaustive search is infeasible.
- A* and Beam Search are natural choices to focus computation on promising regions while maintaining exploration.
- Backtesting provides a task-appropriate fitness function for trading strategies, tightly coupling AI search with domain evaluation.

---

### 8. GitHub Practices (Score: TBD/4)

**Assessment:** Repository structure and test layout meet project expectations for this module.

**Evidence:**
- `src/module_2_strategy_search/` contains implementation files and a demo script.
- `unit_tests/module_2_strategy_search/` and `integration_tests/module_2/` mirror the module and validate both isolated behavior and cross-module integration.
- Shared functionality (`src/shared/`) is imported consistently rather than duplicated.

**Notes:** Detailed scoring for commit messages, pull requests, and division of labor is left to instructor review of the Git history.

---

## Scores Summary

| Criterion                    | Score | Max |
|-----------------------------|-------|-----|
| 1. Functionality            | 8     | 8   |
| 2. Code Elegance and Quality| 8     | 8   |
| 3. Testing                  | 8     | 8   |
| 4. Individual Participation | TBD   | 6   |
| 5. Documentation            | 5     | 5   |
| 6. I/O Clarity              | 5     | 5   |
| 7. Topic Engagement         | 6     | 6   |
| 8. GitHub Practices         | TBD   | 4   |

**Subtotal (excluding participation and GitHub practices):** **40 / 40**

---

## Module Explanation (for In-Person Demo)

### Inputs

**What the module accepts:**
- OHLCV price/volume history for a single asset as a `pandas.DataFrame`.
- Optional parameter ranges (`ParamRanges`) defining lower/upper bounds for RSI/MACD/MA/volume/volatility thresholds.
- Optional custom trading rules from Module 1; defaults are provided if not supplied.
- Configuration knobs: `top_k`, search `method` (`"beam"` or `"astar"`), and search budgets inside the algorithms.

**Example:**

```python
from src.module_2_strategy_search import search_top_strategies
from src.shared.market_data import generate_synthetic_ohlcv

ohlcv = generate_synthetic_ohlcv(days=252, seed=42)
top_strategies = search_top_strategies(ohlcv, top_k=5, method="beam")
```

### Outputs

**What the module produces:**
- A list of `CandidateStrategy` objects, ordered by Sharpe ratio (best first), each encapsulating:
  - Tuned parameter values (`params`).
  - Backtest performance metrics (`sharpe`, `total_return`, `win_rate`, `max_drawdown`, `num_trades`).
  - A short textual summary (`explanation`).

**Example console output (from `demo.py`):**

```text
BEAM SEARCH — top 5 strategies
  1. Sharpe=+0.42  Return=+8.2%  MaxDD=-5.1%  WinRate=54%  Trades=89  [RSI(28/72), MACD_eps=0.0123, MA_margin=0.0234]
  2. ...
```

### Relationship to Other Modules

- **From Module 1 (Knowledge Base):** Module 2 relies on Module 1’s Horn rules and indicator evaluation (`evaluate_rules_on_indicators`) to generate trading actions from indicators.
- **To Module 3 (Search/Optimization):** The top `CandidateStrategy` objects form a natural **seed population** for Module 3’s genetic algorithm or other higher-level search, with `params` and `sharpe` providing both configuration and fitness.

### AI Concepts to Highlight in a Demo

- **Informed Search:** Explain how A* and Beam Search prioritize promising parameter regions using heuristics and diversity filters rather than brute force.
- **Backtesting as Evaluation:** Show how historical OHLCV data is converted into returns and risk-adjusted metrics (Sharpe ratio) to evaluate a strategy.
- **Generalization:** Using train/test splits (as in `demo.py`), demonstrate that top strategies on training data may or may not generalize to test data, motivating Module 3’s broader exploration.

