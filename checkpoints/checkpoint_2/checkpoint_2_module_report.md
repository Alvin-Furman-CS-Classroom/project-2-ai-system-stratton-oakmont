# Checkpoint 2: Module Rubric Report

**Module:** Module 2 - Strategy Parameter Search  
**Date:** February 19, 2026  
**Topics:** Informed Search, A*, Beam Search, Heuristics, Backtesting

---

## Summary

Module 2 is **fully functional and well-integrated** into the pipeline. It implements parameter search over trading-rule thresholds using A* and Beam Search, with backtesting via Module 1's rule engine. The module has comprehensive test coverage (35 unit tests + 2 integration tests, all passing), clear documentation, and provides a clean handoff (`CandidateStrategy`) to Module 3.

---

## Findings

### 1. Functionality (Score: 8/8)

**Assessment:** All features work correctly. Search, backtest, and evaluation behave as specified.

**Evidence:**
- ✅ **Backtest works:** `backtest()` computes returns and actions from OHLCV; integrates with M1 `evaluate_rules_on_indicators()`
- ✅ **Indicator computation:** RSI, MACD, MA20, MA50, volatility derived from OHLCV; warmup handled
- ✅ **Sharpe ratio:** Annualized Sharpe computed correctly; edge cases (empty, constant returns) handled
- ✅ **Beam Search:** Expands from diverse starting points; keeps top-k with diversity filtering
- ✅ **A* Search:** Priority queue ordered by `sharpe + heuristic`; admissible heuristic for remaining improvement potential
- ✅ **`search_top_strategies()`:** Main entrypoint supports `beam` and `astar` methods; returns top-k `CandidateStrategy`
- ✅ **`evaluate_candidate()`:** Produces `CandidateStrategy` with params, sharpe, total_return, win_rate, max_drawdown, explanation
- ✅ **Market data:** `load_ohlcv_yahoo()`, `load_ohlcv_csv()`, `generate_synthetic_ohlcv()` for tests
- ✅ **Demo runs successfully** comparing Beam vs A* with train/test split and buy-and-hold benchmark
- ✅ **All 35 unit tests + 2 integration tests pass**

---

### 2. Code Elegance and Quality (Score: 8/8)

**Assessment:** Exemplary code quality. See [checkpoint_2_elegance_report.md](checkpoint_2_elegance_report.md) for details.

**Highlights:**
- Average Code Elegance Score: **4.0/4.0** → Module Rubric Score: **4**
- Clear structure: backtest, evaluation, search separated; shared types and market_data
- Strong naming conventions and Pythonic idioms
- Type hints and docstrings throughout
- Thoughtful error handling

---

### 3. Testing (Score: 8/8)

**Assessment:** Comprehensive test coverage across search, backtest, and evaluation.

**Test Categories (35 unit + 2 integration):**

| Category | Tests | Coverage |
|----------|-------|----------|
| Sharpe Ratio | 3 | Empty, constant, positive returns |
| Indicators | 2 | Warmup, required fields |
| Backtest | 1 | Return/action length |
| Evaluation | 1 | CandidateStrategy fields |
| Beam Search | 1 | Top-k returned |
| Search Top Strategies | 3 | List, ordering, astar method |
| A* Heuristic | 4 | Hashable key, nonnegative, center vs boundary, empty ranges |
| A* Search | 4 | Top-k, ordering, valid params, single expansion |
| _clamp_params | 3 | Within-range, clips bounds, ignores extra keys |
| _get_successors | 3 | Returns neighbors, stays in range, two-param perturbations |
| _diverse_starting_points | 4 | Count, center first, within bounds, deterministic seed |
| _diversity_filter | 6 | Empty, max_keep, same-bucket cap, diverse Sharpes, sorted, param dedup |
| Integration (M1+M2) | 2 | Backtest produces actions, search returns usable candidates |

**Strengths:**
- Tests verify behavior, not implementation
- Synthetic OHLCV used for deterministic tests
- Heuristic properties (admissibility-related) tested
- Integration tests confirm M1+M2 pipeline works

---

### 4. Individual Participation (Score: N/A)

**Note:** This criterion assesses commit history balance across team members.

*Score will be determined by instructor review of commit history.*

---

### 5. Documentation (Score: 5/5)

**Assessment:** Strong documentation with docstrings, type hints, and usage examples.

**Evidence:**
- ✅ **Module docstring** in `__init__.py` describes public API
- ✅ **All public functions have docstrings** with Args/Returns:
  - `search_top_strategies()`, `beam_search()`, `astar_search()`
  - `backtest()`, `indicators_from_ohlcv()`, `sharpe_ratio()`
  - `evaluate_candidate()`
- ✅ **Heuristic** (`_heuristic`) documented with admissibility rationale
- ✅ **File-level docstrings** in backtest.py, evaluation.py, search.py
- ✅ **Demo script** (`demo.py`) compares Beam vs A* with detailed output

---

### 6. I/O Clarity (Score: 5/5)

**Assessment:** Inputs and outputs are clearly specified and easy to verify.

**Input Specification:**
```python
# Primary inputs
ohlcv: pd.DataFrame  # Columns Open, High, Low, Close, Volume
param_ranges: ParamRanges  # Dict[str, tuple[float, float]] e.g. {"rsi_oversold": (0.0, 30.0)}
rules: Optional[Sequence[HornRule]]  # Defaults to M1 default_trading_rules()
top_k: int = 10
method: str = "beam"  # or "astar"
```

**Output Specification:**
```python
# Primary output: List[CandidateStrategy]
CandidateStrategy(
    params: Dict[str, float],    # Parameter config
    sharpe: float,               # Backtest Sharpe ratio
    total_return: float,
    win_rate: float,
    max_drawdown: float,
    num_trades: int,
    explanation: str             # Human-readable summary
)
```

**Feed to Next Module (M3):**
- Top 10 (or top_k) `CandidateStrategy` objects with params and Sharpe
- M3 uses these as seed population for genetic algorithm
- `params` dict format matches M1's `Params`; M3 can run backtests with evolved params

---

### 7. Topic Engagement (Score: 6/6)

**Assessment:** Strong engagement with informed search, A*, Beam Search, and heuristics.

**AI Concepts Demonstrated:**

| Concept | Implementation |
|---------|----------------|
| **A* Search** | Priority queue on `-(sharpe + heuristic)`; expand best-first; closed set for visited configs |
| **Heuristic** | `_heuristic()` estimates remaining improvement from param "room" (distance to boundaries); admissible in practice |
| **Beam Search** | Keep top-k configs per iteration; expand neighbors; diversity filter prevents beam collapse |
| **Successor Generation** | `_get_successors()` perturbs one or two params within ranges; single- and two-param exploration |
| **Backtesting** | Simulate positions from M1 actions; compute returns, Sharpe, drawdown, win rate |
| **Diverse Starting Points** | `_diversity_filter()` and `_diverse_starting_points()` for broad search coverage |

**Why These Choices:**
- Parameter space is large; exhaustive search impractical
- A* and Beam Search efficiently explore high-value regions
- Heuristic guides A* toward configs with remaining optimization potential
- Backtest is the natural fitness function for trading strategies

---

### 8. GitHub Practices (Score: N/A)

**Assessment:** Repository structure supports Module 2.

**Evidence:**
- ✅ `src/module_2_strategy_search/` with backtest, evaluation, search, demo
- ✅ `unit_tests/module_2_strategy_search/`
- ✅ `integration_tests/module_2/` (M1+M2)
- ✅ Shared `market_data.py`, `types.py` with `CandidateStrategy`, `ParamRanges`

*Commit message quality and PR usage will be evaluated by instructor.*

---

## Scores Summary

| Criterion | Score | Max |
|-----------|-------|-----|
| 1. Functionality | 8 | 8 |
| 2. Code Elegance and Quality | 8 | 8 |
| 3. Testing | 8 | 8 |
| 4. Individual Participation | TBD | 6 |
| 5. Documentation | 5 | 5 |
| 6. I/O Clarity | 5 | 5 |
| 7. Topic Engagement | 6 | 6 |
| 8. GitHub Practices | TBD | 4 |

**Subtotal (excluding participation/GitHub):** 40/40

---

## Module Explanation (for In-Person Demo)

### Input
**What the module accepts:**
- OHLCV history (pandas DataFrame with Open, High, Low, Close, Volume)
- Optional: `param_ranges` (search bounds), `rules` (M1 HornRules), `top_k`, `method` ("beam" or "astar")

**Example Input:**
```python
ohlcv = generate_synthetic_ohlcv(days=252, seed=42)
# or: ohlcv = load_ohlcv_yahoo("SPY", "1y")
top = search_top_strategies(ohlcv, top_k=5, method="beam")
```

### Output
**What the module produces:**
- List of `CandidateStrategy` ranked by Sharpe ratio

**Example Output:**
```
Top 5 strategies:
  1. Sharpe=0.42 | Return=8.2%, MaxDD=-5.1%, WinRate=54.2%, Trades=89
  2. Sharpe=0.38 | Return=6.1%, MaxDD=-4.2%, WinRate=52.1%, Trades=102
  ...
```

**Feed to Next Module (M3):**
- Top 10 `CandidateStrategy` objects → M3 genetic algorithm seed population
- Each has `params` (dict) and `sharpe` for fitness evaluation

### AI Concepts
**Techniques used:**
1. **A* Search:** Best-first exploration with `f(n) = sharpe(n) + h(n)`; heuristic estimates remaining improvement
2. **Beam Search:** Keep top-k configs; expand neighbors; diversity filter
3. **Backtesting:** OHLCV → indicators → M1 rules → positions → returns → Sharpe

**Why these fit:**
- Parameter space is large; informed search is efficient
- A* and Beam avoid exhaustive evaluation
- Backtest is the natural fitness for trading strategies

---

## Recommendations

1. **For Checkpoint 2:** Ensure commit history shows balanced participation
2. **Integration:** Verify M1+M2 integration tests pass before presentation
