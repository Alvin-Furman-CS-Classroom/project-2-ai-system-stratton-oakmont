"""Interactive full pipeline: ticker -> M1 -> M2 -> M3 -> M4.

Prompts for a ticker symbol, fetches real OHLCV from Yahoo Finance,
then runs every module in sequence and prints results.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.module_1_knowledge_base import (
    evaluate_rules_on_indicators,
    indicators_to_facts,
)
from src.module_2_strategy_search import search_top_strategies
from src.module_2_strategy_search.backtest import (
    WARMUP_BARS,
    backtest,
    indicators_from_ohlcv,
    sharpe_ratio,
)
from src.module_3_evolution import GAConfig
from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    finalize_unified_selection,
    gather_unified_candidate_pools,
)
from src.module_4_sentiment.demo_visualization import (
    build_sentiment_demo_figure,
    save_figure_png,
    write_html_report,
)
from src.module_4_sentiment.pipeline import analyze_market_sentiment
from src.shared.market_data import generate_synthetic_ohlcv, load_ohlcv_yahoo


DIVIDER = "=" * 64
THIN_DIVIDER = "-" * 64


def _prompt_ticker() -> str:
    while True:
        raw = input("\nEnter a ticker symbol (e.g. AAPL, TSLA, SPY): ").strip().upper()
        if raw:
            return raw
        print("  Ticker cannot be empty. Try again.")


def _prompt_period() -> str:
    valid = ("1mo", "3mo", "6mo", "1y", "2y", "5y")
    default = "1y"
    raw = input(f"Historical period [{default}] ({', '.join(valid)}): ").strip().lower()
    if raw in valid:
        return raw
    if raw == "":
        return default
    print(f"  Unrecognised period '{raw}', using {default}.")
    return default


def _load_ohlcv(symbol: str, period: str) -> pd.DataFrame:
    print(f"\n  Fetching {symbol} ({period}) from Yahoo Finance ...")
    try:
        ohlcv = load_ohlcv_yahoo(symbol=symbol, period=period)
        print(f"  Loaded {len(ohlcv)} bars  ({ohlcv.index[0].date()} to {ohlcv.index[-1].date()})")
        return ohlcv
    except Exception as exc:
        print(f"  Could not fetch {symbol}: {exc}")
        print("  Falling back to synthetic data (252 days).")
        return generate_synthetic_ohlcv(days=252, seed=42)


# ── Module 1 ─────────────────────────────────────────────────────────────────

def run_module_1(ohlcv: pd.DataFrame, symbol: str) -> None:
    print(f"\n{DIVIDER}")
    print("MODULE 1 - Knowledge Base: Rule-Based Inference on Latest Bar")
    print(DIVIDER)

    indicators_list = indicators_from_ohlcv(ohlcv)
    latest = indicators_list[-1]
    if latest is None:
        print("  Not enough data for indicator computation on the last bar.")
        return

    print(f"  Symbol:      {symbol}")
    print(f"  RSI:         {latest.rsi:.2f}")
    print(f"  MACD:        {latest.macd:.4f}")
    print(f"  MA20:        {latest.ma20:.2f}")
    print(f"  MA50:        {latest.ma50:.2f}")
    print(f"  Volume:      {latest.volume:,.0f}")
    if latest.volatility is not None:
        print(f"  Volatility:  {latest.volatility:.4f}")

    facts = indicators_to_facts(latest)
    true_facts = [k for k, v in facts.items() if v]
    print(f"\n  Active facts: {true_facts}")

    result = evaluate_rules_on_indicators(latest)
    print(f"  Action:       {result.action.value}")
    print(f"  Fired rules:  {result.fired_rules}")
    if result.inference_chain:
        print("  Inference chain:")
        for step in result.inference_chain:
            premises = ", ".join(
                f"{'NOT ' if lit.negated else ''}{lit.symbol}"
                for lit in step.supporting_literals
            )
            print(f"    {step.rule_id}: {premises} -> {step.added_fact}")


# ── Module 2 ─────────────────────────────────────────────────────────────────

def run_module_2(ohlcv_train: pd.DataFrame, ohlcv_test: pd.DataFrame, symbol: str):
    print(f"\n{DIVIDER}")
    print("MODULE 2 - Strategy Search (Beam & A*)")
    print(DIVIDER)

    bh_close = ohlcv_train["Close"].values[WARMUP_BARS:]
    bh_rets = np.diff(bh_close) / bh_close[:-1]
    bh_sharpe = float(np.sqrt(252) * bh_rets.mean() / bh_rets.std()) if bh_rets.std() > 0 else 0.0
    bh_total = float(bh_close[-1] / bh_close[0] - 1)

    print(f"  Data: {len(ohlcv_train)} train / {len(ohlcv_test)} test bars  ({symbol})")
    print(f"  Buy-and-Hold baseline (train): Sharpe={bh_sharpe:+.3f}, Return={bh_total:+.2%}")

    top_k = 5
    all_strategies = []
    for method in ("beam", "astar"):
        print(f"\n  {THIN_DIVIDER}")
        print(f"  {method.upper()} SEARCH - top {top_k}")
        print(f"  {THIN_DIVIDER}")
        strategies = search_top_strategies(ohlcv_train, top_k=top_k, method=method)
        all_strategies.extend(strategies)
        for i, s in enumerate(strategies, 1):
            p = s.params
            print(
                f"    {i}. Sharpe={s.sharpe:+.3f}  Return={s.total_return:+.2%}  "
                f"MaxDD={s.max_drawdown:+.2%}  WinRate={s.win_rate:.0%}  Trades={s.num_trades}"
            )

        best = strategies[0]
        rets, _ = backtest(ohlcv_test, best.params)
        oos_sharpe = sharpe_ratio(rets)
        oos_ret = float(np.prod(1 + rets) - 1) if len(rets) > 0 else 0.0
        print(f"    Out-of-sample best: Sharpe={oos_sharpe:+.3f}, Return={oos_ret:+.2%}")

    return all_strategies


# ── Module 3 ─────────────────────────────────────────────────────────────────

def run_module_3(ohlcv: pd.DataFrame, symbol: str):
    """Returns (SelectionResult, combined_candidate_list)."""
    print(f"\n{DIVIDER}")
    print("MODULE 3 - Evolution + Unified Strategy Selection")
    print(DIVIDER)

    ga_config = GAConfig(population_size=20, generations=20, seed=42)
    constraints = StrategyConstraints(
        min_sharpe=0.0,
        min_total_return=0.0,
        min_win_rate=0.50,
        max_drawdown_min=-0.30,
        min_trades=5,
    )
    preferences = SelectionPreferences()

    m2_list, ga_list, combined = gather_unified_candidate_pools(
        ohlcv, top_k=5, ga_config=ga_config
    )

    m2_best = max(s.sharpe for s in m2_list) if m2_list else 0.0
    ga_best = max(s.sharpe for s in ga_list) if ga_list else 0.0
    print(
        f"  Candidates: {len(m2_list)} from Module 2 (best Sharpe {m2_best:+.3f}), "
        f"{len(ga_list)} from GA (best Sharpe {ga_best:+.3f}), "
        f"{len(combined)} total"
    )

    result = finalize_unified_selection(
        m2_list, ga_list, combined,
        constraints=constraints,
        preferences=preferences,
    )

    print(f"\n  Selected strategy summary:")
    print(f"    {result.summary}")
    print(f"  Why: {result.reason}")
    if result.origin:
        print(f"  Winner source: {result.origin}")
    print(
        f"  Best Sharpe - Module 2: {result.m2_best_sharpe:+.3f}, "
        f"GA: {result.ga_best_sharpe:+.3f}"
    )

    return result, combined


# ── Module 4 ─────────────────────────────────────────────────────────────────

def run_module_4(symbol: str, candidate_strategies, m3_selected):
    print(f"\n{DIVIDER}")
    print("MODULE 4 - Sentiment Analysis (Alpha Vantage News)")
    print(DIVIDER)

    try:
        result = analyze_market_sentiment(
            tickers=symbol,
            candidate_strategies=candidate_strategies,
            m3_selected=m3_selected,
            news_limit=30,
            fit_classifier_from_feed=True,
        )
    except Exception as exc:
        print(f"  Sentiment fetch failed: {exc}")
        print("  (Module 4 requires ALPHA_VANTAGE_API_KEY in .env)")
        return None

    print(f"  Regime:       {result.regime.value}")
    print(f"  Confidence:   {result.confidence:.3f}")
    print(f"  Classifier:   {result.classification_method}")
    print(f"  Articles:     {len(result.articles)}")
    print()
    print("  Top headlines:")
    for i, headline in enumerate(result.top_headlines, 1):
        print(f"    {i}. {headline[:90]}{'...' if len(headline) > 90 else ''}")
    print()
    if result.recommended_strategy is not None:
        s = result.recommended_strategy
        print("  Recommended strategy (adjusted for regime):")
        print(
            f"    Sharpe={s.sharpe:+.3f}  Return={s.total_return:+.2%}  "
            f"MaxDD={s.max_drawdown:+.2%}  Trades={s.num_trades}"
        )
        print(f"    Params: {s.params}")
    else:
        print("  Recommended strategy: (none)")
    print(f"\n  Rationale: {result.recommendation_reason}")

    _generate_visual_report(result, symbol)
    return result


def _generate_visual_report(result, symbol: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ROOT / "data"
    png_path = out_dir / "pipeline_report.png"
    html_path = out_dir / "pipeline_report.html"
    subtitle = f"Full pipeline | {symbol}"

    fig = build_sentiment_demo_figure(result, subtitle=subtitle)
    try:
        save_figure_png(fig, png_path)
        write_html_report(
            result,
            png_path=png_path,
            html_path=html_path,
            subtitle=subtitle,
        )
        print(f"\n  Visual report saved:")
        print(f"    PNG:  {png_path.resolve()}")
        print(f"    HTML: {html_path.resolve()}")
    finally:
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(DIVIDER)
    print("  Intelligent Trading Agent - Full Pipeline")
    print(DIVIDER)

    symbol = _prompt_ticker()
    period = _prompt_period()

    ohlcv = _load_ohlcv(symbol, period)
    if len(ohlcv) <= WARMUP_BARS:
        print(f"  Only {len(ohlcv)} bars - need >{WARMUP_BARS}. Aborting.")
        return

    # M1: knowledge-base inference on latest bar
    run_module_1(ohlcv, symbol)

    # M2: strategy search (train/test split)
    split = len(ohlcv) // 2
    ohlcv_train = ohlcv.iloc[:split].copy()
    ohlcv_test = ohlcv.iloc[split:].copy()

    if len(ohlcv_train) > WARMUP_BARS and len(ohlcv_test) > WARMUP_BARS:
        run_module_2(ohlcv_train, ohlcv_test, symbol)
    else:
        print(f"\n  Skipping Module 2: not enough bars for train/test split.")

    # M3: evolution + unified selection (also returns the full candidate pool)
    m3_result, candidate_pool = run_module_3(ohlcv, symbol)
    m3_selected = m3_result.strategy if m3_result else None

    # M4: sentiment analysis using the M3 candidate pool
    run_module_4(symbol, candidate_pool, m3_selected)

    print(f"\n{DIVIDER}")
    print(f"  Pipeline complete for {symbol}.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
