"""Module 4 demo: Alpha Vantage news → regime (LR or heuristic) → strategy pick.

**Online** (default): calls the real NEWS_SENTIMENT API (needs ``ALPHA_VANTAGE_API_KEY``
in the environment or a ``.env`` file at the repo root).

**Offline** (``--offline``): uses a canned article list so you can run without network
or API quota (good for quick checks and classrooms).

Run from the repository root::

    python -m src.module_4_sentiment.demo
    python -m src.module_4_sentiment.demo --tickers IBM --limit 20
    python -m src.module_4_sentiment.demo --offline

Visual report (PNG + HTML under ``data/`` by default), optional chart window::

    python -m src.module_4_sentiment.demo --offline --show
    python -m src.module_4_sentiment.demo --offline --open
    python -m src.module_4_sentiment.demo --no-viz
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import webbrowser

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.module_4_sentiment.alpha_vantage_client import NewsArticle
from src.module_4_sentiment.demo_visualization import (
    build_sentiment_demo_figure,
    save_figure_png,
    write_html_report,
)
from src.module_4_sentiment.pipeline import (
    SentimentAnalysisResult,
    analyze_market_sentiment,
    select_top_headlines_for_regime,
)
from src.module_4_sentiment.regime_classifier import SentimentRegimeClassifier
from src.module_4_sentiment.strategy_recommendation import recommend_strategy_for_regime
from src.shared.types import CandidateStrategy


def _example_candidate_pool() -> tuple[list[CandidateStrategy], CandidateStrategy]:
    """Small synthetic pool mimicking Module 3 handoff (params + backtest metrics)."""
    pool = [
        CandidateStrategy(
            params={"rsi_oversold": 28.0, "rsi_overbought": 72.0},
            sharpe=0.85,
            total_return=0.12,
            max_drawdown=-0.18,
            win_rate=0.55,
            num_trades=42,
            explanation="Balanced RSI bands",
        ),
        CandidateStrategy(
            params={"rsi_oversold": 32.0, "rsi_overbought": 68.0},
            sharpe=1.15,
            total_return=0.18,
            max_drawdown=-0.32,
            win_rate=0.48,
            num_trades=60,
            explanation="Higher turnover / higher Sharpe",
        ),
        CandidateStrategy(
            params={"rsi_oversold": 25.0, "rsi_overbought": 75.0},
            sharpe=0.45,
            total_return=0.06,
            max_drawdown=-0.09,
            win_rate=0.62,
            num_trades=28,
            explanation="Defensive / tight drawdown",
        ),
    ]
    m3_pick = pool[0]
    return pool, m3_pick


def _offline_articles() -> list[NewsArticle]:
    """Enough labeled articles to train the classifier without HTTP."""
    rows: list[NewsArticle] = []
    for i in range(10):
        rows.append(
            NewsArticle(
                title=f"Bearish headline sample {i}",
                url="https://example.com/b",
                time_published="20240115T100000",
                summary="Markets weigh downside risks." * 2,
                overall_sentiment_score=-0.25 + 0.01 * i,
                overall_sentiment_label="Bearish",
                raw={},
            )
        )
    for i in range(10):
        rows.append(
            NewsArticle(
                title=f"Neutral headline sample {i}",
                url="https://example.com/n",
                time_published="20240115T110000",
                summary="Mixed signals from analysts." * 2,
                overall_sentiment_score=0.02 * i,
                overall_sentiment_label="Neutral",
                raw={},
            )
        )
    for i in range(10):
        rows.append(
            NewsArticle(
                title=f"Bullish headline sample {i}",
                url="https://example.com/u",
                time_published="20240115T120000",
                summary="Risk-on tone as equities advance." * 2,
                overall_sentiment_score=0.35 + 0.01 * i,
                overall_sentiment_label="Bullish",
                raw={},
            )
        )
    return rows


def _run_offline(pool: list[CandidateStrategy], m3_selected: CandidateStrategy) -> SentimentAnalysisResult:
    articles = _offline_articles()
    clf = SentimentRegimeClassifier()
    clf.fit_from_articles(articles)
    regime, confidence, method = clf.predict_regime(articles)
    strat, reason = recommend_strategy_for_regime(regime, pool, m3_selected=m3_selected)
    top_headlines = select_top_headlines_for_regime(articles, regime, limit=5)
    return SentimentAnalysisResult(
        regime=regime,
        confidence=confidence,
        classification_method=method,
        top_headlines=top_headlines,
        articles=tuple(articles),
        recommended_strategy=strat,
        recommendation_reason=reason,
    )


def _print_result(result: SentimentAnalysisResult, *, offline: bool) -> None:
    mode = "OFFLINE (mock articles)" if offline else "LIVE (Alpha Vantage)"
    print()
    print("=" * 64)
    print(f"Module 4 - Market sentiment -> regime -> strategy  [{mode}]")
    print("=" * 64)
    print(f"  Regime:              {result.regime.value}")
    print(f"  Confidence:        {result.confidence:.3f}")
    print(f"  Classifier:        {result.classification_method}")
    print(f"  Articles in feed:  {len(result.articles)}")
    print()
    print("  Top headlines:")
    for i, t in enumerate(result.top_headlines, 1):
        print(f"    {i}. {t[:90]}{'...' if len(t) > 90 else ''}")
    print()
    if result.recommended_strategy is not None:
        s = result.recommended_strategy
        print("  Recommended strategy (from pool):")
        print(f"    Sharpe={s.sharpe:+.3f}  Return={s.total_return:+.2%}  "
              f"MaxDD={s.max_drawdown:+.2%}  Trades={s.num_trades}")
        print(f"    Params: {s.params}")
    else:
        print("  Recommended strategy: (none)")
    print()
    print("  Rationale:")
    print(f"    {result.recommendation_reason}")
    print("=" * 64)
    print()


def _run_visual_report(
    result: SentimentAnalysisResult,
    *,
    subtitle: str,
    out_dir: pathlib.Path,
    show_figure: bool,
    open_browser: bool,
) -> None:
    import matplotlib.pyplot as plt

    png_path = out_dir / "m4_demo_report.png"
    html_path = out_dir / "m4_demo_report.html"
    fig = build_sentiment_demo_figure(result, subtitle=subtitle)
    try:
        save_figure_png(fig, png_path)
        write_html_report(result, png_path=png_path, html_path=html_path, subtitle=subtitle)
        print("Wrote visual report:")
        print(f"  PNG:  {png_path.resolve()}")
        print(f"  HTML: {html_path.resolve()}")
        if show_figure:
            plt.show()
        if open_browser:
            webbrowser.open(html_path.as_uri())
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Module 4 sentiment classifier demo.")
    parser.add_argument(
        "--tickers",
        default="SPY",
        help="Comma-separated tickers for NEWS_SENTIMENT (online only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max articles to request from Alpha Vantage (online only).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not call the API; use built-in sample articles.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip PNG/HTML chart output (console only).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Where to write m4_demo_report.png and .html (default: <repo>/data).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib figure window after saving.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the HTML report in your default browser.",
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir if args.output_dir is not None else _ROOT / "data"
    pool, m3_pick = _example_candidate_pool()

    if args.offline:
        result = _run_offline(pool, m3_pick)
        _print_result(result, offline=True)
        if not args.no_viz:
            sub = "Offline demo (mock articles)"
            _run_visual_report(
                result,
                subtitle=sub,
                out_dir=out_dir,
                show_figure=args.show,
                open_browser=args.open_browser,
            )
        return

    result = analyze_market_sentiment(
        tickers=args.tickers,
        candidate_strategies=pool,
        m3_selected=m3_pick,
        news_limit=args.limit,
        fit_classifier_from_feed=True,
    )
    _print_result(result, offline=False)
    if not args.no_viz:
        sub = f"Live Alpha Vantage | tickers={args.tickers!r} | limit={args.limit}"
        _run_visual_report(
            result,
            subtitle=sub,
            out_dir=out_dir,
            show_figure=args.show,
            open_browser=args.open_browser,
        )


if __name__ == "__main__":
    main()
