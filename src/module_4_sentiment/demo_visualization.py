"""Charts and HTML report for the Module 4 demo (matplotlib + optional browser view)."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np

from .alpha_vantage_client import article_headline
from .pipeline import SentimentAnalysisResult
from .regime_classifier import MarketRegime

_COLOR = {
    MarketRegime.BEARISH: "#c0392b",
    MarketRegime.NEUTRAL: "#7f8c8d",
    MarketRegime.BULLISH: "#27ae60",
}


def _single_line_chart_headline(title: str, *, max_chars: int = 54) -> str:
    """One line per bar so y-axis ticks do not overlap (full titles stay in HTML/console)."""
    t = " ".join((title or "").split())
    if not t:
        return "(Untitled)"
    if len(t) <= max_chars:
        return t
    cut = max_chars - 1
    while cut > 0 and t[cut - 1].isspace():
        cut -= 1
    return t[:cut].rstrip() + "…"


def _html_chart_section(img_b64: str) -> str:
    if not img_b64:
        return "<p><em>Chart image not available.</em></p>"
    return (
        "<h2>Charts</h2>"
        f'<p><img src="data:image/png;base64,{img_b64}" alt="Module 4 charts"/></p>'
    )


def build_sentiment_demo_figure(result: SentimentAnalysisResult, *, subtitle: str = "") -> "object":
    """Create a matplotlib Figure. Caller must close figure after save/show."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    scores = [
        float(a.overall_sentiment_score)
        for a in result.articles
        if a.overall_sentiment_score is not None
    ]

    max_bars = 15
    barh_rows: tuple[list[float], list[str], int] | None = None
    if scores:
        rows = [
            (float(a.overall_sentiment_score), article_headline(a))
            for a in result.articles
            if a.overall_sentiment_score is not None
        ]
        rows.sort(key=lambda t: abs(t[0]), reverse=True)
        rows = rows[:max_bars]
        rows.sort(key=lambda t: t[0])
        plot_scores = [s for s, _ in rows]
        y_labs = [_single_line_chart_headline(h) for _, h in rows]
        total_scored = sum(1 for a in result.articles if a.overall_sentiment_score is not None)
        barh_rows = (plot_scores, y_labs, total_scored)

    n_bars = len(barh_rows[0]) if barh_rows else 0
    if n_bars:
        mid_ratio = 1.05 + n_bars * 0.095
        fig_h = min(17.0, 5.2 + n_bars * 0.36)
    else:
        mid_ratio = 1.4
        fig_h = 9.0

    fig = plt.figure(figsize=(13, fig_h))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, mid_ratio, 0.7],
                  hspace=0.40, wspace=0.35, left=0.30, right=0.96, top=0.90, bottom=0.06)

    fig.suptitle("Module 4 - Sentiment & regime", fontsize=14, fontweight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9, color="dimgray")

    # --- Regime bars (left top) ---
    ax_reg = fig.add_subplot(gs[0, 0])
    regimes = [MarketRegime.BEARISH, MarketRegime.NEUTRAL, MarketRegime.BULLISH]
    if result.regime_scores:
        heights = [float(result.regime_scores.get(r, 0.0)) for r in regimes]
    else:
        heights = [result.confidence if result.regime is r else 0.08 for r in regimes]
    colors = [_COLOR[r] for r in regimes]
    bars = ax_reg.bar(
        [r.value for r in regimes],
        heights,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    ax_reg.set_ylim(0, 1.05)
    ax_reg.set_ylabel("Weight / confidence")
    ax_reg.set_title("Detected regime (bar height = model emphasis)")
    for bar, r in zip(bars, regimes):
        value = float(bar.get_height())
        ax_reg.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if r is result.regime else "normal",
        )
        if r is result.regime:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.35)

    # --- Mean score + spread (right top) ---
    ax_stat = fig.add_subplot(gs[0, 1])
    if scores:
        arr = np.array(scores)
        mean_s = float(np.mean(arr))
        ax_stat.axvspan(-1, 0, alpha=0.12, color="red")
        ax_stat.axvspan(0, 1, alpha=0.12, color="green")
        ax_stat.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_stat.axvline(mean_s, color="#2980b9", linewidth=2.5, label=f"Mean = {mean_s:+.3f}")
        ax_stat.hist(arr, bins=min(20, max(8, len(arr) // 3)), color="#34495e", alpha=0.85, edgecolor="white")
        ax_stat.set_xlim(-1.05, 1.05)
        ax_stat.legend(loc="upper right", fontsize=8)
    else:
        ax_stat.text(0.5, 0.5, "No per-article scores", ha="center", va="center")
    ax_stat.set_title("Distribution of article sentiment scores")
    ax_stat.set_xlabel("Sentiment score (API)")

    # --- Per-article scores (middle, full width) ---
    ax_art = fig.add_subplot(gs[1, :])
    if barh_rows:
        plot_scores, y_labels, total = barh_rows
        y_pos = np.arange(len(plot_scores))
        bar_colors = [
            _COLOR[MarketRegime.BEARISH] if s < -0.05
            else _COLOR[MarketRegime.BULLISH] if s > 0.05
            else _COLOR[MarketRegime.NEUTRAL]
            for s in plot_scores
        ]
        ax_art.barh(y_pos, plot_scores, color=bar_colors, alpha=0.85, height=0.72, edgecolor="white")
        ax_art.set_yticks(y_pos)
        ax_art.set_yticklabels(y_labels, fontsize=8)
        ax_art.tick_params(axis="y", pad=6)
        ax_art.axvline(0, color="black", linewidth=0.6)
        ax_art.set_xlabel("Sentiment score")
        ax_art.set_title(f"Top {len(plot_scores)} articles by magnitude (of {total} total)")
        ax_art.set_ylim(-0.6, len(plot_scores) - 0.4)
    else:
        ax_art.text(0.5, 0.5, "No articles to plot", ha="center", va="center")

    # --- Summary text (bottom) ---
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    method = result.classification_method.replace("_", " ")
    lines = [
        f"Classifier: {method}",
        f"Recommended strategy - Sharpe {result.recommended_strategy.sharpe:+.3f}, "
        f"Return {result.recommended_strategy.total_return:+.1%}, "
        f"MaxDD {result.recommended_strategy.max_drawdown:+.1%}"
        if result.recommended_strategy
        else "Recommended strategy: (none)",
    ]
    if result.recommended_strategy:
        lines.append(f"Params: {result.recommended_strategy.params}")
    lines.append("")
    lines.append("Rationale:")
    lines.append(result.recommendation_reason)
    body = "\n".join(lines)
    ax_txt.text(
        0.02,
        0.98,
        body,
        transform=ax_txt.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"),
    )

    return fig


def save_figure_png(fig: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")


def write_html_report(
    result: SentimentAnalysisResult,
    *,
    png_path: Path | None,
    html_path: Path,
    subtitle: str,
) -> None:
    """Write a simple HTML page with embedded chart image if PNG exists."""
    img_b64 = ""
    if png_path is not None and png_path.is_file():
        raw = png_path.read_bytes()
        img_b64 = base64.standard_b64encode(raw).decode("ascii")

    regime_color = _COLOR.get(result.regime, "#333")
    headlines = "".join(
        f"<li>{html.escape(t)}</li>" for t in result.top_headlines[:8]
    )
    strat_block = ""
    if result.recommended_strategy:
        s = result.recommended_strategy
        strat_block = (
            f"<p><strong>Sharpe</strong> {s.sharpe:+.3f} &nbsp; "
            f"<strong>Return</strong> {s.total_return:+.1%} &nbsp; "
            f"<strong>MaxDD</strong> {s.max_drawdown:+.1%}</p>"
            f"<pre>{html.escape(str(s.params))}</pre>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Module 4 sentiment report</title>
  <style>
    body {{ font-family: Segoe UI, system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; }}
    h1 {{ font-size: 1.35rem; }}
    .badge {{
      display: inline-block; padding: 6px 14px; border-radius: 8px; color: #fff;
      background: {regime_color}; font-weight: 600;
    }}
    .meta {{ color: #555; font-size: 0.9rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
    ul {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Module 4 - Market sentiment</h1>
  <p class="meta">{html.escape(subtitle)}</p>
  <p>Regime: <span class="badge">{html.escape(result.regime.value)}</span>
     &nbsp; Confidence: <strong>{result.confidence:.3f}</strong>
     &nbsp; ({html.escape(result.classification_method)})</p>
  {_html_chart_section(img_b64)}
  <h2>Top headlines</h2>
  <ul>{headlines}</ul>
  <h2>Recommended strategy</h2>
  {strat_block}
  <h2>Rationale</h2>
  <p>{html.escape(result.recommendation_reason)}</p>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc, encoding="utf-8")
