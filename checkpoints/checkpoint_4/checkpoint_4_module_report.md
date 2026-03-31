# Checkpoint 4 — Module Rubric Report

**Prepared per:** [`checkpoint_preparation.md`](../../checkpoint_preparation.md)  
**Cross-reference:** [AI System Module Rubric](https://csc-343.path.app/projects/project-2-ai-system/ai-system.rubric.md) (Parts 1–2)  
**Module:** 4 — Market Sentiment Classifier (supervised learning + regime + strategy recommendation)  
**Evidence:** `Project 2 Proposal.md` (Module 4 row); `src/module_4_sentiment/`; `unit_tests/module_4_sentiment/`; `integration_tests/module_4/`; demo `python -m src.module_4_sentiment.demo`

---

## Summary

Module 4 **implements** the proposed pipeline: **Alpha Vantage** news ingestion (`NEWS_SENTIMENT`), **feature extraction** from articles, **multinomial logistic regression** with a **heuristic fallback** when training data are sparse, **regime** output (Bullish / Bearish / Neutral) with confidence, and **regime-aware selection** of a `CandidateStrategy` from the Module 3 pool. **Inputs and outputs** are explicit (`SentimentAnalysisResult`, typed articles, env-based API key). The pipeline also supports **graceful fetch-error fallback** and optional **Module 3 context passthrough** fields. **Tests** (unit + mocked integration) and **demos** (console, PNG/HTML reports) demonstrate behavior without requiring live API calls in CI. **Documentation** lives in module docstrings and this checkpoint narrative; **integration readiness** for Module 5 is clear: downstream receives regime, confidence, recommended strategy, and rationale.

---

## Findings & Scores

Per preparation guide, each area is scored **0–4** (4 = exceeds expectations).

### 1. Specification Clarity — **Score: 4**

- Proposal row for Module 4 defines supervised learning, Alpha Vantage inputs, and outputs (regime, confidence, recommended strategy).  
- Implementation matches: logistic regression over article features, API client, strategy mapping rules in `strategy_recommendation.py`, and orchestration in `pipeline.py`.  
- **Official rubric:** strong **Topic Engagement** and **I/O Clarity** for sentiment/regime.

### 2. Inputs / Outputs — **Score: 4**

- **Inputs:** `tickers` (and optional API params), `Sequence[CandidateStrategy]`, optional `m3_selected`, optional `m3_context`, optional `SentimentRegimeClassifier`; environment `ALPHA_VANTAGE_API_KEY` or explicit `api_key`. Optional reliability controls include `fallback_on_fetch_error` and `fit_classifier_from_feed`.  
- **Outputs:** `SentimentAnalysisResult` with `regime`, `confidence`, `classification_method`, `top_headlines`, `articles`, `recommended_strategy`, `recommendation_reason`, plus optional passthrough fields `m3_origin`, `m3_reason`, `m3_summary`, and `fallback_note`.  
- **Next module feed:** Module 5 can consume regime, confidence, chosen strategy metrics, and volatility/capital from shared types (Module 5 stub documents intended RL inputs in the proposal).

### 3. Dependencies — **Score: 4**

- Depends on **Module 3** handoff type `CandidateStrategy` (and optional M3-selected strategy for neutral mapping).  
- External: **Alpha Vantage** HTTP API, **scikit-learn** (`LogisticRegression` + pipeline), **matplotlib** (demo visualization only).  
- **python-dotenv** optional for `.env` loading. Boundaries are clean: no circular imports.

### 4. Test Coverage — **Score: 4**

- **Unit:** API client (mocked HTTP), features/classifier, strategy recommendation, pipeline (`unit_tests/module_4_sentiment/`).  
- **Integration:** `integration_tests/module_4/test_m1_m4_integration.py` exercises **news → classify → recommend** with mocked fetch (no quota burn).  
- Edge cases: missing API key (with dotenv stub), rate-limit / error payloads, sklearn 1.8+ API (no deprecated `multi_class`).

### 5. Documentation — **Score: 4**

- Public symbols exported from `src/module_4_sentiment/__init__.py`; key functions have docstrings (`fetch_news_sentiment`, `analyze_market_sentiment`, classifier, demo entry).  
- API key variable is documented in module/client docstrings; demo CLI documents `--offline`, `--open`, `--no-viz`.

### 6. Integration Readiness — **Score: 4**

- **Upstream:** Accepts Module 3 **candidate pool** and optional selected strategy.  
- **Downstream:** Delivers a **single recommended** `CandidateStrategy` plus narrative strings suitable for position sizing and reporting.  
- **Demos** run from repo root; visual report (`data/m4_demo_report.png`, `.html`) supports checkpoint evidence and presentation.

---

## Score Table (Preparation Guide Criteria)

| Criterion               | Score |
|-------------------------|-------|
| Specification Clarity   | 4     |
| Inputs / Outputs        | 4     |
| Dependencies            | 4     |
| Test Coverage           | 4     |
| Documentation           | 4     |
| Integration Readiness   | 4     |

**Average:** **4.0 / 4.0**

---

## Mapping to AI System Rubric (Self-Assessment)

- **1.1 Functionality:** API client + regime classifier + strategy pick + demo/visuals. **8 / 8**
- **1.2 Code Elegance:** See `checkpoint_4_elegance_report.md`. **7 / 7**
- **1.3 Documentation:** Docstrings + proposal alignment + demo help. **4 / 4**
- **1.4 I/O Clarity:** Typed results and documented env vars. **3 / 3**
- **1.5 Topic Engagement:** Supervised LR + API features + explainable regime/strategy mapping. **5 / 5**
- **2.1 Test Coverage:** Unit + integration with mocks. **6 / 6**
- **2.2 Test Quality:** Assertions on behavior and error paths. **5 / 5**
- **2.3 Test Organization:** Mirrors `src/` layout. **4 / 4**
- **3.x GitHub:** Team maintains commits/PRs per course policy.

---

## Module explanation (presentation checklist)

### Input (what the module accepts)

- **News query:** e.g. tickers `"SPY"` and optional `limit` for `NEWS_SENTIMENT`.  
- **Strategy pool:** list of `CandidateStrategy` from Module 3 (params + backtest metrics).  
- **Optional:** Module 3’s unified pick for neutral-regime default, and optional Module 3 context (`origin`, `reason`, `summary`) for traceability.  
- **Credentials:** `ALPHA_VANTAGE_API_KEY` (or pass-through `api_key`).

### Output (what it produces)

- **`SentimentAnalysisResult`:** regime enum, confidence, method (`logistic_regression` vs `heuristic`), headlines, full article tuples, **one** recommended `CandidateStrategy`, a short **recommendation_reason** string, optional Module 3 passthrough fields (`m3_origin`, `m3_reason`, `m3_summary`), and optional `fallback_note` when news fetch fallback is used.

### AI concepts

- **Supervised learning:** multinomial logistic regression on per-article features, labels derived from API sentiment labels; **probability averaging** across articles for regime.  
- **Heuristic fallback** when too few labeled rows or fit fails: mean sentiment score thresholds.  
- **Regime-conditioned policy** maps Bullish/Bearish/Neutral to different selection rules over the same `CandidateStrategy` pool.

---

## Participation reminder

The course **participation requirement** is a **gate**: all team members must show substantive contribution in commit history. This report does not assess Git history—instructors will.

---

## Action items before submission

- [x] Run `python -m pytest unit_tests/module_4_sentiment integration_tests/module_4 -q` and keep logs for evidence.  
- [ ] Run `python -m src.module_4_sentiment.demo --offline` (or live with key) and archive **PNG/HTML** from `data/` for the checkpoint.  
- [ ] **Module 5:** wire `SentimentAnalysisResult` into the RL position-sizing API when implemented.

---

*Self-review for Checkpoint 4 (Module 4); instructor assessment is authoritative.*
