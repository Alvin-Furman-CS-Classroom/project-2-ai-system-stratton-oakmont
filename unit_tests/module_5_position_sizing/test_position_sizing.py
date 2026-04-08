"""Unit tests for Module 5: Adaptive Position Sizing Agent."""

import pytest
import numpy as np

from src.module_5_position_sizing.mdp import (
    NUM_STATES,
    NUM_ACTIONS,
    PositionAction,
    POSITION_PCTS,
    RegimeBucket,
    ConfidenceBucket,
    SharpeBucket,
    VolatilityBucket,
    State,
    build_state,
    state_index,
    discretize_regime,
    discretize_confidence,
    discretize_sharpe,
    discretize_volatility,
    compute_reward,
)
from src.module_5_position_sizing.agent import PositionSizingAgent, QAgentConfig
from src.module_5_position_sizing.sizing import (
    PositionSizingResult,
    recommend_position_size,
)
from src.module_4_sentiment.pipeline import SentimentAnalysisResult
from src.module_4_sentiment.regime_classifier import MarketRegime
from src.shared.types import CandidateStrategy


# ── MDP discretization ──────────────────────────────────────────────────

class TestDiscretization:
    def test_regime_bearish(self):
        assert discretize_regime("BEARISH") == RegimeBucket.BEARISH

    def test_regime_bullish(self):
        assert discretize_regime("BULLISH") == RegimeBucket.BULLISH

    def test_regime_neutral_default(self):
        assert discretize_regime("NEUTRAL") == RegimeBucket.NEUTRAL
        assert discretize_regime("something_else") == RegimeBucket.NEUTRAL

    def test_confidence_buckets(self):
        assert discretize_confidence(0.0) == ConfidenceBucket.LOW
        assert discretize_confidence(0.39) == ConfidenceBucket.LOW
        assert discretize_confidence(0.4) == ConfidenceBucket.MEDIUM
        assert discretize_confidence(0.69) == ConfidenceBucket.MEDIUM
        assert discretize_confidence(0.7) == ConfidenceBucket.HIGH
        assert discretize_confidence(1.0) == ConfidenceBucket.HIGH

    def test_sharpe_buckets(self):
        assert discretize_sharpe(-0.5) == SharpeBucket.NEGATIVE
        assert discretize_sharpe(0.0) == SharpeBucket.LOW
        assert discretize_sharpe(0.49) == SharpeBucket.LOW
        assert discretize_sharpe(0.5) == SharpeBucket.MEDIUM
        assert discretize_sharpe(1.0) == SharpeBucket.HIGH

    def test_volatility_buckets(self):
        assert discretize_volatility(0.05) == VolatilityBucket.LOW
        assert discretize_volatility(0.15) == VolatilityBucket.MEDIUM
        assert discretize_volatility(0.30) == VolatilityBucket.HIGH


class TestState:
    def test_build_state_returns_named_tuple(self):
        s = build_state("BULLISH", 0.8, 1.2, 0.10)
        assert isinstance(s, State)
        assert s.regime == RegimeBucket.BULLISH
        assert s.confidence == ConfidenceBucket.HIGH
        assert s.sharpe == SharpeBucket.HIGH
        assert s.volatility == VolatilityBucket.LOW

    def test_state_index_range(self):
        for r in RegimeBucket:
            for c in ConfidenceBucket:
                for sh in SharpeBucket:
                    for v in VolatilityBucket:
                        idx = state_index(State(r, c, sh, v))
                        assert 0 <= idx < NUM_STATES

    def test_state_indices_unique(self):
        indices = set()
        for r in RegimeBucket:
            for c in ConfidenceBucket:
                for sh in SharpeBucket:
                    for v in VolatilityBucket:
                        indices.add(state_index(State(r, c, sh, v)))
        assert len(indices) == NUM_STATES


class TestReward:
    def test_positive_return_positive_reward(self):
        r = compute_reward(0.10, 0.20, -0.05, 0.10)
        assert r > 0

    def test_large_drawdown_lowers_reward(self):
        r_small_dd = compute_reward(0.10, 0.10, -0.05, 0.10)
        r_large_dd = compute_reward(0.10, 0.10, -0.40, 0.10)
        assert r_small_dd > r_large_dd

    def test_zero_position_zero_reward(self):
        assert compute_reward(0.0, 0.20, -0.30, 0.50) == 0.0


# ── Q-learning agent ────────────────────────────────────────────────────

class TestAgent:
    def _state(self):
        return build_state("NEUTRAL", 0.5, 0.6, 0.10)

    def test_initial_q_table_zeros(self):
        agent = PositionSizingAgent()
        assert agent.q_table.shape == (NUM_STATES, NUM_ACTIONS)
        assert np.all(agent.q_table == 0.0)

    def test_select_action_returns_valid(self):
        agent = PositionSizingAgent()
        action = agent.select_action(self._state())
        assert action in PositionAction

    def test_greedy_on_zeros_returns_first_action(self):
        agent = PositionSizingAgent()
        action = agent.select_action(self._state(), greedy=True)
        assert action == PositionAction.PCT_1

    def test_update_changes_q(self):
        agent = PositionSizingAgent()
        s = self._state()
        q_before = agent.q_table[state_index(s)].copy()
        agent.update(s, PositionAction.PCT_5, 1.0, s)
        q_after = agent.q_table[state_index(s)]
        assert not np.array_equal(q_before, q_after)

    def test_epsilon_decays(self):
        agent = PositionSizingAgent(QAgentConfig(epsilon=1.0, epsilon_decay=0.5))
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.5)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.25)

    def test_epsilon_respects_minimum(self):
        agent = PositionSizingAgent(QAgentConfig(epsilon=0.06, epsilon_min=0.05, epsilon_decay=0.5))
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.05)

    def test_train_episode(self):
        agent = PositionSizingAgent(QAgentConfig(epsilon=0.0))
        states = [
            build_state("BULLISH", 0.8, 1.0, 0.10),
            build_state("BULLISH", 0.75, 0.9, 0.12),
            build_state("NEUTRAL", 0.5, 0.6, 0.20),
        ]
        total_r = agent.train_episode(states, strategy_return=0.15,
                                      max_drawdown=-0.10, volatility=0.12)
        assert isinstance(total_r, float)
        assert not np.all(agent.q_table == 0.0)

    def test_q_values_for(self):
        agent = PositionSizingAgent()
        qv = agent.q_values_for(self._state())
        assert len(qv) == NUM_ACTIONS
        assert all(isinstance(k, PositionAction) for k in qv)


# ── Public API ───────────────────────────────────────────────────────────

def _dummy_sentiment(regime: MarketRegime = MarketRegime.BULLISH,
                     confidence: float = 0.8) -> SentimentAnalysisResult:
    strat = CandidateStrategy(params={"rsi_oversold": 30.0}, sharpe=1.0,
                              total_return=0.15, max_drawdown=-0.10)
    return SentimentAnalysisResult(
        regime=regime,
        confidence=confidence,
        classification_method="logistic_regression",
        top_headlines=["headline"],
        articles=(),
        recommended_strategy=strat,
        recommendation_reason="test",
    )


class TestRecommendPositionSize:
    def test_returns_result(self):
        sentiment = _dummy_sentiment()
        strat = sentiment.recommended_strategy
        result = recommend_position_size(sentiment, strat, volatility=0.12)
        assert isinstance(result, PositionSizingResult)
        assert result.position_pct in POSITION_PCTS.values()
        assert result.capital == 10_000.0
        assert result.dollar_amount == pytest.approx(result.capital * result.position_pct)

    def test_custom_capital(self):
        sentiment = _dummy_sentiment()
        strat = sentiment.recommended_strategy
        result = recommend_position_size(sentiment, strat, volatility=0.10, capital=50_000)
        assert result.capital == 50_000.0

    def test_reasoning_and_risk(self):
        sentiment = _dummy_sentiment()
        strat = sentiment.recommended_strategy
        result = recommend_position_size(sentiment, strat, volatility=0.10)
        assert isinstance(result.reasoning, str) and len(result.reasoning) > 0
        assert isinstance(result.risk_assessment, str) and len(result.risk_assessment) > 0

    def test_trained_agent_prefers_larger_in_bullish(self):
        agent = PositionSizingAgent(QAgentConfig(epsilon=0.0))
        bullish_states = [build_state("BULLISH", 0.85, 1.2, 0.08) for _ in range(5)]
        for _ in range(200):
            agent.train_episode(bullish_states, strategy_return=0.25,
                                max_drawdown=-0.05, volatility=0.08)

        sentiment = _dummy_sentiment(MarketRegime.BULLISH, 0.85)
        strat = CandidateStrategy(params={}, sharpe=1.2, total_return=0.25,
                                  max_drawdown=-0.05)
        result = recommend_position_size(sentiment, strat, volatility=0.08,
                                         agent=agent)
        assert result.position_pct >= 0.10
