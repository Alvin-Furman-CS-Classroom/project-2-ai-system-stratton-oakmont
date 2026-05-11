"""Fast unit tests for Module 3 GA helpers (no full evolution runs)."""

from __future__ import annotations

import numpy as np
import pytest
import src.module_3_evolution.evolution as evo

from src.module_3_evolution.evolution import (
    GAConfig,
    _build_next_generation,
    _clamp_params,
    _mutate,
    _random_params,
    _tournament_select,
    _uniform_crossover,
)
from src.shared import CandidateStrategy


@pytest.fixture
def tiny_ranges() -> dict[str, tuple[float, float]]:
    return {"a": (0.0, 10.0), "b": (-5.0, 5.0)}


def test_clamp_params_inside_range_unchanged(tiny_ranges):
    p = {"a": 3.0, "b": 0.0}
    assert _clamp_params(p, tiny_ranges) == p


def test_clamp_params_clips_below_and_above(tiny_ranges):
    out = _clamp_params({"a": -1.0, "b": 99.0}, tiny_ranges)
    assert out["a"] == 0.0
    assert out["b"] == 5.0


def test_random_params_within_bounds(tiny_ranges):
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = _random_params(rng, tiny_ranges)
        assert 0.0 <= p["a"] <= 10.0
        assert -5.0 <= p["b"] <= 5.0


def test_uniform_crossover_child_values_from_parents_only(tiny_ranges):
    rng = np.random.default_rng(1)
    pa = {"a": 1.0, "b": 2.0}
    pb = {"a": 9.0, "b": -4.0}
    for _ in range(30):
        child = _uniform_crossover(pa, pb, rng)
        assert child["a"] in (1.0, 9.0)
        assert child["b"] in (2.0, -4.0)


def test_mutate_output_within_ranges(tiny_ranges):
    rng = np.random.default_rng(2)
    start = {"a": 5.0, "b": 0.0}
    for _ in range(50):
        out = _mutate(start, tiny_ranges, mutation_rate=1.0, rng=rng)
        assert 0.0 <= out["a"] <= 10.0
        assert -5.0 <= out["b"] <= 5.0


def test_tournament_select_returns_highest_sharpe_in_sample():
    rng = np.random.default_rng(3)
    pop = [
        CandidateStrategy(params={}, sharpe=0.1),
        CandidateStrategy(params={}, sharpe=0.9),
        CandidateStrategy(params={}, sharpe=0.5),
    ]
    # With size 3, entire population competes → best is 0.9
    winner = _tournament_select(pop, tournament_size=3, rng=rng)
    assert winner.sharpe == 0.9


def test_tournament_select_invalid_size_falls_back_to_global_best():
    rng = np.random.default_rng(4)
    pop = [
        CandidateStrategy(params={}, sharpe=0.1),
        CandidateStrategy(params={}, sharpe=0.7),
    ]
    assert _tournament_select(pop, tournament_size=1, rng=rng).sharpe == 0.7
    assert _tournament_select(pop, tournament_size=10, rng=rng).sharpe == 0.7


def test_build_next_generation_population_size_and_elitism(tiny_ranges):
    """Elites are copies of top strategies' params; population fills to size."""
    rng = np.random.default_rng(5)
    config = GAConfig(
        population_size=8,
        generations=1,  # unused here
        elitism=2,
        tournament_size=2,
        crossover_rate=0.5,
        mutation_rate=0.3,
        seed=5,
        param_ranges=tiny_ranges,
    )
    evaluated = [
        CandidateStrategy(params={"a": 1.0, "b": 0.0}, sharpe=float(i))
        for i in range(6)
    ]
    next_gen = _build_next_generation(evaluated, config, rng, tiny_ranges)
    assert len(next_gen) == config.population_size
    # Top two by sharpe are sharpe 5 and 4 → params a=1,b=0 for all in this construction
    # Actually sharpe is float(i) for i in 0..5, so best is i=5 then i=4
    best_params = {"a": 1.0, "b": 0.0}
    assert next_gen[0] == best_params or next_gen[1] == best_params


def test_build_next_generation_caps_elitism_to_population_size(tiny_ranges):
    """If elitism > population_size, generation size still matches population_size."""
    rng = np.random.default_rng(6)
    config = GAConfig(
        population_size=2,
        generations=1,
        elitism=3,
        tournament_size=2,
        crossover_rate=0.5,
        mutation_rate=0.3,
        seed=6,
        param_ranges=tiny_ranges,
    )
    evaluated = [
        CandidateStrategy(params={"a": float(i), "b": 0.0}, sharpe=float(i))
        for i in range(4)
    ]
    next_gen = _build_next_generation(evaluated, config, rng, tiny_ranges)
    assert len(next_gen) == config.population_size


def test_build_next_generation_injects_configured_immigrants(tiny_ranges, monkeypatch):
    rng = np.random.default_rng(7)
    config = GAConfig(
        population_size=10,
        generations=1,
        elitism=2,
        tournament_size=2,
        crossover_rate=0.0,
        mutation_rate=0.0,
        immigrant_fraction=0.3,
        seed=7,
        param_ranges=tiny_ranges,
    )
    evaluated = [
        CandidateStrategy(params={"a": 1.0, "b": 0.0}, sharpe=float(i))
        for i in range(6)
    ]

    calls = {"n": 0}

    def fake_random_params(_rng, _ranges):
        calls["n"] += 1
        return {"a": 0.0, "b": -5.0}

    monkeypatch.setattr(evo, "_random_params", fake_random_params)

    next_gen = _build_next_generation(evaluated, config, rng, tiny_ranges)

    expected_immigrants = 3  # round(0.3 * 10)
    assert calls["n"] == expected_immigrants
    assert next_gen.count({"a": 0.0, "b": -5.0}) == expected_immigrants
    assert len(next_gen) == config.population_size


def test_ga_config_default_param_ranges_is_copy():
    c1 = GAConfig()
    c2 = GAConfig()
    assert c1.param_ranges is not c2.param_ranges
