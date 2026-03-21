"""Module 3: Strategy Evolution Engine (Genetic Algorithms).

- ``evolve_from_seeds``: refines Module 2 ``CandidateStrategy`` seeds.
- ``evolve_randomly``: evolves from random parameters (from-scratch in param space).

The package demo (``python -m src.module_3_evolution.demo``) labels the final winner
as MODULE 2 (params match a seed) vs GA-GENERATED (evolved genome). For comparing
beam search vs GA-from-scratch in one selection step, use
``python -m src.module_3_strategy_selection.demo``.
"""

from __future__ import annotations

from .evolution import GAConfig, evolve_from_seeds, evolve_randomly

__all__ = ["GAConfig", "evolve_from_seeds", "evolve_randomly"]
