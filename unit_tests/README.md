# Unit tests

Layout mirrors `src/`:

| Path | Focus |
|------|--------|
| `module_1_knowledge_base/` | KB / rules |
| `module_2_strategy_search/` | Search & evaluation |
| `module_3_evolution/` | GA (`test_evolution.py` = integration-style; `test_evolution_helpers.py` = fast pure helpers) |
| `module_3_strategy_selection/` | Constraints, scoring, reporting, unified M2+GA selection |
| `module_4_sentiment/`, `module_5_position_sizing/` | Later modules |

## Commands

From the **repository root**:

```bash
pytest unit_tests -q
pytest unit_tests/module_3_evolution unit_tests/module_3_strategy_selection -q
pytest unit_tests/module_3_strategy_selection/test_unified.py -v
```

Integration tests (end-to-end across modules) live in `integration_tests/` and run with the same `pytest` invocation (see `pytest.ini`).
