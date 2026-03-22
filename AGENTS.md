## Project Context

- System: Intelligent Trading Agent: Strategy Discovery Through Search and Adaptive Risk Management
- Team: Casen Shoemake, Kyler Bailey, Collin Riddle
- Pipeline: Module 1 -> Module 2 -> Module 3 -> Module 4 -> Module 5

## Current Module Definitions

- Module 1: Rule-based trading inference (BUY/SELL/HOLD + inference trace)
- Module 2: Strategy parameter search (Beam/A*) ranked by Sharpe
- Module 3: Evolution and unified selection
	- Uses Module 2 candidate pool
	- Also runs GA from random initialization
	- Combines both pools and selects one final strategy with reason
- Module 4: Sentiment classifier
- Module 5: Reinforcement learning position sizing

## Constraints

- Keep module boundaries clean and testable.
- Prefer small, safe changes over broad rewrites.
- Do not change README unless explicitly requested.

## Agent Workflow

1. Confirm target module behavior from project docs and current code.
2. Propose a short plan when work is non-trivial.
3. Implement approved changes.
4. Run relevant unit and integration tests.
5. Report only meaningful findings and risks.

## Useful Test Commands

- Module 1 and 2: python -m pytest unit_tests/module_1_knowledge_base unit_tests/module_2_strategy_search integration_tests/module_2 -q
- Module 3: python -m pytest unit_tests/module_3_evolution unit_tests/module_3_strategy_selection integration_tests/module_3 -q

## Key References

- Project Instructions: https://csc-343.path.app/projects/project-2-ai-system/ai-system.project.md
- Code elegance rubric: https://csc-343.path.app/rubrics/code-elegance.rubric.md
- Course schedule: https://csc-343.path.app/resources/course.schedule.md
- Rubric: https://csc-343.path.app/projects/project-2-ai-system/ai-system.rubric.md
