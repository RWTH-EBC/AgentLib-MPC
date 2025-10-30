# GitHub Copilot Review Agent Instructions

## Repository Overview

**AgentLib-MPC** provides model-predictive-control (MPC) and distributed-MPC building blocks that extend the AgentLib ecosystem and can also be used as a focused standalone library for MPC workflows. It contains datatypes, CasADi-based models and predictors, MPC controller modules, and optimization backend adapters (CasADi-based solvers and mixed-integer backends).

### Key Information
- **Language**: Python (3.9-3.12)
- **License**: BSD-3-Clause (see `LICENSE`)
- **Testing**: pytest (unit tests live in `tests/`)
- **Code Quality**: pylint (project contains `pylintrc`; aim: high score)
- **Dependencies**: Core (numpy, casadi, pydantic>=2.0, scipy, pandas), Optional (FMU support, machine-learning backends)
- **CI/CD**: GitHub Actions (runs pytest, coverage, pylint) and Sphinx documentation under `docs/`
- **Package layout**: The main package is `agentlib_mpc` with subpackages for data structures, models, modules and optimization backends.

## Project Structure

```
agentlib_mpc/
├── data_structures/        # datatypes, ADMM/coordination dataclasses, interpolation, MPC datamodels
├── models/                 # CasADi based models, predictors and serialized model wrappers
├── modules/                # MPC modules (mpc.py, mpc_full.py, minlp_mpc.py) and subpackages (dmpc, estimation, ml_model_training)
├── optimization_backends/  # backend interfaces and CasADi solver adapters
├── utils/                  # analysis, sampling, plotting helpers
tests/                      # pytest test-suite mirroring functionality
examples/                   # runnable examples and configs (ADMM, MPC examples)
docs/                       # Sphinx documentation and tutorials
```

### Critical Architecture Components
- **MPC modules** (`agentlib_mpc.modules`): controller implementations such as `mpc.py`, `mpc_full.py` and `minlp_mpc.py`. These orchestrate prediction, optimization and actuation logic and are the primary review surface for behavioral changes.
- **Models** (`agentlib_mpc.models`): CasADi-backed model builders (`casadi_model.py`, `casadi_ml_model.py`, `casadi_predictor.py`) and serialized model wrappers. Changes here directly affect optimization problem formulation and solver inputs.
- **Data structures** (`agentlib_mpc.data_structures`): typed data models (MPC datamodels, ADMM datatypes, ML model datatypes) used across modules and backends. These are typically Pydantic or typed dataclasses and form the public configuration/API contract.
- **Optimization backends** (`agentlib_mpc.optimization_backends`): backend adapter API (`backend.py`) and concrete implementations (CasADi wrappers). Changes here affect solver selection, options, and numerical behavior.
- **Examples & CI tests**: `examples/` contains runnable scripts that CI may exercise; `tests/` contains pytest tests that validate integration of models, backends and modules.

### Configuration Pattern
Most public configuration/data models use Pydantic v2 models or clear typed dataclasses (see `agentlib_mpc.data_structures`). Reviewers should validate schema changes carefully: they are part of the public contract and impact example configs and `examples/configs`.

## Review Guidelines

### Core Principle: Concise, High-Impact Reviews
**Focus only on critical issues that affect:**
- Backward compatibility and breaking changes
- CI/CD pipeline failures
- Missing tests for new functionality
- Undocumented changes to public APIs

**Avoid commenting on:**
- Minor refactoring preferences
- Well-tested internal implementation details

### Primary Objectives
1. **Provide concise PR summary** (2-3 sentences max) highlighting purpose and scope
2. **Classify changes** as bug fixes, new features, refactoring, or documentation
3. **Assess backward compatibility** - flag breaking changes to public APIs
4. **Highlight CI risks** - identify changes likely to cause test/build failures
5. **Respond in English only**

### Specific Review Focus Areas

#### 1. Breaking Changes & Compatibility
- **Flag breaking changes** to: public datamodels and configs in `agentlib_mpc.data_structures` (MPC datamodels, ADMM/coordinator types, ML model datatypes), public MPC module entrypoints (`agentlib_mpc.modules` — e.g. `mpc.py`, `mpc_full.py`, `minlp_mpc.py`), model/predictor APIs in `agentlib_mpc.models` (CasADi model builders, `casadi_predictor`), the optimization backend interface (`agentlib_mpc.optimization_backends.backend.Backend` and concrete adapters), and serialized model formats (`serialized_ml_model.py`).
- **Configuration schema changes**: Any modification to Pydantic models in `agentlib_mpc.data_structures` affects example configs and CI examples in `examples/` and `tests/`. Prefer additive, optional fields for compatibility; when renaming/removing fields, add a clear deprecation path and migration notes.
- **Solver/backend interface changes**: Changing function signatures, return shapes, option names, or semantics in `optimization_backends` is breaking — solver adapters and callers (MPC modules) rely on stable behavior.
- **Numerical & model changes**: Modifications to CasADi model formulations (variable ordering, discretization, parameter binding) or to objective/constraint construction can change controller behavior and must be treated as breaking for reproducibility.

#### 2. Code Quality & Standards
- **Pydantic usage**: Ensure v2 syntax (e.g., `Field()`, `field_validator`, not v1 `validator`)
- **Docstrings**: Check for docstrings on new public classes/methods (Google style)
- **Pylint compliance**: Flag obvious issues (unused imports, naming violations, etc.)

#### 3. Testing Requirements
- **New features require tests**: Flag new modules, functions, or classes without corresponding test files
- **Test file naming**: Should match `test_<module>.py` pattern in `tests/`
- **Coverage concerns**: Highlight complex logic added without test coverage

#### 4. Documentation & Examples
- **Type hints**: Ensure type hints are added for new functions/methods
- **Examples**: Add examples for new features and functionality

#### 5. CI/CD Considerations
- **Import errors**: New dependencies must be in `setup.cfg`/`pyproject.toml` optional-dependencies
- **Python version compatibility**: Code must work on Python 3.9-3.12
- **Test execution**: Changes to `tests/` structure or execution patterns
- **Example scripts**: `examples/` should remain functional (tested by CI)

#### 6. Version & Release
- **Version bump**: Check if `agentlib/__init__.py` `__version__` updated appropriately
- **CHANGELOG**: Verify changes documented under correct version header
- **Breaking changes**: Major version bump (0.x.0) vs minor (0.8.x)

### Review Output Format

Provide structured, concise feedback (max 10 bullet points total):
1. **Summary**: 2-3 sentence overview of PR purpose
2. **Change Classification**: Bug fix / Feature / Refactor / Documentation / Mixed
3. **Backward Compatibility**: Compatible / Breaking (explain) / Deprecation needed
4. **CI Risk**: Low / Medium / High (explain if Medium/High)
5. **Key Issues**: Numbered list of critical concerns only (max 5 items)
6. **Suggestions**: Optional - only for significant improvements

**Brevity is essential. Skip sections with nothing critical to report.**

### What NOT to Flag
- Minor style issues (handled by pylint)
- Personal preference on implementation approach
- Overly detailed nitpicking on internal/private methods
- Changes to examples or test code (unless broken)

## Common Patterns to Recognize

- **CasADi model builders**: Look for symbolic variable creation, parameter binding, and builder functions in `agentlib_mpc.models` (e.g., `casadi_model.py`, `casadi_ml_model.py`, `casadi_predictor.py`). Pay attention to variable ordering and parameter/shape conventions.
- **Serialized ML models**: `serialized_ml_model.py` and related datatypes define formats for saving/loading learned models (ANN, GPR, linreg). Ensure deserialization compatibility and shape/feature-order consistency.
- **Optimization backend adapters**: `agentlib_mpc.optimization_backends.backend` defines the abstract backend API; concrete adapters in `optimization_backends/casadi_/` convert problem definitions to solver calls. Watch for option name changes and differences in solver return formats.
- **MPC modules composition**: `mpc.py`, `mpc_full.py`, and `minlp_mpc.py` assemble predictors, objectives, constraints, and backends. These modules are high-impact review targets for correctness, numerical stability, and API usage.
- **ADMM / distributed coordination patterns**: Check `agentlib_mpc.data_structures.coordinator_datatypes` and `modules/dmpc/` for message shapes, partitioning logic, and convergence parameter conventions.
- **Examples & CI expectations**: `examples/` contains runnable scripts and `configs/` that CI may exercise; mismatches between examples and code (config names, default solver options) often cause failures.
- **Config injection**: Modules receive configuration via Pydantic datamodels in `agentlib_mpc.data_structures` — validate field names, defaults, and validators when reviewing changes.

---
*Instructions version: 1.0 | Target: GitHub Copilot Review Agent | Max length: 2 pages*
