# RD-Agent Project Memory

## Project Overview

RD-Agent is an automated R&D framework for financial machine learning, supporting multiple scenarios including Qlib (quantitative trading) and Kaggle competitions.

## Current Work Context (2025-12-01)

### Architecture Refactoring - In Progress

We are executing a three-phase architectural refactoring plan documented in `ARCHITECTURE_REFACTOR_PLAN.md`.

#### Phase 1: `sub_results` Consolidation - COMPLETED ✅

**Goal:** Remove unused `sub_results` field from base `Experiment` class.

**Changes Made:**
- `rdagent/core/experiment.py` - Removed `self.sub_results` initialization (was lines 418-420)
- `rdagent/scenarios/kaggle/experiment/kaggle_experiment.py` - Added `sub_results` to `KGModelExperiment` and `KGFactorExperiment`
- `rdagent/core/proposal.py:165-172` - Fixed docstring for `get_sota_hypothesis_and_experiment()`

**Verification:** All `sub_results` usages are in Kaggle-specific code only:
- `rdagent/scenarios/kaggle/developer/runner.py:54,87,127` - Sets `exp.sub_results`
- `rdagent/scenarios/kaggle/developer/feedback.py:87,105` - Reads `exp.sub_results`

#### Phase 2a: Deprecation Warnings - COMPLETED ✅

**Goal:** Add deprecation warnings for `version` field before full removal.

**Changes Made:**
- `rdagent/core/experiment.py:35-57` - Added `DeprecationWarning` when `version != 1`
- `rdagent/core/experiment.py:10` - Added `import warnings`
- `rdagent/oai/llm_utils.py:9-11` - Added `md5_hash` re-export for backward compatibility

**Key Discovery:** The `version` field (1=Qlib, 2=Kaggle) controls execution behavior in:
- `rdagent/components/coder/factor_coder/factor.py:130,140,153,155` - Data path, execution template
- `rdagent/components/coder/model_coder/model.py:113,124,135` - Environment, code template
- `rdagent/components/coder/factor_coder/eva_utils.py:399,402,416` - Evaluator selection

Note: `evolving_version` in CoSTEER is a **different concept** - it controls knowledge base format, not execution.

#### Phase 2b-c: Scenario-Specific Workspaces - PENDING

**Goal:** Replace version-based branching with scenario-specific workspace subclasses.

**Approach:**
1. Create `QlibFactorFBWorkspace` and `KGFactorFBWorkspace`
2. Create `QlibModelFBWorkspace` and `KGModelFBWorkspace`
3. Move version-specific execute() logic to subclasses
4. Remove `version` parameter from `Task`

**Risk Level:** Medium - Constructor signatures will change

#### Phase 3: `self.scen` Removal - PENDING

**Goal:** Remove redundant `self.scen` storage from Gen/Feedback classes; access via `trace.scen` instead.

**Risk Level:** High - Requires changes to 50+ files

### Poetiq Exploration-Based R&D Mode - COMPLETED ✅

Implemented Poetiq strategies from ARC-AGI solver as an alternative to the SOTA paradigm.

**Paradigm Shift:**
| Aspect | Standard Mode | Poetiq Mode |
|--------|---------------|-------------|
| Goal | Beat the SOTA | Explore until threshold met |
| Feedback | Binary decision | Soft score 0.0-1.0 |
| Context | SOTA experiment | Top-K experiments + trajectory |
| Prompts | "surpass SOTA" | "diverse exploration" |
| Exit | Max iterations | Early exit on threshold |

**New Module:** `rdagent/components/poetiq/`
- `conf.py` - `PoetiqSettings` with `POETIQ_` env prefix
- `feedback.py` - `SoftScore`, `ScoredHypothesisFeedback`, `compute_soft_score()`
- `selection.py` - `StochasticSOTASelector`, `ConsensusVotingSelector`
- `exploration.py` - `ParallelHypothesisGen`
- `early_exit.py` - `EarlyExitChecker`
- `trajectory.py` - `TrajectoryFormatter`

**Integration Points:**
- `rdagent/components/workflow/rd_loop.py:49-50,220-227` - Early exit checker + parallel hypothesis generation buffer
- `rdagent/core/proposal.py:171-199` - Poetiq selectors hook SOTA retrieval (consensus → stochastic)
- `rdagent/scenarios/qlib/proposal/model_proposal.py:20-119` - Poetiq context preparation
- `rdagent/scenarios/qlib/developer/feedback.py:134-305` - Scored feedback generation
- `rdagent/scenarios/qlib/poetiq_prompts.yaml` - Exploration-focused prompts

**Configuration:**
```bash
POETIQ_ENABLED=true                    # Enable Poetiq mode
POETIQ_SCORE_THRESHOLD=0.5             # Decision threshold
POETIQ_EARLY_EXIT_METRIC=IC            # Metric to monitor
POETIQ_EARLY_EXIT_THRESHOLD=0.05       # Exit when IC >= 0.05
POETIQ_EARLY_EXIT_DIRECTION=higher     # "higher" or "lower"
POETIQ_STOCHASTIC_SOTA_K=3             # Sample from top-K
POETIQ_CONSENSUS_ENABLED=false         # Cluster voting
```

**Tests:** 35 tests in `test/poetiq/`

### Monorepo Lab Integration (qlib-quant-lab folded in) - IN PROGRESS

**Goal:** Consolidate the Qlib orchestration/lab tools into RD-Agent as a first-class module managed via uv extras.

**What was added:**
- New package `rdagent_lab/` (CLI, services, research adapter, models, analytics, experimental templates).
  - CLI: `rdagent lab ...` (and `rdagent-lab` script) with subcommands for train/backtest/research/live.
  - Services: Qlib-backed training (`services/training.py`) and Qlib/VectorBT backtests (`services/backtest.py`).
  - Research adapter: `research/rdagent_adapter.py` calls `rdagent.app.qlib_rd_loop.quant` directly (no subprocess parsing).
  - Experimental templates: `research/{symplectic_templates,scenario_generator}.py`.
  - Model registry wrappers: LightGBM/Transformer in `models/traditional` and `models/deep`.
- Main CLI wired to include `lab` Typer subcommand.
- Packaging/tooling updated to include `rdagent_lab`; new optional dependency bundles in `requirements/` for `quant_lab`, `backtest`, `rl`, `llm`, `live`, `simulation`, `api`.
- README documents the experimental Lab CLI and install/run examples.

**How to use (dev):**
- Install extras: `uv pip install -e .[quant-lab,backtest,rl,llm]` (add `simulation`/`api`/`live` as needed).
- Run: `rdagent lab train model --model lgbm --features Alpha158`, `rdagent lab backtest vectorbt <preds> <prices>`, `rdagent lab research quant --iterations 1`.

**Open items:**
- Migrate remaining configs/templates/docs from the old qlib-quant-lab repo into `configs/` and docs.
- Replace any remaining subprocess-based RD-Agent invocations with the adapter pattern.
- Add smoke tests for the lab CLI (typer missing in base env; extras required).

### Multi-Provider Subscription Proxy Support - VERIFIED WORKING ✅

Added support for using subscription-based AI services (Claude Max, ChatGPT Pro, Gemini) via CLIProxyAPI.

**Files Modified:**
- `rdagent/oai/llm_conf.py:129-145` - Added proxy settings
- `rdagent/oai/backend/litellm.py:55-86,104-134,202-214` - Added proxy configuration + embedding bypass

**Key Configuration:**
```bash
USE_SUBSCRIPTION_PROXY=true
SUBSCRIPTION_PROXY_URL=http://localhost:8317/v1
CHAT_MODEL=openai/claude-opus-4-5-20251101  # Use openai/ prefix for CLIProxyAPI

# Embeddings bypass (CLIProxyAPI doesn't support embeddings)
EMBEDDING_OPENAI_API_KEY=sk-proj-xxx
EMBEDDING_MODEL=text-embedding-3-small
```

## Architecture Notes

### Workspace Hierarchy

Two levels of workspaces exist:

1. **Scenario-level workspaces** (already scenario-specific):
   - `QlibFBWorkspace` in `rdagent/scenarios/qlib/experiment/workspace.py`
   - `KGFBWorkspace` in `rdagent/scenarios/kaggle/experiment/workspace.py`
   - Handle experiment-level execution (backtest, submission scoring)

2. **Component-level workspaces** (have version branching - needs refactoring):
   - `FactorFBWorkspace` in `rdagent/components/coder/factor_coder/factor.py`
   - `ModelFBWorkspace` in `rdagent/components/coder/model_coder/model.py`
   - Handle code testing/validation during development

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `Task` | `rdagent/core/experiment.py` | Base task with `version` field (deprecated) |
| `Experiment` | `rdagent/core/experiment.py` | Base experiment container |
| `Trace` | `rdagent/core/proposal.py` | Experiment history with DAG structure |
| `FBWorkspace` | `rdagent/core/experiment.py` | File-based workspace for code execution |
| `RDLoop` | `rdagent/components/workflow/rd_loop.py` | Main R&D loop workflow |
| `SoftScore` | `rdagent/components/poetiq/feedback.py` | Continuous score [0-1] with components |
| `ScoredHypothesisFeedback` | `rdagent/components/poetiq/feedback.py` | Feedback with soft scoring |
| `EarlyExitChecker` | `rdagent/components/poetiq/early_exit.py` | Threshold-based loop termination |
| `TrajectoryFormatter` | `rdagent/components/poetiq/trajectory.py` | Format experiment history for LLM |

## Development Commands

```bash
# Linting
uv run ruff check rdagent/

# Import verification
uv run python -c "from rdagent.core.experiment import Task; from rdagent.core.proposal import Trace"

# Test deprecation warning
uv run python -c "
import warnings
warnings.filterwarnings('always', category=DeprecationWarning)
from rdagent.components.coder.factor_coder.factor import FactorTask
t = FactorTask('test', 'test', 'test', version=2)
"
```

## Files Modified in Current Session

| File | Changes |
|------|---------|
| `rdagent/core/experiment.py` | Deprecation warning for `version`, removed `sub_results` |
| `rdagent/core/proposal.py` | Fixed `get_sota_hypothesis_and_experiment` docstring |
| `rdagent/scenarios/kaggle/experiment/kaggle_experiment.py` | Added `sub_results`, fixed type annotations |
| `rdagent/oai/llm_utils.py` | Added `md5_hash` re-export |
| `ARCHITECTURE_REFACTOR_PLAN.md` | Updated with findings and completion status |
| `rdagent/components/poetiq/*` | **NEW** - Full Poetiq module (7 files) |
| `rdagent/components/workflow/rd_loop.py` | Early exit + Poetiq parallel hypothesis buffering |
| `rdagent/core/proposal.py` | Poetiq selectors for SOTA retrieval |
| `rdagent/scenarios/qlib/proposal/model_proposal.py` | Poetiq context preparation |
| `rdagent/scenarios/qlib/developer/feedback.py` | Scored feedback generation + resilient metric handling |
| `rdagent/scenarios/qlib/poetiq_prompts.yaml` | **NEW** - Exploration-focused prompts (safe result rendering) |
| `test/poetiq/*` | **NEW** - 36 unit tests for Poetiq components |
| `rdagent/app/cli.py` | Added `lab` Typer subcommand for integrated lab CLI |
| `rdagent_lab/**` | **NEW** - Lab package (CLI, services, adapter, models, analytics, templates) |
| `requirements/{quant_lab,backtest,rl,llm,live,simulation,api}.txt` | **NEW** optional dependency bundles |
| `pyproject.toml` | Included `rdagent_lab` in packaging/tooling; registered `rdagent-lab` script and extras |
| `README.md` | Documented experimental lab CLI and install examples |
