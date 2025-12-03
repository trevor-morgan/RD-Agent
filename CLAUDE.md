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

### Docker Workflow Improvements - COMPLETED ✅

Implemented auto-cleanup, build caching, and smart rebuild detection for Docker workflow.

**Problem Solved:** Before these changes, `prepare()` triggered a full Docker rebuild on EVERY workspace execution - 4+ rebuilds per 23 minutes, creating 170GB+ of dangling images.

**Features:**
1. **Smart rebuild detection** - Skip build if image exists and Dockerfile unchanged (hash-based)
2. **Build lock** - Prevent concurrent builds from multiple processes
3. **BuildKit enabled** - Parallel layer builds, better caching
4. **Auto-cleanup ON by default** - Pre-build (dangling images) + post-run (stopped containers)
5. **Consolidated Dockerfiles** - 17 RUN commands → 1 for better layer caching
6. **.dockerignore files** - Reduce build context 50-80%
7. **CLI cleanup command** - `rdagent cleanup` for manual cleanup

**New Files:**
- `rdagent/utils/docker_cleanup.py` - `DockerCleanupManager`, `pre_build_cleanup()`, `post_run_cleanup()`
- `rdagent/app/utils/docker_cleanup_cli.py` - CLI subcommand
- 4 `.dockerignore` files in docker directories

**Key Functions in `rdagent/utils/env.py`:**
- `compute_dockerfile_hash()` - Hash Dockerfile for change detection
- `should_rebuild_image()` - Decide if rebuild needed (checks hash label)
- `DockerBuildLock` - File-based lock to prevent concurrent builds
- Image labels: `rdagent.dockerfile.hash` stores hash for smart caching

**Modified Files:**
- `rdagent/utils/env.py:72-204` - New helper functions and DockerBuildLock class
- `rdagent/utils/env.py:665-667` - Added `skip_build_if_exists`, `smart_rebuild` to `DockerConf`
- `rdagent/utils/env.py:938-1001` - Smart rebuild + build lock in `DockerEnv.prepare()`
- `rdagent/app/cli.py` - Added `cleanup` subcommand
- 3 Dockerfiles - Consolidated RUN commands

**CLI Usage:**
```bash
rdagent cleanup status           # Show disk usage
rdagent cleanup all              # Clean dangling images + stopped containers
rdagent cleanup all --cache      # Also clean build cache
rdagent cleanup all --images     # Also remove local_* images
rdagent cleanup all --dry-run    # Preview without cleaning
```

**Configuration (all default ON, disable if needed):**
```bash
DOCKER_USE_BUILDKIT=false
DOCKER_AUTO_CLEANUP_BEFORE_BUILD=false
DOCKER_AUTO_CLEANUP_AFTER_RUN=false
DOCKER_SKIP_BUILD_IF_EXISTS=false    # Force rebuild even if image exists
DOCKER_SMART_REBUILD=false           # Disable hash-based change detection
```

**Expected Impact:**
| Metric | Before | After |
|--------|--------|-------|
| Builds per run | 4+ (every prepare()) | 1 (only when Dockerfile changes) |
| Dangling images | ~170GB accumulated | Auto-cleaned |
| Build time (cache hit) | Full rebuild | <1 second (skip) |

### Monorepo Lab Integration (qlib-quant-lab folded in) - COMPLETED ✅

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

### qlib-quant-lab Components Ported (2025-12-02) - COMPLETED ✅

Consolidated ~3,500 lines of production-ready components from qlib-quant-lab into rdagent_lab.

**Ported Components:**

| Module | Components | Lines | Description |
|--------|------------|-------|-------------|
| `rdagent_lab/models/novel/` | `SymplecticNet`, `MarketStateNet` | ~700 | Physics-informed neural networks |
| `rdagent_lab/models/ensemble/` | `PerformanceTracker`, `DiversityRegularizer`, `WeightingMethod` | ~250 | Adaptive ensemble utilities |
| `rdagent_lab/research/` | `MonteCarloSimulator`, `ScenarioResult`, `StressScenario` | ~300 | Monte Carlo simulation for risk |
| `rdagent_lab/data/` | `AlpacaCollector`, `DataFormatConverter` | ~600 | Data collection and format conversion |
| `rdagent_lab/utils/` | `extract_features_and_labels`, `validate_data_for_training` | ~200 | Qlib data extraction helpers |
| `seed_models/` | `simple_gru.py`, `symplectic_net.py`, `market_state_net.py` | ~700 | RD-Agent seed models |

**Novel Models (Pure PyTorch nn.Module):**

1. **SymplecticNet** (`rdagent_lab/models/novel/symplectic.py`):
   - `FractionalDifferencer`: Rough volatility with Hurst exponent H~0.12
   - `SymplecticAttention`: Phase space volume preserving attention
   - `HamiltonianBlock`: Energy-conserving latent evolution
   - `HolographicMemory`: Complex-valued associative memory for regime detection

2. **MarketStateNet** (`rdagent_lab/models/novel/market_state.py`):
   - `TopologyFeatures`: Betti numbers from correlation networks
   - `CompressionComplexity`: zlib compression ratio as predictability proxy
   - `ActivityMetrics`: Aggregate momentum/volatility across assets
   - `QuantilePredictor`: Pinball loss for uncertainty quantification
   - `AnomalyDetector`: Reconstruction-based anomaly detection

**Ensemble Utilities** (`rdagent_lab/models/ensemble/adaptive_ensemble.py`):
- `WeightingMethod`: Enum for equal, IC-weighted, Sharpe-weighted, learned
- `PerformanceTracker`: EMA-based performance tracking with rolling IC/Sharpe
- `DiversityRegularizer`: Penalizes correlated predictions, encourages diversity

**Monte Carlo Simulator** (`rdagent_lab/research/monte_carlo.py`):
- `MonteCarloSimulator`: Scenario simulation with VaR/CVaR computation
- `StressScenario`: Configurable shock scenarios (magnitude, duration, affected features)
- `STANDARD_STRESS_SCENARIOS`: Predefined market_crash, volatility_spike, flash_crash, gradual_decline

**Data Collection** (`rdagent_lab/data/`):
- `AlpacaCollector`: Download S&P 500 data from Alpaca Markets, convert to Qlib format
- `DataFormatConverter`: Fix broken Alpaca data format (start_index prefix issue)
- `SP500_SYMBOLS`: Complete list of S&P 500 constituents (as of 2024)

**Data Extraction** (`rdagent_lab/utils/`):
- `extract_features_and_labels()`: Safely extract (X, y) from Qlib DatasetH
- `validate_data_for_training()`: Check for NaN, shape, minimum samples
- `DataExtractionError`: Custom exception with helpful messages

**Seed Models** (`rdagent/scenarios/qlib/experiment/model_template/seed_models/`):

Use these with the `--seed-model` CLI option:

```bash
# Simple GRU baseline
rdagent fin_model \
    --seed-model ./seed_models/simple_gru.py \
    --seed-hypothesis "Simple 2-layer GRU with attention pooling" \
    --loop-n 5

# Full SymplecticNet (Sharpe ~0.89 on S&P 500)
rdagent fin_model \
    --seed-model ./seed_models/symplectic_net.py \
    --seed-hypothesis "Symplectic physics-informed model with Hamiltonian dynamics" \
    --loop-n 5

# MarketStateNet with topological features
rdagent fin_model \
    --seed-model ./seed_models/market_state_net.py \
    --seed-hypothesis "Market state detection with topological features and compression complexity" \
    --loop-n 5
```

**Usage Examples:**

```python
# Novel models
from rdagent_lab.models.novel import SymplecticNet, MarketStateNet
model = SymplecticNet(d_feat=20, d_model=128, n_heads=4)
pred = model(torch.randn(32, 20, 20))

# Ensemble utilities
from rdagent_lab.models.ensemble import PerformanceTracker, WeightingMethod
tracker = PerformanceTracker(n_models=3, lookback=60)
tracker.update(model_idx=0, predictions=preds, labels=labels)
weights = tracker.get_weights(WeightingMethod.IC_WEIGHTED)

# Monte Carlo simulation
from rdagent_lab.research import MonteCarloSimulator, STANDARD_STRESS_SCENARIOS
simulator = MonteCarloSimulator(model)
result = simulator.simulate(initial_state, n_scenarios=1000, n_steps=20)
print(f"VaR(95%): {result.var_95:.4f}, CVaR: {result.cvar_95:.4f}")
stress_results = simulator.stress_test(initial_state, STANDARD_STRESS_SCENARIOS)

# Data collection
from rdagent_lab.data import AlpacaCollector
collector = AlpacaCollector()
collector.download(start="2020-01-01", end="2025-11-26")

# Data extraction
from rdagent_lab.utils import extract_features_and_labels, validate_data_for_training
x_train, y_train = extract_features_and_labels(dataset, "train")
validate_data_for_training(x_train, y_train, min_samples=100)
```

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

**Available Models via CLIProxyAPI:**

| Provider | Model Names | Notes |
|----------|-------------|-------|
| Claude Max | `openai/claude-opus-4-5-20251101`, `openai/claude-sonnet-4-5-20250929` | Use `CHAT_TEMPERATURE=1` for thinking models |
| ChatGPT Pro | `openai/gpt-5.1-codex-max-xhigh` | Includes reasoning tokens |
| Gemini (AI Ultra) | `openai/gemini-3-pro-preview`, `openai/gemini-3-pro-preview-11-2025`, `openai/gemini-3-pro-preview-11-2025-thinking` | Released Nov 18, 2025 |

**Gemini 3 Pro Notes:**
- Requires **AI Ultra subscription** or paid Gemini API key
- Uses `thinking_level` parameter (`low`/`high`) for reasoning depth - no "thinking max" mode
- "Deep Think" mode (even stronger reasoning) coming for AI Ultra subscribers
- CLIProxyAPI v6.3.21+ supports updated Gemini model list

**CLIProxyAPI Setup:**
```bash
# Install
curl -fsSL https://raw.githubusercontent.com/brokechubb/cliproxyapi-installer/refs/heads/master/cliproxyapi-installer | bash

# Authenticate (from ~/cliproxyapi)
./cli-proxy-api -claude-login   # Claude Max
./cli-proxy-api -codex-login    # ChatGPT Pro
./cli-proxy-api -login          # Gemini (Google Account)

# Start proxy
./cli-proxy-api
```

### Infrastructure Updates (RunPod & Uv) - COMPLETED ✅

Optimized the project for deployment on RunPod (GPU cloud) using a fast `uv`-based backend instead of Docker-in-Docker.

**Features:**
1.  **Uv Backend** - New `UvEnv` class in `utils/env.py` replaces Conda/Docker for faster environment management.
2.  **RunPod Optimization** - Deployment strategy uses direct execution on host + SSH tunneling for proxy access.
3.  **Data Patching** - Identified stale default Qlib data (ends 2020) and patched templates to use `2010-2020` range.
4.  **Hardware Compatibility** - Updated PyTorch installation instructions to handle newer GPUs (RTX 4090/5090).

**New Files:**
- `rdagent/utils/env.py` - Added `UvEnv` and `UvConf`.
- `rdagent/components/coder/model_coder/conf.py` - Added `uv` support to factory.

**Deployment Commands:**
```bash
# Connect to RunPod + Tunnel Proxy + Launch Agent
ssh -R 38882:localhost:8317 user@pod-ip -t "cd /workspace/RD-Agent && uv run python -m rdagent.app.cli fin_model"
```

**Remote Configuration (.env):**
```ini
MODEL_CoSTEER_env_type=uv
RUNNING_TIMEOUT_PERIOD=3600
LITELLM_USE_SUBSCRIPTION_PROXY=true
LITELLM_SUBSCRIPTION_PROXY_URL=http://localhost:38882/v1
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
| `DockerCleanupManager` | `rdagent/utils/docker_cleanup.py` | Docker resource cleanup (images, containers, cache) |

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

# Docker cleanup commands
rdagent cleanup status           # Show Docker disk usage
rdagent cleanup all --dry-run    # Preview cleanup
rdagent cleanup all              # Clean dangling images + stopped containers
rdagent cleanup all --cache      # Also clean build cache
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
| `rdagent/app/cli.py` | Added `lab` and `cleanup` Typer subcommands |
| `rdagent_lab/**` | **NEW** - Lab package (CLI, services, adapter, models, analytics, templates) |
| `requirements/{quant_lab,backtest,rl,llm,live,simulation,api}.txt` | **NEW** optional dependency bundles |
| `pyproject.toml` | Included `rdagent_lab` in packaging/tooling; registered `rdagent-lab` script and extras |
| `README.md` | Documented experimental lab CLI and install examples |
| `rdagent/utils/docker_cleanup.py` | **NEW** - Docker cleanup module with `DockerCleanupManager` |
| `rdagent/app/utils/docker_cleanup_cli.py` | **NEW** - CLI for `rdagent cleanup` commands |
| `rdagent/utils/env.py` | Added BuildKit + auto-cleanup config to `DockerConf`, modified `DockerEnv` |
| `rdagent/scenarios/*/docker/.dockerignore` | **NEW** - 4 .dockerignore files to reduce build context |
| `rdagent/scenarios/*/docker/Dockerfile` | Consolidated RUN commands for better layer caching |