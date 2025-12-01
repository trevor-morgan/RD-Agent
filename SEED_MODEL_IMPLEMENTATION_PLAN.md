# RD-Agent Seed Model Implementation Plan

## Overview

This plan adds the ability to provide your own model as a "seed" for RD-Agent's evolutionary loop.
Instead of starting from scratch, the LLM will analyze your model and try to improve it.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Current RD-Agent Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Loop 0: LLM generates simple model → Train → Evaluate → SOTA if good       │
│  Loop 1: LLM tries to beat SOTA → Train → Evaluate → Update SOTA            │
│  Loop N: Continue...                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           New Flow with Seed Model                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Pre-Loop: Load user's model → Train → Evaluate → Inject as SOTA            │
│  Loop 0: LLM sees user's model as baseline → Proposes improvements          │
│  Loop 1: LLM tries to beat improved version → Train → Evaluate              │
│  Loop N: Continue evolving from user's architecture...                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Changes Required

### 1. US Market Configuration (New Files)

**Files to create:**
- `rdagent/scenarios/qlib/experiment/model_template/conf_us_baseline_model.yaml`
- `rdagent/scenarios/qlib/experiment/model_template/conf_us_sota_model.yaml`

**Key differences from Chinese configs:**
- `provider_uri: ~/.qlib/qlib_data/alpaca_us`
- `region: us`
- `market: sp500`
- `benchmark: SPY` (or ^GSPC)
- Date ranges: 2020-2025 (Alpaca data range)

### 2. Data Region Environment Variable

**File:** `rdagent/utils/env.py`

**Changes:**
- Add `QLIB_DATA_REGION` environment variable support
- Default to `cn_data` for backwards compatibility
- Support: `cn_data`, `us_data`, `alpaca_us`

**Location:** `QTDockerEnv.prepare()` method (around line 964)

### 3. CLI Parameter for Seed Model

**File:** `rdagent/app/qlib_rd_loop/model.py`

**New parameters:**
```python
def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    seed_model: str | None = None,      # NEW: Path to model.py
    seed_hypothesis: str | None = None,  # NEW: Description of the model
    data_region: str = "cn_data",        # NEW: Data region to use
):
```

### 4. Seed Model Loader

**File:** `rdagent/components/workflow/rd_loop.py`

**New method:**
```python
def load_seed_model(
    self,
    model_path: str,
    hypothesis_text: str,
    data_region: str = "alpaca_us"
) -> None:
    """
    Load user's model as initial SOTA in the trace.

    1. Read model.py code
    2. Create QlibModelExperiment with the code
    3. Run through QlibModelRunner to get metrics
    4. Create HypothesisFeedback with decision=True
    5. Add to trace.hist
    """
```

### 5. Configuration Setting

**File:** `rdagent/app/qlib_rd_loop/conf.py`

**New settings:**
```python
class ModelBasePropSetting(BasePropSetting):
    # ... existing settings ...

    # Seed model settings
    seed_model_path: str | None = None
    seed_hypothesis: str | None = None
    data_region: str = "cn_data"  # cn_data, us_data, alpaca_us
```

### 6. SymplecticModel Wrapper (Example)

**File:** `qlib-quant-lab/scripts/rdagent_seed_models/symplectic_seed.py`

Wraps SymplecticNet in RD-Agent's expected interface:
```python
import torch.nn as nn

class Net(nn.Module):
    """RD-Agent compatible wrapper for SymplecticNet."""

    def __init__(self, num_features, num_timesteps=None):
        super().__init__()
        self.model = SymplecticNet(
            d_feat=num_features,
            d_model=128,
            n_heads=4,
            n_layers=3,
            # ... config
        )

    def forward(self, x):
        return self.model(x)
```

## Implementation Order

1. **Phase 1: US Market Config** - Create YAML files (independent, can test immediately)
2. **Phase 2: Data Region Env Var** - Modify env.py (needed for Phase 1 to work)
3. **Phase 3: CLI Parameters** - Modify model.py and conf.py
4. **Phase 4: Seed Model Loader** - Core logic in rd_loop.py
5. **Phase 5: SymplecticModel Wrapper** - Create wrapper for testing
6. **Phase 6: Integration Test** - Test full flow

## Testing Strategy

### Test 1: US Config Only
```bash
# Verify US config works without seed model
export QLIB_DATA_REGION=alpaca_us
rdagent fin_model --loop-n 1
```

### Test 2: Seed Model
```bash
# Test with SymplecticModel as seed
rdagent fin_model \
    --seed-model ./symplectic_seed.py \
    --seed-hypothesis "Symplectic physics-informed model with Hurst exponent" \
    --data-region alpaca_us \
    --loop-n 3
```

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Seed model fails to train | Add fallback to start fresh |
| US data format incompatible | Test Alpha158 with Alpaca data first |
| Docker timeout on seed eval | Use longer timeout for seed phase |
| LLM doesn't understand architecture | Include detailed hypothesis description |

## Success Criteria

1. ✅ US market data works with RD-Agent
2. ✅ Seed model trains and evaluates successfully
3. ✅ Seed model appears as SOTA in LLM context
4. ✅ LLM proposes meaningful improvements to seed architecture
5. ✅ At least one evolved model beats seed model's Sharpe
