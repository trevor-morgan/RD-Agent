# RD-Agent Core Architecture Refactor Plan

## Overview

This document outlines a three-phase refactoring plan to address long-term architectural debt in the RD-Agent core module.

---

## Phase 1: `sub_results` Consolidation (Quick Wins)

### Goal
Remove unused `sub_results` field from base `Experiment` class and move it to Kaggle-specific classes only.

### Current State
```python
# rdagent/core/experiment.py:418-420
self.sub_results: dict[str, float] = (
    {}
)  # TODO: in Kaggle, now sub results are all saved in self.result, remove this in the future.
```

### Changes Required

#### 1. Remove from `rdagent/core/experiment.py`
- Delete lines 418-420 (`self.sub_results` initialization)
- Remove the TODO comment

#### 2. Add to Kaggle experiment classes
- `rdagent/scenarios/kaggle/experiment/kaggle_experiment.py`
  - Add `self.sub_results: dict[str, float] = {}` to `KGFactorExperiment.__init__`
  - Add `self.sub_results: dict[str, float] = {}` to `KGModelExperiment.__init__`

#### 3. Verify usages (Kaggle only)
- `rdagent/scenarios/kaggle/developer/runner.py:54,87,127` - Sets `exp.sub_results`
- `rdagent/scenarios/kaggle/developer/feedback.py:87,105` - Reads `exp.sub_results`

#### 4. Fix docstring in `rdagent/core/proposal.py:165-172`
```python
# Current (incorrect):
def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
    """Access the last experiment result, sub-task, and the corresponding hypothesis."""
    # Returns 2 items, docstring says 3

# Fixed:
def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
    """Get the SOTA hypothesis and experiment from trace history."""
```

### Risk Assessment
- **Risk Level**: Low
- **Breaking Changes**: None (Kaggle classes will have the field, others won't use it)
- **Testing**: Run Kaggle and Qlib scenarios to verify

---

## Phase 2: Execution Strategy Unification

### Goal
Replace version-based branching with Strategy pattern for scenario-specific execution.

### Current State
```python
# Multiple files have version checks:
if self.target_task.version == 1:  # Qlib
    ...
elif self.target_task.version == 2:  # Kaggle
    ...
```

### New Architecture

#### 1. Create `rdagent/core/executor.py`
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from pathlib import Path

@dataclass
class ExecutionResult:
    """Standardized execution result."""
    stdout: str
    stderr: str
    exit_code: int
    metrics: dict[str, float]
    raw_data: Any = None

class ExecutionStrategy(ABC):
    """Abstract strategy for scenario-specific execution."""

    @abstractmethod
    def get_environment(self) -> "Env":
        """Get the execution environment (Docker, Conda, etc.)."""

    @abstractmethod
    def prepare_workspace(self, workspace: "FBWorkspace") -> None:
        """Prepare workspace with scenario-specific files."""

    @abstractmethod
    def get_entry_command(self, workspace: "FBWorkspace") -> str:
        """Get the command to execute."""

    @abstractmethod
    def parse_results(self, raw_output: str, workspace: "FBWorkspace") -> ExecutionResult:
        """Parse execution output into standardized result."""
```

#### 2. Implement strategies

**`rdagent/scenarios/qlib/execution.py`**
```python
class QlibExecutionStrategy(ExecutionStrategy):
    def get_environment(self) -> Env:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            return QTDockerEnv()
        return QlibCondaEnv(conf=QlibCondaConf())

    def prepare_workspace(self, workspace: FBWorkspace) -> None:
        # Link Qlib data, create YAML configs
        pass

    def get_entry_command(self, workspace: FBWorkspace) -> str:
        return "qrun conf.yaml && python read_exp_res.py"

    def parse_results(self, raw_output: str, workspace: FBWorkspace) -> ExecutionResult:
        # Parse Qlib metrics from output
        pass
```

**`rdagent/scenarios/kaggle/execution.py`**
```python
class KaggleExecutionStrategy(ExecutionStrategy):
    def __init__(self, competition: str):
        self.competition = competition

    def get_environment(self) -> Env:
        return KGDockerEnv(self.competition)

    def prepare_workspace(self, workspace: FBWorkspace) -> None:
        # Mount competition data, inject templates
        pass

    def get_entry_command(self, workspace: FBWorkspace) -> str:
        return "python train.py"

    def parse_results(self, raw_output: str, workspace: FBWorkspace) -> ExecutionResult:
        # Parse submission score from CSV
        pass
```

#### 3. Refactor workspace classes

```python
# rdagent/core/experiment.py
class FBWorkspace:
    def __init__(self, strategy: ExecutionStrategy | None = None, ...):
        self.strategy = strategy

    def execute(self) -> ExecutionResult:
        if self.strategy is None:
            raise ValueError("No execution strategy configured")

        env = self.strategy.get_environment()
        self.strategy.prepare_workspace(self)
        entry = self.strategy.get_entry_command(self)

        raw_output = env.run(entry, str(self.workspace_path))
        return self.strategy.parse_results(raw_output, self)
```

#### 4. Remove version field
- Delete `version` parameter from `AbsTask.__init__`
- Remove all `version == 1` / `version == 2` checks

### Files to Modify
- `rdagent/core/experiment.py` - Remove version, add strategy
- `rdagent/core/executor.py` - New file with abstractions
- `rdagent/components/coder/factor_coder/factor.py` - Use strategy
- `rdagent/components/coder/model_coder/model.py` - Use strategy
- `rdagent/scenarios/qlib/execution.py` - New strategy implementation
- `rdagent/scenarios/kaggle/execution.py` - New strategy implementation
- `rdagent/scenarios/qlib/experiment/workspace.py` - Integrate strategy
- `rdagent/scenarios/kaggle/experiment/workspace.py` - Integrate strategy

### Risk Assessment
- **Risk Level**: Medium
- **Breaking Changes**: Constructor signatures change
- **Testing**: Full integration tests for both scenarios

---

## Phase 3: `self.scen` Duplication Removal

### Goal
Remove redundant `self.scen` storage from Gen/Feedback classes; access via `trace.scen` instead.

### Current State
```python
# Every Gen class stores scen redundantly
class HypothesisGen(ABC):
    def __init__(self, scen: Scenario) -> None:
        self.scen = scen  # REDUNDANT - trace also has scen

    def gen(self, trace: Trace) -> Hypothesis:
        # Could use trace.scen instead of self.scen
        pass
```

### Migration Strategy

#### Step 1: Add deprecation warnings
```python
class HypothesisGen(ABC):
    def __init__(self, scen: Scenario | None = None) -> None:
        if scen is not None:
            warnings.warn(
                "Passing scen to __init__ is deprecated. Use trace.scen in methods.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._scen = scen  # Keep for backward compat

    @property
    def scen(self) -> Scenario:
        warnings.warn("self.scen is deprecated. Use trace.scen.", DeprecationWarning)
        return self._scen
```

#### Step 2: Update base class signatures
```python
# rdagent/core/proposal.py

class HypothesisGen(ABC):
    """Generate hypotheses for experiments."""

    # NO __init__ with scen parameter

    @abstractmethod
    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        """Generate hypothesis using trace.scen for scenario info."""

class Experiment2Feedback(ABC):
    """Generate feedback from experiment results."""

    # NO __init__ with scen parameter

    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> ExperimentFeedback:
        """Generate feedback using trace.scen for scenario info."""
```

#### Step 3: Update all subclasses

**Pattern for each subclass:**
```python
# Before
class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self.targets = "hypothesis_and_experiment"

    def gen(self, trace: Trace, ...) -> Hypothesis:
        scenario_desc = self.scen.get_scenario_all_desc()  # Uses self.scen
        ...

# After
class LLMHypothesisGen(HypothesisGen):
    def __init__(self) -> None:
        self.targets = "hypothesis_and_experiment"

    def gen(self, trace: Trace, ...) -> Hypothesis:
        scenario_desc = trace.scen.get_scenario_all_desc()  # Uses trace.scen
        ...
```

#### Step 4: Update workflow initialization

```python
# Before (rdagent/components/workflow/rd_loop.py)
class RDLoop:
    def __init__(self, PROP_SETTING):
        scen = import_class(PROP_SETTING.scen)()
        self.hypothesis_gen = import_class(PROP_SETTING.hypothesis_gen)(scen)
        self.summarizer = import_class(PROP_SETTING.summarizer)(scen)
        self.trace = Trace(scen=scen)

# After
class RDLoop:
    def __init__(self, PROP_SETTING):
        scen = import_class(PROP_SETTING.scen)()
        self.hypothesis_gen = import_class(PROP_SETTING.hypothesis_gen)()  # No scen
        self.summarizer = import_class(PROP_SETTING.summarizer)()  # No scen
        self.trace = Trace(scen=scen)  # Single source of truth
```

#### Step 5: Handle Developer class (special case)

`Developer` doesn't receive `trace` in `develop()`. Options:

**Option A: Add trace parameter**
```python
class Developer(ABC):
    @abstractmethod
    def develop(self, exp: Experiment, trace: Trace) -> None:
        """Develop experiment using trace.scen for scenario info."""
```

**Option B: Store trace reference**
```python
class Developer(ABC):
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    @abstractmethod
    def develop(self, exp: Experiment) -> None:
        """Access scenario via self.trace.scen."""
```

**Option C: Access via experiment** (if experiment stores trace ref)
```python
class Experiment:
    def __init__(self, ..., trace: Trace | None = None):
        self._trace = trace

    @property
    def scen(self) -> Scenario | None:
        return self._trace.scen if self._trace else None
```

### Files Requiring Changes (50+ files)

**Core (signatures):**
- `rdagent/core/proposal.py`
- `rdagent/core/developer.py`
- `rdagent/core/interactor.py`
- `rdagent/core/evolving_framework.py`

**Components:**
- `rdagent/components/proposal/__init__.py`
- `rdagent/components/coder/CoSTEER/*.py`
- `rdagent/components/coder/factor_coder/*.py`
- `rdagent/components/coder/model_coder/*.py`
- `rdagent/components/workflow/rd_loop.py`

**Scenarios (highest impact):**
- `rdagent/scenarios/data_science/dev/feedback.py`
- `rdagent/scenarios/data_science/proposal/exp_gen/*.py`
- `rdagent/scenarios/qlib/developer/*.py`
- `rdagent/scenarios/qlib/proposal/*.py`
- `rdagent/scenarios/kaggle/developer/*.py`
- `rdagent/scenarios/kaggle/proposal/*.py`

**App loops:**
- `rdagent/app/qlib_rd_loop/*.py`
- `rdagent/app/kaggle/loop.py`
- `rdagent/app/data_science/loop.py`

### Risk Assessment
- **Risk Level**: High
- **Breaking Changes**: Constructor signatures, all subclasses need updates
- **Testing**: Full test suite, all scenarios
- **Mitigation**: Use deprecation warnings first, migrate incrementally

---

## Implementation Timeline

| Phase | Effort | Risk | Dependencies |
|-------|--------|------|--------------|
| Phase 1 | 1-2 hours | Low | None |
| Phase 2 | 2-3 days | Medium | Phase 1 |
| Phase 3 | 1-2 weeks | High | Phase 2 (optional) |

---

## Success Criteria

### Phase 1 ✅ COMPLETED (2025-12-01)
- [x] `sub_results` removed from base `Experiment` (rdagent/core/experiment.py:418-420)
- [x] Added to Kaggle experiments (rdagent/scenarios/kaggle/experiment/kaggle_experiment.py)
- [x] Fixed docstring in proposal.py:165-172
- [x] Verified all usages are Kaggle-specific

### Phase 2
- [ ] `version` field removed from `AbsTask`
- [ ] Both scenarios use strategy pattern
- [ ] No execution logic in workspace classes
- [ ] Clean separation of concerns

### Phase 3
- [ ] No `self.scen` in Gen/Feedback classes
- [ ] Single source of truth via `trace.scen`
- [ ] All workflows updated
- [ ] Deprecation warnings removed after full migration

---

## Findings from Phase 2 Investigation (2025-12-01)

### Current Architecture Discovery

The version-based branching is more nuanced than initially assessed:

**Two Levels of Workspaces:**
1. **Scenario-level workspaces** (already scenario-specific):
   - `QlibFBWorkspace` in `rdagent/scenarios/qlib/experiment/workspace.py`
   - `KGFBWorkspace` in `rdagent/scenarios/kaggle/experiment/workspace.py`
   - These handle experiment-level execution (backtest, submission scoring)

2. **Component-level workspaces** (have version branching):
   - `FactorFBWorkspace` in `rdagent/components/coder/factor_coder/factor.py`
   - `ModelFBWorkspace` in `rdagent/components/coder/model_coder/model.py`
   - These handle code testing/validation during development

**Version Branching Locations:**

| File | Lines | Purpose |
|------|-------|---------|
| `factor_coder/factor.py` | 130, 140, 153, 155 | Data path, execution template |
| `model_coder/model.py` | 113, 124, 135 | Environment, code template |
| `factor_coder/eva_utils.py` | 399, 402, 416 | Evaluator selection |
| `CoSTEER/knowledge_management.py` | 64, 74 | Knowledge base version (different concept) |

**Key Insight:** The `evolving_version` in CoSTEER is a different concept from `Task.version` - it controls knowledge base format, not execution behavior.

### Revised Phase 2 Approach

Given the complexity, Phase 2 should be broken into increments:

**Phase 2a: Documentation & Deprecation**
- Document version semantics clearly
- Add deprecation warnings to `Task.version`
- No behavioral changes

**Phase 2b: Extract Scenario-Specific Logic**
- Create `QlibFactorFBWorkspace` and `KGFactorFBWorkspace` subclasses
- Create `QlibModelFBWorkspace` and `KGModelFBWorkspace` subclasses
- Move version-specific execute() logic to subclasses

**Phase 2c: Remove Version Field**
- Update all task creation sites to use correct workspace class
- Remove version parameter from Task
- Remove version checks from base workspace classes

### Alternative: Configuration Objects

Instead of subclasses, inject configuration:
```python
@dataclass
class WorkspaceConfig:
    env_factory: Callable[[], Env]
    data_path: Path
    execution_template: str

class FactorFBWorkspace:
    def __init__(self, config: WorkspaceConfig, ...):
        self.config = config
```

This avoids class explosion but requires config objects to be passed through the call chain.
