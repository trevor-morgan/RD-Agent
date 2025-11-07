# RD-Agent: Research & Development Agent - Comprehensive Codebase Overview

**Project Size:** ~3,957 lines of Python code across multiple modules  
**Latest Version:** 0.8.0  
**Primary Focus:** Autonomous R&D for Data-Driven Scenarios (Finance, Medical, General ML)

---

## 1. OVERALL ARCHITECTURE & PHILOSOPHY

### Core Vision
RD-Agent automates the industrial R&D process by breaking it into two key components:
- **R (Research/Proposal):** Generate new ideas and hypotheses using LLMs
- **D (Development):** Implement and evolve ideas, learning from feedback

### Architectural Paradigm: Evolution Loop
```
Hypothesis Generation → Experiment Planning → Code Implementation → Execution → Feedback → Learning
         (R)                     ↓                    (D)              ↓         ↑
         └─────────────────────────────────────────────────────────────────────┘
                            Iterative Evolution
```

### Design Principles
1. **Modularity:** Components are loosely coupled and composable
2. **Extensibility:** Scenarios and strategies can be inherited and customized
3. **Traceability:** Every step is logged for inspection and debugging
4. **Parallel Processing:** Multi-process execution for scaling
5. **Knowledge Reuse:** RAG (Retrieval Augmented Generation) for learning across iterations

---

## 2. HIGH-LEVEL DIRECTORY STRUCTURE

```
rdagent/
├── app/                          # CLI and application entry points
│   ├── cli.py                   # Main CLI interface (Typer-based)
│   ├── data_science/            # Data Science scenario apps
│   ├── qlib_rd_loop/            # Quantitative Finance apps (factors, models, quant)
│   ├── kaggle/                  # Kaggle competition apps
│   ├── general_model/           # General model extraction from papers
│   ├── finetune/                # Fine-tuning applications
│   ├── benchmark/               # Benchmarking tools
│   └── utils/                   # Health check, info collection
│
├── components/                   # Core algorithm/logic modules
│   ├── agent/                   # LLM agent integration (Pydantic-AI, MCP)
│   ├── coder/                   # Code generation (CoSTEER, Factor, Model, Data Science)
│   ├── proposal/                # Hypothesis/proposal generation
│   ├── runner/                  # Code execution infrastructure
│   ├── workflow/                # Orchestration (RDLoop)
│   ├── benchmark/               # Evaluation metrics
│   ├── document_reader/         # PDF/paper reading
│   ├── interactor/              # User interaction
│   ├── knowledge_management/    # Graph-based knowledge
│   └── loader/                  # Data loading utilities
│
├── core/                         # Abstract base classes and interfaces
│   ├── scenario.py              # Scenario base class
│   ├── experiment.py            # Task, Workspace, RunningInfo
│   ├── proposal.py              # Hypothesis, Feedback classes
│   ├── evolving_framework.py    # EvolvingStrategy, RAGStrategy
│   ├── evolving_agent.py        # EvoAgent, RAGEvoAgent
│   ├── developer.py             # Developer base class
│   ├── evaluation.py            # Feedback, Evaluator abstractions
│   ├── conf.py                  # Global settings
│   └── utils.py                 # Helper functions
│
├── scenarios/                    # Domain-specific implementations
│   ├── qlib/                    # Quantitative trading with Qlib
│   │   ├── proposal/            # Factor/Model hypothesis generation
│   │   ├── developer/           # Factor/Model coding
│   │   ├── experiment/          # Factor/Model experiment execution
│   │   └── prompts.yaml         # LLM prompts
│   ├── data_science/            # Kaggle/MLE competitions
│   │   ├── scen/                # Scenario definitions (DataScienceScen, KaggleScen)
│   │   ├── proposal/            # Hypothesis generation
│   │   ├── dev/                 # Coders and runners
│   │   ├── experiment/          # Workspace management
│   │   └── loop.py              # DataScienceRDLoop
│   ├── kaggle/                  # Kaggle-specific implementations
│   ├── general_model/           # General model extraction
│   ├── shared/                  # Shared utilities across scenarios
│   └── README files for each scenario
│
├── oai/                          # LLM Integration Layer
│   ├── backend/
│   │   ├── litellm.py           # LiteLLM backend (primary)
│   │   ├── pydantic_ai.py       # Pydantic-AI integration
│   │   ├── base.py              # Abstract backend
│   │   └── deprec.py            # Deprecated OpenAI direct
│   ├── llm_utils.py             # LLM utilities
│   ├── llm_conf.py              # LLM configuration
│   └── utils/
│       └── embedding.py         # Embedding models
│
├── log/                          # Logging and UI
│   ├── logger.py                # Main logger (rdagent_logger)
│   ├── base.py                  # Logging base classes
│   ├── storage.py               # Log persistence
│   ├── timer.py                 # Timing/performance tracking
│   ├── mle_summary.py           # MLE benchmark summary
│   ├── ui/                      # Streamlit UI applications
│   ├── server/                  # Web server for real-time logs
│   └── utils/                   # Logging utilities
│
└── utils/                        # General-purpose utilities
    ├── workflow/                # LoopBase, LoopMeta (session control)
    ├── agent/                   # Agent utilities (templates, workflows)
    ├── repo/                    # Repository utilities (git, diff, patch)
    └── env.py                   # Environment management
```

---

## 3. CORE COMPONENTS & THEIR RESPONSIBILITIES

### 3.1 EXPERIMENT FRAMEWORK (core/experiment.py)

**Key Classes:**
- `Task`: Represents a unit of work with name, description, user instructions
- `Workspace`: Abstract container holding task implementation and feedback
- `RunningInfo`: Result and timing information from execution
- `UserInstructions`: List of user directives (top priority)

**Relationships:**
```
Task (what to do)
  ↓
Workspace (how to implement)
  ↓
RunningInfo (what was achieved + timing)
```

**Implementations by Scenario:**
- `FBWorkspace`: File-based workspace (general)
- `DSExperiment`: Data science specific
- `FactorExperiment`: Quantitative finance factors

---

### 3.2 EVOLVING FRAMEWORK (core/evolving_framework.py)

**Key Classes:**
- `EvolvableSubjects`: The target object being evolved (e.g., code, factors, models)
- `EvolvingStrategy`: Abstract strategy for evolving subjects
- `EvoStep`: Represents evolution step with subject, queried knowledge, and feedback
- `EvolvingKnowledgeBase`: Knowledge base for RAG queries
- `RAGStrategy`: Retrieval-augmented generation strategy

**Generic Type Parameters:**
- `ASpecificEvolvableSubjects`: Allows type-safe evolution of different object types

**Design Pattern:**
```
EvoStep = (EvolvableSubjects, QueriedKnowledge, Feedback)
    ↓
EvolvingStrategy.evolve(evo, trace, queried_knowledge) → new EvolvableSubjects
    ↓
RAGStrategy.query(evo, trace) → QueriedKnowledge
```

---

### 3.3 PROPOSAL FRAMEWORK (core/proposal.py)

**Key Classes:**
- `Hypothesis`: Proposed idea with reason and justification
- `ExperimentFeedback`: Feedback from experiment execution
  - `decision`: Boolean (success/failure)
  - `reason`: Explanation for feedback
  - `exception`: Captured runtime errors
- `HypothesisFeedback`: Extended feedback from hypothesis testing
- `Trace`: Historical record of experiments and feedback
- `HypothesisGen`: Abstract class for generating hypotheses
- `Hypothesis2Experiment`: Converts hypotheses to executable experiments
- `Experiment2Feedback`: Generates feedback from experiment results

**Data Flow:**
```
Hypothesis (idea)
    ↓
Hypothesis2Experiment
    ↓
Experiment (with tasks)
    ↓
[Execution]
    ↓
Experiment2Feedback
    ↓
Feedback (decision + reason)
    ↓
Trace.hist (stored for context)
```

---

### 3.4 AGENT FRAMEWORK (components/agent/)

**Abstraction Layers:**

1. **BaseAgent Interface:**
   - Abstract `query(query: str) → str` method
   - Forces implementation of system prompt and toolsets

2. **PAIAgent (Pydantic-AI Implementation):**
   - Uses Pydantic-AI for structured LLM interactions
   - Optional Prefect caching support
   - Supports MCP (Model Context Protocol) servers for tools
   - Async-based with sync wrapper

3. **LLM Backend Selection (oai/backend/):**
   - **LiteLLMAPIBackend** (Primary): Supports 100+ LLM providers
   - **OpenAIChatModel** (Deprecated): Direct OpenAI only
   - Configuration via environment variables or `.env` files

**Supported Providers:**
- OpenAI (GPT-4, GPT-4o, etc.)
- Azure OpenAI
- DeepSeek
- Anthropic (Claude)
- Local/Ollama
- Custom LiteLLM proxies

---

### 3.5 CODER FRAMEWORK (components/coder/)

**Architecture: CoSTEER (Collaborative Evolving Strategy)**

Key Components:
- `EvolvingItem`: Container for sub-tasks and their implementation status
- `MultiProcessEvolvingStrategy`: Base class for multi-process code evolution
  - Implements task scheduling
  - Parallel code generation
  - Knowledge reuse from successful tasks

- `CoSTEERSingleFeedback`: Feedback for individual task
- `CoSTEERMultiFeedback`: Aggregated feedback for multiple tasks

**Implementations:**
1. **Factor Coder** (`factor_coder/`)
   - Evolves quantitative factors
   - Uses Qlib framework

2. **Model Coder** (`model_coder/`)
   - Evolves ML/DL model structures
   - Supports benchmark ground truth code

3. **Data Science Coder** (`data_science/`)
   - Multiple specialized coders:
     - `DataLoaderCoSTEER`: Raw data loading
     - `FeatureCoSTEER`: Feature engineering
     - `ModelCoSTEER`: Model implementation
     - `EnsembleCoSTEER`: Ensemble methods
     - `PipelineCoSTEER`: End-to-end pipelines
     - `WorkflowCoSTEER`: Workflow automation

**Task Scheduling:**
- Feedback-based scheduling
- Improvement mode (retry failed tasks)
- Parallel execution with task dependencies
- Knowledge reuse from previous successes

---

### 3.6 RUNNER FRAMEWORK (components/runner/)

**Responsibilities:**
- Execute generated code in isolated environments
- Capture output, errors, and metrics
- Handle timeout and resource constraints
- Provide execution feedback

**Scenario-Specific Runners:**
- `DSCoSTEERRunner`: Data science experiment runner
- `FactorRunner`: Factor execution in Qlib
- `ModelRunner`: Model training/evaluation

---

### 3.7 WORKFLOW ORCHESTRATION (utils/workflow/)

**LoopBase & LoopMeta:**
- Metaclass-based step composition
- Automatic step discovery from base classes
- Session persistence and resumption
- Parallel execution with semaphores
- Error handling (skip/withdraw options)

**RDLoop (components/workflow/rd_loop.py):**
- Base workflow for R&D iterations
- Steps: `direct_exp_gen` → `coding` → `running` → `feedback`
- Data flow via dictionaries through steps
- Exception handling and logging

**Scenario-Specific Loops:**
- `DataScienceRDLoop`: Kaggle/MLE competitions
- `QuantRDLoop`: Quantitative trading (factors + models)
- Customizable via `PROP_SETTING` configuration

---

### 3.8 SCENARIO FRAMEWORK (rdagent/scenarios/)

**Base Class: Scenario**
```python
class Scenario(ABC):
    @property
    def background(self) -> str: ...
    def get_source_data_desc(self, task) -> str: ...
    @property
    def rich_style_description(self) -> str: ...
    def get_scenario_all_desc(self, task, ...) -> str: ...
    def get_runtime_environment(self) -> str: ...
```

**Implementations:**

1. **QuantTrading (scenarios/qlib/)**
   - Factor and model co-optimization
   - Integration with Qlib framework
   - Market data handling
   - Backtesting infrastructure

2. **DataScience (scenarios/data_science/)**
   - Competition-agnostic design
   - Support for Kaggle and MLE benchmarks
   - Data exploration and profiling
   - Multi-task learning (data loading, feature, model, ensemble, pipeline)

3. **Kaggle (scenarios/kaggle/)**
   - Web scraping for competition info
   - Leaderboard tracking
   - Submission management

4. **GeneralModel (scenarios/general_model/)**
   - Paper/report reading
   - Model extraction from documentation
   - No dataset dependency

---

## 4. DATA FLOW & COMPONENT INTERACTIONS

### 4.1 Quantitative Trading Loop (fin_quant)

```
QuantRDLoop.__init__()
├── HypothesisGen (quant_hypothesis_gen) - proposes factor or model
├── Hypothesis2Experiment (factor/model) - converts to tasks
├── Coder (factor/model) - generates code
├── Runner (factor/model) - executes and evaluates
├── Summarizer (factor/model) - provides feedback
└── QuantTrace - tracks all iterations

Iteration Flow:
  1. direct_exp_gen: await HypothesisGen → Hypothesis2Experiment
  2. coding: Coder.develop(Experiment)
  3. running: Runner.develop(Experiment)
  4. feedback: Summarizer.generate_feedback(Experiment)
  5. record: Store in QuantTrace
```

**Key Abstractions:**
- Action type: "factor" or "model"
- Alternating optimization for factor-model co-evolution
- Historical trace for learning

### 4.2 Data Science Loop (data_science)

```
DataScienceRDLoop.__init__()
├── DSProposalV2ExpGen (idea_pool + proposal) - generates ideas
├── DataLoaderCoSTEER - handles data loading
├── FeatureCoSTEER - engineers features
├── ModelCoSTEER - trains models
├── EnsembleCoSTEER - creates ensembles
├── PipelineCoSTEER - orchestrates full pipeline
└── DSExperiment2Feedback - provides feedback

Multi-Task Evolution:
  Tasks = [DataLoaderTask, FeatureTask, ModelTask, ...]
  For each task:
    1. Schedule based on previous feedback
    2. Generate implementation with CoSTEER
    3. Execute in workspace
    4. Evaluate and provide feedback
    5. Store success in knowledge base
```

**Unique Features:**
- MCTS (Monte Carlo Tree Search) based scheduling
- Multi-stage learning (different exp_gen for different stages)
- Knowledge base for successful task implementations
- Workspace backup and recovery

### 4.3 General Model Loop (general_model)

```
Input: Paper/Report URL
  ↓
Document Reader (extract text/sections)
  ↓
Agent: Extract model structure + formulas
  ↓
Code Generation: Implement extracted models
  ↓
Test on available data (if any)
  ↓
Output: Executable code
```

---

## 5. MAIN ENTRY POINTS

### 5.1 CLI Commands (rdagent/app/cli.py)

```
rdagent fin_factor        # Iteratively evolve quantitative factors
rdagent fin_model         # Iteratively evolve models
rdagent fin_quant         # Joint factor-model evolution
rdagent fin_factor_report # Extract factors from financial reports
rdagent general_model <URL>  # Extract models from papers
rdagent data_science --competition <name>  # Kaggle/MLE competitions
rdagent ui --port 19899 --log-dir <path>  # Streamlit dashboard
rdagent health_check      # Verify Docker, ports, LLM
rdagent collect_info      # Environment information
```

### 5.2 Programmatic Entry Points

**Direct Loop Creation:**
```python
from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantRDLoop

loop = QuantRDLoop(QUANT_PROP_SETTING)
asyncio.run(loop.run(step_n=10))  # Run 10 steps
```

**Configuration-based:**
```python
from rdagent.components.workflow.conf import BasePropSetting
PROP_SETTING = BasePropSetting(
    scen="rdagent.scenarios.qlib.proposal.QuantScen",
    hypothesis_gen="rdagent.scenarios.qlib.proposal.QuatHypothesisGen",
    ...
)
```

---

## 6. DESIGN PATTERNS

### 6.1 Strategy Pattern
- `EvolvingStrategy`: Define how to evolve subjects
- `RAGStrategy`: Define how to retrieve and generate knowledge
- `Evaluator`: Define how to evaluate subjects
- Implementations swappable at configuration time

### 6.2 Metaclass Pattern
- `LoopMeta`: Automatically discovers step methods in class hierarchy
- Enables flexible step composition in subclasses
- `steps` attribute built dynamically

### 6.3 Generic/Type Parameterization
- `EvolvableSubjects[T]`: Polymorphic evolution support
- `EvoStep[T]`: Type-safe evolution traces
- `Developer[ASpecificExp]`: Task-specific implementations

### 6.4 Factory Pattern
- `import_class()`: Dynamic class instantiation from strings
- Configuration-driven component creation
- Enables plugin architecture

### 6.5 Template Method Pattern
- `RDLoop`: Template for R&D iterations
- Subclasses override `_propose()`, `_exp_gen()`, etc.
- `LoopBase`: Template for session management

### 6.6 Observer/Logging Pattern
- `rdagent_logger`: Global logging with context tags
- Automatic logging of objects with `logger.log_object(obj, tag="...")`
- Session-based trace collection

### 6.7 Dependency Injection
- Configuration objects pass dependencies
- Loose coupling between components
- Testability and flexibility

---

## 7. DIFFERENT SCENARIOS

### 7.1 Quantitative Trading (Qlib)

**Goal:** Automate factor engineering and model development for stock trading

**Components:**
- `QuantHypothesisGen`: Proposes factor or model improvements
- `FactorCoder`: Generates factor code
- `ModelCoder`: Generates model code
- Integration with Qlib (market data, backtesting)

**Unique Aspects:**
- Action dispatch: "factor" vs "model"
- Alternating optimization
- Financial domain knowledge in prompts
- Backtesting for evaluation

**Commands:**
```bash
rdagent fin_factor       # Factor loop only
rdagent fin_model        # Model loop only
rdagent fin_quant        # Joint optimization
rdagent fin_factor_report --report-folder <path>  # From reports
```

### 7.2 Data Science / ML Engineering (Kaggle/MLE)

**Goal:** Automate feature engineering, model selection, and hyperparameter tuning

**Unique Multi-Task Evolution:**
1. Data Loading
2. Feature Engineering
3. Model Selection/Implementation
4. Ensemble
5. Pipeline Orchestration

**Components:**
- `DSProposalV2ExpGen`: Generates experiment ideas
- Multiple scenario-specific coders (see 3.5)
- `DSCoSTEERRunner`: Executes in Docker
- `DSExperiment2Feedback`: Metric-based feedback

**Features:**
- Competition description parsing
- Data profiling and exploration
- MCTS-based task scheduling
- Knowledge base for successful implementations
- Timeout extension based on feedback

**Scenarios Supported:**
- Kaggle competitions (public)
- MLE-bench (75 competitions)
- Custom local datasets

**Commands:**
```bash
rdagent data_science --competition tabular-playground-series-dec-2021
```

### 7.3 General Model Extraction

**Goal:** Extract model structures and formulas from papers/reports

**Workflow:**
1. Document Reader: Extract relevant sections
2. LLM Agent: Parse formulas and architecture
3. Code Generation: Implement extracted models
4. Validation: Test on available data

**Special Handling:**
- PDF parsing with Azure Document Intelligence (optional)
- LaTeX formula extraction
- Code snippet detection
- Minimal dataset requirement

**Command:**
```bash
rdagent general_model "https://arxiv.org/pdf/2210.09789"
```

---

## 8. KEY DESIGN PATTERNS & ABSTRACTIONS

### 8.1 Separation of Concerns

```
Proposal Generation (Research)
    ↓
    └─→ LLM-based idea generation
        └─→ Statistical/rule-based validation

Code Implementation (Development)
    ↓
    └─→ LLM-based code generation
        └─→ Syntax checking
        └─→ Execution validation

Evaluation & Feedback
    ↓
    └─→ Execution-based metrics
        └─→ Statistical analysis
        └─→ Domain-specific validation
```

### 8.2 Layered Architecture

```
Application Layer (app/)
    ├─→ CLI
    └─→ Loops (RDLoop, DataScienceRDLoop, etc.)

Orchestration Layer (components/workflow/)
    └─→ LoopBase (session management, step execution)

Algorithm Layer (components/, core/)
    ├─→ Proposal generation (HypothesisGen)
    ├─→ Code generation (Coders)
    ├─→ Execution (Runners)
    └─→ Feedback (Evaluators)

LLM Integration Layer (oai/)
    ├─→ LiteLLM backend
    ├─→ Pydantic-AI integration
    └─→ Embedding models

Infrastructure Layer (log/, utils/)
    ├─→ Logging & persistence
    ├─→ Session management
    ├─→ Utilities
    └─→ Environment setup
```

### 8.3 Extensibility Points

**Easy to Extend:**
1. Add new scenario: Inherit from `Scenario`
2. Add new coder: Inherit from `MultiProcessEvolvingStrategy`
3. Add new proposer: Inherit from `HypothesisGen`
4. Add new evaluator: Implement `Evaluator` interface
5. Add new loop: Inherit from `LoopBase`
6. Add new LLM provider: Supported via LiteLLM

---

## 9. KNOWLEDGE MANAGEMENT & LEARNING

### 9.1 RAG Strategy Pattern

```
EvolvingKnowledgeBase
    ├─→ query(evo, trace) → QueriedKnowledge
    └─→ generate_knowledge(trace) → Knowledge

Knowledge Lifecycle:
    1. Initialize from examples/templates
    2. Query for context-relevant knowledge
    3. Generate new knowledge from successes
    4. Dump to persistent storage
    5. Load and merge across processes
```

### 9.2 Specific Implementations

**DataScience (scenarios/data_science/)**
- `DSKnowledgeBase`: Stores successful task implementations
- `success_task_to_knowledge_dict`: Maps task description → implementation

**Qlib (scenarios/qlib/)**
- Factor/Model templates from literature
- Success tracking for future reuse

### 9.3 Parallel Knowledge Sharing

```
Process 1          Process 2          Process 3
  ↓                  ↓                  ↓
Knowledge Base   Knowledge Base   Knowledge Base
  ↓                  ↓                  ↓
Dump (FileLock) → Shared Storage ← Load (FileLock)
```

---

## 10. EXECUTION & SESSION MANAGEMENT

### 10.1 Workspace Concept

```
Workspace
├─→ target_task: Task
├─→ feedback: Feedback
└─→ running_info: RunningInfo
    ├─→ result: object (varies by scenario)
    └─→ running_time: float
```

**Implementation:**
- `FBWorkspace`: File-based (disk storage)
- Snapshot capability via copy
- Immutable during evaluation

### 10.2 Session Persistence (LoopBase)

```
Session Creation:
    → loop.save() → pickle serialization → session dir

Session Resumption:
    → path → loop.load() → restore state → continue from step

Session Tracking:
    → loop_trace[loop_idx] → list[LoopTrace]
        └─→ LoopTrace(start, end, step_idx)
```

### 10.3 Parallel Loop Management

```
RD_AGENT_SETTINGS.step_semaphore
    ├─→ int: Global limit (all steps)
    └─→ dict: Per-step limits {"coding": 3, "running": 2}

Concurrent Execution:
    while unfinished_loop_count < max_parallel:
        → create new loop
        → yield results as they complete
        → respawn when capacity available
```

---

## 11. LOGGING & MONITORING

### 11.1 Logging Architecture

```
rdagent_logger (global)
    ├─→ Context tags: logger.tag("evo_loop_0")
    ├─→ Object logging: logger.log_object(obj, tag="...")
    ├─→ Structured storage: log/storage.py
    └─→ Real-time streaming: log/server/

Session Structure:
    log/
    └── {timestamp}/
        ├── __session__/
        │   ├── 0/  # Loop 0
        │   │   ├── propose/
        │   │   ├── coding/
        │   │   ├── running/
        │   │   └── feedback/
        │   ├── 1/  # Loop 1
        │   ...
        └── workspace/  # Checkpoints
```

### 11.2 UI & Visualization

**Streamlit Dashboard (log/ui/):**
- Trace visualization
- Session browser
- Results comparison
- Real-time monitoring (server_ui)

**Web Server (log/server/):**
- Real-time log streaming
- WebSocket support
- Dashboard integration

---

## 12. CONFIGURATION SYSTEM

### 12.1 Global Settings (core/conf.py)

```python
RDAgentSettings:
    workspace_path: Path  # Default: git_ignore_folder/RD-Agent_workspace
    multi_proc_n: int     # Number of processes
    cache_with_pickle: bool
    step_semaphore: int | dict  # Parallelization control
    stdout_context_len: int  # Log truncation
    enable_mlflow: bool
    max_parallel: int()  # Derived from semaphore
    is_force_subproc(): bool
```

### 12.2 Scenario-Specific Settings

```python
BasePropSetting:
    scen: str  # Scenario class path
    hypothesis_gen: str
    hypothesis2experiment: str
    coder: str
    runner: str
    summarizer: str
    # ... more fields for advanced scenarios
```

### 12.3 LLM Configuration (oai/llm_conf.py)

```python
LLMSettings:
    chat_model: str  # e.g., "gpt-4o"
    embedding_model: str
    chat_max_tokens: int
    embedding_max_tokens: int
    # Provider-specific keys via environment variables
```

---

## 13. IMPORTANT PATTERNS & CONVENTIONS

### 13.1 Step Naming Convention

Steps in RDLoop follow implicit dependencies:
```
direct_exp_gen → coding → running → feedback → record
```

Previous outputs accessible as:
```python
prev_out["direct_exp_gen"]  # From direct_exp_gen step
prev_out["coding"]          # From coding step
```

### 13.2 Exception Handling

```python
class RDLoop:
    skip_loop_error = (CoderError, RunnerError)       # Skip this loop
    withdraw_loop_error = (PolicyError,)  # Stop entire loop
```

### 13.3 Type Safety

```python
ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)
Developer[ASpecificExp]  # Generic over experiment type
```

### 13.4 Async/Await Pattern

```python
async def direct_exp_gen(prev_out):
    while True:
        if can_create_new_loop():
            return {...}
        await asyncio.sleep(1)  # Check capacity
```

---

## 14. DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────┐
│                   CLI (cli.py)                  │
└────────────────────┬────────────────────────────┘
                     ↓
        ┌─────────────────────────────┐
        │  RDLoop / DataScienceRDLoop │
        └────────┬────────────────────┘
                 ↓
    ┌────────────────────────────────────────┐
    │                                        │
 ┌──→ HypothesisGen        Hypothesis2Exp ←──┐
 │  ↓ (proposal)           (exp planning)     │
 │  │                          ↓              │
 │  │                      ┌──────────────┐   │
 │  └─→ Coder ← ─ ─ ─ ─ ─→│ Experiment   │   │
 │     (develop)           │ (Task list)  │   │
 │     ↓                   └──────────────┘   │
 │  Runner ← ─ ─ ─ ─ ─ ┘      ↑              │
 │  (execute)                 │              │
 │     ↓                       │              │
 │  Evaluator2Feedback ────────┘              │
 │  (feedback)                                │
 └────────────────────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │  Scenario (domain knowledge)│
    │  Logger (tracing)           │
    │  Storage (persistence)      │
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │    LLM Backend (LiteLLM)    │
    │    Workspace Management     │
    │    RAG Knowledge Base       │
    └─────────────────────────────┘
```

---

## 15. QUICK REFERENCE: KEY CLASSES

| Class | Module | Purpose |
|-------|--------|---------|
| `Scenario` | core/scenario.py | Domain context and environment |
| `Task` | core/experiment.py | Work unit definition |
| `Workspace` | core/experiment.py | Implementation container |
| `Hypothesis` | core/proposal.py | Proposed idea |
| `Feedback` | core/evaluation.py | Evaluation result |
| `HypothesisGen` | core/proposal.py | Idea generator |
| `Developer` | core/developer.py | Implementation generator |
| `Evaluator` | core/evaluation.py | Performance evaluator |
| `EvolvingStrategy` | core/evolving_framework.py | Evolution algorithm |
| `RAGEvoAgent` | core/evolving_agent.py | Main evolution loop |
| `LoopBase` | utils/workflow/loop.py | Session management |
| `RDLoop` | components/workflow/rd_loop.py | R&D iteration template |
| `MultiProcessEvolvingStrategy` | components/coder/CoSTEER/evolving_strategy.py | Multi-task evolution |
| `PAIAgent` | components/agent/base.py | LLM agent wrapper |
| `LiteLLMAPIBackend` | oai/backend/litellm.py | LLM integration |

---

## 16. METRICS & PERFORMANCE TRACKING

### Timer System (log/timer.py)

```python
RDAgentTimer:
    record_time(tag: str, time: float)
    
Usage:
    with RD_Agent_TIMER_wrapper("my_step"):
        # Code to measure
        pass
```

### MLE Summary (log/mle_summary.py)

```python
grade_summary(log_path):
    # Parses MLE-bench results
    # Grades agent performance
    # Generates summary report
```

---

## 17. ADVANCED FEATURES

### 17.1 Distributed Knowledge Base

- Shared across parallel workers
- File-lock based synchronization
- Incremental knowledge generation

### 17.2 Timeout Extension

Data Science scenario:
```python
if feedback_shows_potential_improvement:
    timeout_increase_count += 1
    extend_timeout()
```

### 17.3 Document Reading

- PDF/arXiv support
- Azure Document Intelligence integration
- LaTeX formula extraction

### 17.4 MCTS Scheduling

Data Science proposal uses Monte Carlo Tree Search for:
- Intelligent task ordering
- Parallel execution planning
- Resource allocation

---

## 18. COMMON WORKFLOWS

### Workflow 1: Run Data Science Agent

```python
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.scenarios.data_science.loop import DataScienceRDLoop

DS_RD_SETTING.competition = "tabular-playground-series-dec-2021"
loop = DataScienceRDLoop(DS_RD_SETTING)
asyncio.run(loop.run(step_n=10, loop_n=5))
```

### Workflow 2: Resume from Checkpoint

```python
loop = DataScienceRDLoop.load(
    path="log/__session__/1/0_propose",
    checkout=True,
    replace_timer=True
)
asyncio.run(loop.run(step_n=10))
```

### Workflow 3: Custom Scenario

```python
from rdagent.core.scenario import Scenario
from rdagent.components.workflow.rd_loop import RDLoop

class MyScenario(Scenario):
    @property
    def background(self) -> str:
        return "My domain"
    # ... implement other methods

class MyRDLoop(RDLoop):
    def _propose(self):
        # Custom proposal
        pass

loop = MyRDLoop(PROP_SETTING)
asyncio.run(loop.run())
```

---

## 19. TESTING & VALIDATION

### Health Check (app/utils/health_check.py)

```bash
rdagent health_check                      # All checks
rdagent health_check --no-check-env       # Skip environment
rdagent health_check --no-check-docker    # Skip Docker
```

Checks:
- Docker availability
- Port availability
- LLM configuration
- Environment setup

### CI/CD (rdagent/app/CI/)

- Automated testing on commits
- Code quality (Ruff, MyPy)
- Dependency management

---

## 20. EXTENSIBILITY & CUSTOMIZATION

### To Add a New Scenario:

1. Create `scenarios/your_scenario/` directory
2. Inherit from `Scenario`
3. Implement required methods
4. Create scenario-specific coders/runners/evaluators
5. Update configuration in `app/`
6. Add CLI command in `app/cli.py`

### To Add a New LLM Provider:

```python
# LiteLLM handles automatically via:
LiteLLMAPIBackend(model="provider/model-name")
```

### To Add a New Evolution Strategy:

1. Inherit from `EvolvingStrategy`
2. Implement `evolve()` method
3. Reference in `PROP_SETTING`

### To Add a New Component:

1. Define in `components/`
2. Inherit from appropriate base
3. Implement required methods
4. Configure in settings

---

## 21. SUMMARY OF KEY INSIGHTS

1. **Modular Design**: Clear separation of proposal, development, and evaluation
2. **LLM-Centric**: Heavy use of LLMs for creative ideation and code generation
3. **Feedback-Driven**: Learns from execution feedback to improve future iterations
4. **Scenario-Agnostic**: Core framework supports multiple domains (finance, ML, etc.)
5. **Scalable**: Multi-process execution, session persistence, knowledge reuse
6. **Observable**: Comprehensive logging, session tracking, UI dashboards
7. **Configurable**: Dependency injection, settings-based composition
8. **Extensible**: Clear abstractions for adding new scenarios, strategies, evaluators

---

**Document Generated:** November 7, 2025  
**Codebase Version:** 0.8.0  
**Total Python Lines:** ~3,957

