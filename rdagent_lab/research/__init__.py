"""RD-Agent adapters and experimental research templates.

This module contains:

Production-Ready
----------------
- **RDAgentAdapter**: Direct library integration with RD-Agent quant workflows.
  Use this to call `rdagent.app.qlib_rd_loop.quant.main()` without subprocess.

- **MonteCarloSimulator**: Monte Carlo scenario simulation with VaR/CVaR metrics.
  Risk assessment and stress testing via simulation.

Experimental (use with caution)
-------------------------------
- **symplectic_templates**: Symplectic Transformer and rough volatility factors.
  Novel architectures for factor modeling, not yet validated at scale.

- **scenario_generator**: DDPM-based synthetic time series generation.
  Useful for data augmentation and stress testing, but computationally expensive.

Example
-------
```python
# Production usage
from rdagent_lab.research import RDAgentAdapter, MonteCarloSimulator
adapter = RDAgentAdapter()
result = adapter.run_quant()

# Monte Carlo simulation
simulator = MonteCarloSimulator(model)
result = simulator.simulate(initial_state, n_scenarios=1000, n_steps=20)
print(f"VaR(95%): {result.var_95:.4f}")

# Experimental (explicit import required)
from rdagent_lab.research.symplectic_templates import SymplecticTransformer
from rdagent_lab.research.scenario_generator import DiffusionModel
```
"""

from rdagent_lab.research.rdagent_adapter import RDAgentAdapter
from rdagent_lab.research.monte_carlo import (
    MonteCarloSimulator,
    ScenarioResult,
    StressScenario,
    STANDARD_STRESS_SCENARIOS,
)

__all__ = [
    "RDAgentAdapter",
    "MonteCarloSimulator",
    "ScenarioResult",
    "StressScenario",
    "STANDARD_STRESS_SCENARIOS",
]
