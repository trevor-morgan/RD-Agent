"""RD-Agent adapters and experimental research templates.

This module contains:

Production-Ready
----------------
- **RDAgentAdapter**: Direct library integration with RD-Agent quant workflows.
  Use this to call `rdagent.app.qlib_rd_loop.quant.main()` without subprocess.

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
from rdagent_lab.research import RDAgentAdapter
adapter = RDAgentAdapter()
result = adapter.run_quant()

# Experimental (explicit import required)
from rdagent_lab.research.symplectic_templates import SymplecticTransformer
from rdagent_lab.research.scenario_generator import DiffusionModel
```
"""

from rdagent_lab.research.rdagent_adapter import RDAgentAdapter

__all__ = ["RDAgentAdapter"]
