"""Integration tests for the pitfall_detector module.

These tests simulate the full workflow from code with pitfalls to
detection, diagnosis, and correction.
"""

import pandas as pd
from rdagent.components.coder.pitfall_detector import (
    PitfallLinter,
    auto_correct_factor_code,
    diagnose_factor_result,
    get_pitfall_prompt_context,
    has_critical_pitfalls,
)


class TestEndToEndWorkflow:
    """Test the complete detection → diagnosis → correction workflow."""

    def test_multiindex_series_pitfall_workflow(self):
        """Test full workflow for the MultiIndex Series to DataFrame pitfall.

        This simulates the actual bug that was discovered:
        1. Code uses pd.DataFrame(series, columns=[...])
        2. This creates an empty DataFrame
        3. We detect the pitfall, diagnose the cause, and suggest correction
        """
        # Step 1: The problematic code
        bad_code = """
import pandas as pd

# Load data with MultiIndex (datetime, instrument)
df = pd.read_hdf("daily_pv.h5", key="data")

# Calculate momentum
momentum = df["$close"] / df.groupby(level="instrument")["$close"].shift(20) - 1.0

# BUG: This creates an EMPTY DataFrame with MultiIndex Series!
result = pd.DataFrame(momentum, columns=["CloseMomentum_20d"])

# Save empty result
result.to_hdf("result.h5", key="data")
"""

        # Step 2: Pre-execution linting detects the issue
        assert has_critical_pitfalls(bad_code)

        linter = PitfallLinter()
        results = linter.lint_code(bad_code)

        pandas_001 = [r for r in results if r.pitfall.id == "PANDAS_001"]
        assert len(pandas_001) == 1
        assert pandas_001[0].line_number == 11

        # Step 3: Simulate execution producing empty DataFrame
        empty_result = pd.DataFrame(columns=["CloseMomentum_20d"])

        # Step 4: Runtime diagnostics identifies root cause
        diagnosis = diagnose_factor_result(empty_result, bad_code)
        assert diagnosis.pitfall_id == "PANDAS_001"
        assert "to_frame" in diagnosis.fix.lower()
        assert diagnosis.confidence >= 0.9

        # Step 5: Auto-corrector fixes the code
        corrected_code, applied = auto_correct_factor_code(bad_code)
        assert len(applied) >= 1
        assert "to_frame" in corrected_code
        assert 'pd.DataFrame(momentum, columns=["CloseMomentum_20d"])' not in corrected_code

        # Step 6: Corrected code passes linting
        assert not has_critical_pitfalls(corrected_code)

    def test_missing_dollar_prefix_workflow(self):
        """Test workflow for missing $ prefix in Qlib column access."""
        bad_code = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")

# BUG: Missing $ prefix - will raise KeyError
momentum = df["close"] / df["close"].shift(20) - 1.0

result = momentum.to_frame(name="Momentum")
"""

        # Pre-execution detection
        assert has_critical_pitfalls(bad_code)

        # Correction
        corrected_code, applied = auto_correct_factor_code(bad_code)
        assert "$close" in corrected_code
        assert '"close"' not in corrected_code or '"$close"' in corrected_code

        # Verify correction
        assert not has_critical_pitfalls(corrected_code)

    def test_prompt_context_generation(self):
        """Test that prompt context is correctly generated for LLM guidance."""
        bad_code = """result = pd.DataFrame(factor, columns=["Factor"])"""

        context = get_pitfall_prompt_context(bad_code)

        # Context should include pitfall information
        assert "PANDAS_001" in context
        assert "pitfall" in context.lower() or "Pitfall" in context

    def test_combined_issues(self):
        """Test detection of multiple issues in one piece of code."""
        bad_code = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")

# Issue 1: Missing $ prefix
momentum = df["close"] / df["close"].shift(20) - 1.0

# Issue 2: DataFrame constructor with Series
result = pd.DataFrame(momentum, columns=["Momentum"])
"""

        linter = PitfallLinter()
        results = linter.lint_code(bad_code)

        # Should detect both issues
        pitfall_ids = {r.pitfall.id for r in results}
        assert "PANDAS_001" in pitfall_ids
        assert "PANDAS_004" in pitfall_ids

        # Correction should fix both
        corrected, applied = auto_correct_factor_code(bad_code)
        assert len(applied) >= 2
        assert "$close" in corrected
        assert "to_frame" in corrected


class TestRealWorldScenarios:
    """Test with real-world-like factor code patterns."""

    def test_correct_factor_code(self):
        """Test that correctly written factor code passes all checks."""
        good_code = """
import pandas as pd

# Load Qlib data with correct column names
df = pd.read_hdf("daily_pv.h5", key="data")
df = df.sort_index()

# Calculate 20-day close momentum with correct syntax
close_shift_20 = df.groupby(level="instrument")["$close"].shift(20)
momentum = df["$close"] / close_shift_20 - 1.0

# Use correct to_frame() method
result = momentum.to_frame(name="CloseMomentum_20d")

# Save result
result.to_hdf("result.h5", key="data")
"""

        # Should pass all checks
        assert not has_critical_pitfalls(good_code)

        linter = PitfallLinter()
        critical_results = linter.get_critical_issues(good_code)
        assert len(critical_results) == 0

    def test_volume_weighted_factor(self):
        """Test detection in volume-weighted factor calculation."""
        bad_code = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")

# Calculate VWAP - missing $ prefix
vwap = (df["close"] * df["volume"]).groupby(level="instrument").cumsum() / \
       df["volume"].groupby(level="instrument").cumsum()

result = pd.DataFrame(vwap, columns=["VWAP"])
"""

        # Should detect both issues
        assert has_critical_pitfalls(bad_code)

        linter = PitfallLinter()
        results = linter.lint_code(bad_code)

        # Multiple missing $ prefix issues
        pandas_004 = [r for r in results if r.pitfall.id == "PANDAS_004"]
        assert len(pandas_004) >= 2  # close and volume

        # DataFrame constructor issue
        pandas_001 = [r for r in results if r.pitfall.id == "PANDAS_001"]
        assert len(pandas_001) == 1
