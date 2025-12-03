"""Tests for the auto-corrector module."""


from rdagent.components.coder.pitfall_detector import (
    CodeCorrection,
    PitfallCorrector,
    auto_correct_factor_code,
    suggest_factor_corrections,
)


class TestPitfallCorrector:
    """Test cases for PitfallCorrector."""

    def test_correct_dataframe_constructor(self):
        """Test correction of pd.DataFrame(series, columns=[...])."""
        code = """result = pd.DataFrame(momentum, columns=["Momentum"])"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)

        pandas_001_corrections = [c for c in corrections if c.pitfall_id == "PANDAS_001"]
        assert len(pandas_001_corrections) == 1

        correction = pandas_001_corrections[0]
        assert 'to_frame(name="Momentum")' in correction.corrected
        assert correction.confidence >= 0.8

    def test_correct_missing_dollar_prefix(self):
        """Test correction of missing $ prefix."""
        code = """close_price = df["close"]"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)

        pandas_004_corrections = [c for c in corrections if c.pitfall_id == "PANDAS_004"]
        assert len(pandas_004_corrections) == 1

        correction = pandas_004_corrections[0]
        assert "$close" in correction.corrected
        assert correction.confidence >= 0.9

    def test_apply_single_correction(self):
        """Test applying a single correction to code."""
        code = """result = pd.DataFrame(momentum, columns=["Momentum"])"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)

        assert len(corrections) >= 1
        corrected = corrector.apply_correction(code, corrections[0])

        assert "to_frame" in corrected
        assert "pd.DataFrame" not in corrected

    def test_apply_all_corrections(self):
        """Test applying all high-confidence corrections."""
        code = """
import pandas as pd
momentum = df["close"] / df["close"].shift(20) - 1
result = pd.DataFrame(momentum, columns=["Momentum"])
"""

        corrector = PitfallCorrector()
        corrected_code, applied = corrector.apply_all_corrections(code)

        # Should have applied corrections for both issues
        assert len(applied) >= 2
        assert "$close" in corrected_code
        assert "to_frame" in corrected_code

    def test_format_corrections(self):
        """Test formatting of correction suggestions."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)
        formatted = corrector.format_corrections(corrections)

        assert "PANDAS_001" in formatted
        assert "Original:" in formatted
        assert "Corrected:" in formatted

    def test_no_corrections_for_clean_code(self):
        """Test that clean code produces no corrections."""
        code = """
import pandas as pd
momentum = df["$close"] / df["$close"].shift(20) - 1
result = momentum.to_frame(name="Momentum")
"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)

        # Should have no critical corrections
        critical = [c for c in corrections if c.pitfall_id in ("PANDAS_001", "PANDAS_004")]
        assert len(critical) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_suggest_factor_corrections(self):
        """Test suggest_factor_corrections function."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""
        corrections = suggest_factor_corrections(code)

        assert len(corrections) >= 1
        assert any(c.pitfall_id == "PANDAS_001" for c in corrections)

    def test_auto_correct_factor_code(self):
        """Test auto_correct_factor_code function."""
        code = """result = pd.DataFrame(momentum, columns=["Momentum"])"""
        corrected, applied = auto_correct_factor_code(code)

        assert len(applied) >= 1
        assert "to_frame" in corrected


class TestCodeCorrection:
    """Test CodeCorrection dataclass."""

    def test_correction_attributes(self):
        """Test that corrections have required attributes."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        corrector = PitfallCorrector()
        corrections = corrector.suggest_corrections(code)

        for correction in corrections:
            assert isinstance(correction, CodeCorrection)
            assert correction.original
            assert correction.corrected
            assert correction.pitfall_id
            assert 0 <= correction.confidence <= 1
            assert correction.explanation
