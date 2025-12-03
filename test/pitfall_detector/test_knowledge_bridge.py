"""Tests for the knowledge bridge module."""


from rdagent.components.coder.pitfall_detector import (
    PitfallKnowledgeBridge,
    get_pitfall_knowledge_bridge,
    get_pitfall_prompt_context,
)


class TestPitfallKnowledgeBridge:
    """Test cases for PitfallKnowledgeBridge."""

    def test_builtin_knowledge_loaded(self):
        """Test that built-in knowledge is loaded on init."""
        bridge = PitfallKnowledgeBridge()

        # Should have entries for built-in patterns
        assert len(bridge.entries) >= 5
        assert any(e.pitfall_id == "PANDAS_001" for e in bridge.entries)
        assert any(e.pitfall_id == "PANDAS_004" for e in bridge.entries)

    def test_query_similar_pitfalls(self):
        """Test querying for similar pitfalls in code."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        bridge = PitfallKnowledgeBridge()
        similar = bridge.query_similar_pitfalls(code)

        assert len(similar) >= 1
        assert any(e.pitfall_id == "PANDAS_001" for e in similar)

    def test_query_no_pitfalls(self):
        """Test querying clean code returns no matches."""
        code = """result = x.to_frame(name="Y")"""

        bridge = PitfallKnowledgeBridge()
        similar = bridge.query_similar_pitfalls(code)

        # Should not match PANDAS_001
        assert not any(e.pitfall_id == "PANDAS_001" for e in similar)

    def test_get_correction_context(self):
        """Test getting correction context for a pitfall."""
        bridge = PitfallKnowledgeBridge()
        context = bridge.get_correction_context("PANDAS_001")

        assert "PANDAS_001" in context
        assert "multiindex" in context.lower() or "series" in context.lower()
        assert "to_frame" in context.lower()

    def test_format_prompt_context(self):
        """Test formatting prompt context for code with pitfalls."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        bridge = PitfallKnowledgeBridge()
        context = bridge.format_prompt_context(code)

        assert "Pitfall" in context
        assert "PANDAS_001" in context

    def test_format_prompt_context_clean_code(self):
        """Test that clean code produces empty prompt context."""
        code = """result = x.to_frame(name="Y")"""

        bridge = PitfallKnowledgeBridge()
        context = bridge.format_prompt_context(code)

        # Clean code should not produce any context
        assert context == ""

    def test_register_custom_pitfall(self):
        """Test registering a custom pitfall."""
        bridge = PitfallKnowledgeBridge()
        initial_count = len(bridge.entries)

        bridge.register_pitfall(
            pitfall_id="CUSTOM_001",
            bad_code="bad_code_example",
            good_code="good_code_example",
            error_context="Custom error",
            notes="Custom notes",
        )

        assert len(bridge.entries) == initial_count + 1
        assert any(e.pitfall_id == "CUSTOM_001" for e in bridge.entries)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_pitfall_knowledge_bridge_singleton(self):
        """Test that get_pitfall_knowledge_bridge returns singleton."""
        bridge1 = get_pitfall_knowledge_bridge()
        bridge2 = get_pitfall_knowledge_bridge()

        assert bridge1 is bridge2

    def test_get_pitfall_prompt_context(self):
        """Test get_pitfall_prompt_context function."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""
        context = get_pitfall_prompt_context(code)

        assert "PANDAS_001" in context
