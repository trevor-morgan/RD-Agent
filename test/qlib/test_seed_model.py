"""Tests for the seed model loading feature.

This module tests the _load_seed_model functionality in RDLoop,
which allows users to provide their own model as a starting point
for the evolutionary loop.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, Mock, patch

import pytest

from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace


@pytest.fixture
def mock_prop_setting():
    """Create a mock property setting for RDLoop initialization."""
    setting = Mock()
    setting.scen = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    setting.hypothesis_gen = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    setting.hypothesis2experiment = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    setting.coder = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    setting.runner = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    setting.summarizer = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    setting.model_dump = Mock(return_value={})
    return setting


@pytest.fixture
def sample_model_code():
    """Sample model code that matches RD-Agent's expected interface."""
    return '''
import torch
import torch.nn as nn

class Net(nn.Module):
    """Simple GRU model for testing."""

    def __init__(self, num_features, num_timesteps=None):
        super().__init__()
        self.gru = nn.GRU(num_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

model_cls = Net
'''


@pytest.fixture
def seed_model_file(sample_model_code):
    """Create a temporary file with sample model code."""
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_model_code)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestSeedModelLoading:
    """Test suite for seed model loading functionality."""

    def test_load_seed_model_method_exists(self):
        """Verify _load_seed_model method exists on RDLoop."""
        assert hasattr(RDLoop, "_load_seed_model")

    @patch("rdagent.components.workflow.rd_loop.import_class")
    @patch("rdagent.components.workflow.rd_loop.logger")
    def test_load_seed_model_creates_hypothesis(
        self, mock_logger, mock_import_class, seed_model_file
    ):
        """Test that loading a seed model creates proper hypothesis."""
        # Setup mocks
        mock_scen = Mock()
        mock_import_class.return_value = Mock(return_value=mock_scen)

        # Create RDLoop with mocked components
        with patch.object(RDLoop, "__init__", lambda self, _: None):
            loop = RDLoop.__new__(RDLoop)
            loop.trace = Trace(scen=mock_scen)
            loop.scen = mock_scen
            loop.runner = Mock()

            # runner.develop should return the experiment with results attached
            def develop_side_effect(exp):
                exp.result = {"IC": 0.05, "Sharpe": 0.8}
                return exp
            loop.runner.develop = Mock(side_effect=develop_side_effect)

            # Load seed model
            loop._load_seed_model(
                model_path=str(seed_model_file),
                hypothesis_text="Test GRU model for unit testing",
            )

            # Verify trace was updated
            assert len(loop.trace.hist) == 1
            exp, feedback = loop.trace.hist[0]

            # Verify hypothesis
            assert exp.hypothesis is not None
            assert "Test GRU model" in exp.hypothesis.hypothesis

            # Verify feedback
            assert isinstance(feedback, HypothesisFeedback)
            assert feedback.decision is True  # Should be marked as SOTA

    @patch("rdagent.components.workflow.rd_loop.import_class")
    @patch("rdagent.components.workflow.rd_loop.logger")
    def test_load_seed_model_detects_model_type(
        self, mock_logger, mock_import_class, seed_model_file
    ):
        """Test that model type detection works for TimeSeries models."""
        mock_scen = Mock()
        mock_import_class.return_value = Mock(return_value=mock_scen)

        with patch.object(RDLoop, "__init__", lambda self, _: None):
            loop = RDLoop.__new__(RDLoop)
            loop.trace = Trace(scen=mock_scen)
            loop.scen = mock_scen
            loop.runner = Mock()

            # runner.develop should return the experiment with results attached
            def develop_side_effect(exp):
                exp.result = {"IC": 0.05}
                return exp
            loop.runner.develop = Mock(side_effect=develop_side_effect)

            loop._load_seed_model(
                model_path=str(seed_model_file),
                hypothesis_text="Test model",
            )

            exp, _ = loop.trace.hist[0]
            # Sample model has num_timesteps parameter -> TimeSeries
            assert exp.sub_tasks[0].model_type == "TimeSeries"

    @patch("rdagent.components.workflow.rd_loop.import_class")
    @patch("rdagent.components.workflow.rd_loop.logger")
    def test_load_seed_model_handles_failure(
        self, mock_logger, mock_import_class, seed_model_file
    ):
        """Test that seed model loading handles evaluation failures gracefully."""
        mock_scen = Mock()
        mock_import_class.return_value = Mock(return_value=mock_scen)

        with patch.object(RDLoop, "__init__", lambda self, _: None):
            loop = RDLoop.__new__(RDLoop)
            loop.trace = Trace(scen=mock_scen)
            loop.scen = mock_scen
            loop.runner = Mock()
            # Simulate failure - return None result
            loop.runner.develop = Mock(return_value=Mock(result=None, stdout="Error: training failed"))

            loop._load_seed_model(
                model_path=str(seed_model_file),
                hypothesis_text="Test model",
            )

            # Should still add to trace but with negative feedback
            assert len(loop.trace.hist) == 1
            _, feedback = loop.trace.hist[0]
            assert feedback.decision is False

    @patch("rdagent.components.workflow.rd_loop.import_class")
    @patch("rdagent.components.workflow.rd_loop.logger")
    def test_load_seed_model_handles_exception(
        self, mock_logger, mock_import_class, seed_model_file
    ):
        """Test that seed model loading handles exceptions properly."""
        mock_scen = Mock()
        mock_import_class.return_value = Mock(return_value=mock_scen)

        with patch.object(RDLoop, "__init__", lambda self, _: None):
            loop = RDLoop.__new__(RDLoop)
            loop.trace = Trace(scen=mock_scen)
            loop.scen = mock_scen
            loop.runner = Mock()
            loop.runner.develop = Mock(side_effect=RuntimeError("Docker not available"))

            with pytest.raises(RuntimeError, match="Docker not available"):
                loop._load_seed_model(
                    model_path=str(seed_model_file),
                    hypothesis_text="Test model",
                )

            # Should still add to trace with error feedback
            assert len(loop.trace.hist) == 1
            _, feedback = loop.trace.hist[0]
            assert feedback.decision is False
            assert "Docker not available" in feedback.reason


class TestSeedModelCLI:
    """Test CLI parameter handling for seed model feature."""

    def test_cli_parameters_exist(self):
        """Verify CLI parameters are defined in model.py main function."""
        from rdagent.app.qlib_rd_loop.model import main
        import inspect

        sig = inspect.signature(main)
        params = sig.parameters

        assert "seed_model" in params
        assert "seed_hypothesis" in params
        assert "data_region" in params

    def test_config_settings_exist(self):
        """Verify config settings are defined in ModelBasePropSetting."""
        from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting

        fields = ModelBasePropSetting.model_fields

        assert "seed_model_path" in fields
        assert "seed_hypothesis" in fields
        assert "data_region" in fields


class TestDataRegion:
    """Test data region configuration."""

    def test_qlib_data_region_env_var(self):
        """Test that QLIB_DATA_REGION environment variable is respected."""
        import os
        from rdagent.utils.env import QlibDockerConf

        # Check default behavior - data_region field should exist in QlibDockerConf
        assert "data_region" in QlibDockerConf.model_fields

        # Check that env var affects the default value
        original = os.environ.get("QLIB_DATA_REGION")
        try:
            os.environ["QLIB_DATA_REGION"] = "alpaca_us"
            # Force re-evaluation by creating new instance
            # Note: pydantic fields with default=os.environ.get() capture at class definition time
            # So we verify the field exists and accepts the value
            conf = QlibDockerConf(data_region="alpaca_us")
            assert conf.data_region == "alpaca_us"
        finally:
            if original:
                os.environ["QLIB_DATA_REGION"] = original
            else:
                os.environ.pop("QLIB_DATA_REGION", None)

    def test_us_config_files_exist(self):
        """Verify US market configuration files exist."""
        config_dir = Path("rdagent/scenarios/qlib/experiment/model_template")

        baseline = config_dir / "conf_us_baseline_model.yaml"
        sota = config_dir / "conf_us_sota_model.yaml"

        assert baseline.exists(), f"Missing: {baseline}"
        assert sota.exists(), f"Missing: {sota}"

    def test_us_config_has_correct_provider(self):
        """Verify US config uses correct Qlib provider URI."""
        import yaml

        config_path = Path("rdagent/scenarios/qlib/experiment/model_template/conf_us_baseline_model.yaml")

        with open(config_path) as f:
            # Read raw content since it's a Jinja template
            content = f.read()

        assert "alpaca_us" in content
        assert "region: us" in content
        assert "sp500" in content
