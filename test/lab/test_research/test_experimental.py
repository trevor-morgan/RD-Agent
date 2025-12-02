"""Smoke tests for experimental research modules.

These tests verify that experimental modules can be imported and basic
forward passes work without error. They do NOT validate correctness.
"""

from __future__ import annotations

import pytest

# Skip all tests if torch not installed
torch = pytest.importorskip("torch")


# --- symplectic_templates tests ---


def test_fractional_differencer_import() -> None:
    """Should import FractionalDifferencer."""
    from rdagent_lab.research.symplectic_templates import FractionalDifferencer

    assert FractionalDifferencer is not None


def test_fractional_differencer_forward() -> None:
    """Should run forward pass without error."""
    from rdagent_lab.research.symplectic_templates import FractionalDifferencer

    diff = FractionalDifferencer(order=0.12, window_size=32)
    x = torch.randn(2, 100, 4)
    out = diff(x)

    assert out.shape == x.shape


def test_fractional_differencer_2d_input() -> None:
    """Should handle 2D input by adding dimension."""
    from rdagent_lab.research.symplectic_templates import FractionalDifferencer

    diff = FractionalDifferencer(order=0.12, window_size=16)
    x = torch.randn(2, 50)  # 2D input
    out = diff(x)

    assert out.shape == (2, 50, 1)


def test_symplectic_attention_block_import() -> None:
    """Should import SymplecticAttentionBlock."""
    from rdagent_lab.research.symplectic_templates import SymplecticAttentionBlock

    assert SymplecticAttentionBlock is not None


def test_symplectic_attention_block_forward() -> None:
    """Should run forward pass without error."""
    from rdagent_lab.research.symplectic_templates import SymplecticAttentionBlock

    block = SymplecticAttentionBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 20, 32)
    out = block(x)

    assert out.shape == x.shape


def test_symplectic_transformer_import() -> None:
    """Should import SymplecticTransformer."""
    from rdagent_lab.research.symplectic_templates import SymplecticTransformer

    assert SymplecticTransformer is not None


def test_symplectic_transformer_forward() -> None:
    """Should run forward pass without error."""
    from rdagent_lab.research.symplectic_templates import SymplecticTransformer

    model = SymplecticTransformer(d_feat=6, d_model=32, num_layers=2)
    x = torch.randn(2, 60, 6)
    out = model(x)

    assert out.shape == (2, 60)


def test_symplectic_transformer_different_configs() -> None:
    """Should work with various configurations."""
    from rdagent_lab.research.symplectic_templates import SymplecticTransformer

    # Smaller model
    model = SymplecticTransformer(d_feat=4, d_model=16, n_heads=2, num_layers=1)
    x = torch.randn(1, 30, 4)
    out = model(x)

    assert out.shape == (1, 30)


def test_rough_volatility_factor_template_import() -> None:
    """Should import RoughVolatilityFactorTemplate."""
    from rdagent_lab.research.symplectic_templates import RoughVolatilityFactorTemplate

    assert RoughVolatilityFactorTemplate is not None


def test_rough_volatility_factor_template_generate_code() -> None:
    """Should generate valid Python code string."""
    from rdagent_lab.research.symplectic_templates import RoughVolatilityFactorTemplate

    template = RoughVolatilityFactorTemplate(hurst_exponent=0.1)
    code = template.generate_factor_code(formula_idx=0, window=60)

    assert isinstance(code, str)
    assert "def rough_vol_factor_0" in code
    assert "frac_diff" in code


# --- scenario_generator tests ---


def test_sinusoidal_time_embedding_import() -> None:
    """Should import SinusoidalTimeEmbedding."""
    from rdagent_lab.research.scenario_generator import SinusoidalTimeEmbedding

    assert SinusoidalTimeEmbedding is not None


def test_sinusoidal_time_embedding_forward() -> None:
    """Should generate time embeddings."""
    from rdagent_lab.research.scenario_generator import SinusoidalTimeEmbedding

    embed = SinusoidalTimeEmbedding(dim=64)
    t = torch.tensor([0, 50, 100])
    out = embed(t)

    assert out.shape == (3, 64)


def test_time_series_unet_import() -> None:
    """Should import TimeSeriesUNet."""
    from rdagent_lab.research.scenario_generator import TimeSeriesUNet

    assert TimeSeriesUNet is not None


def test_time_series_unet_forward() -> None:
    """Should run forward pass without error."""
    from rdagent_lab.research.scenario_generator import TimeSeriesUNet

    unet = TimeSeriesUNet(in_channels=4, model_dim=32, time_dim=64)
    x = torch.randn(2, 4, 64)
    t = torch.tensor([10, 50])
    out = unet(x, t)

    assert out.shape == x.shape


def test_diffusion_scheduler_import() -> None:
    """Should import DiffusionScheduler."""
    from rdagent_lab.research.scenario_generator import DiffusionScheduler

    assert DiffusionScheduler is not None


def test_diffusion_scheduler_linear() -> None:
    """Should create linear noise schedule."""
    from rdagent_lab.research.scenario_generator import DiffusionScheduler

    scheduler = DiffusionScheduler(num_timesteps=100, schedule="linear")

    assert len(scheduler.betas) == 100
    assert len(scheduler.alphas) == 100
    assert len(scheduler.alpha_bars) == 100


def test_diffusion_scheduler_cosine() -> None:
    """Should create cosine noise schedule."""
    from rdagent_lab.research.scenario_generator import DiffusionScheduler

    scheduler = DiffusionScheduler(num_timesteps=100, schedule="cosine")

    assert len(scheduler.betas) == 100


def test_diffusion_scheduler_add_noise() -> None:
    """Should add noise to samples."""
    from rdagent_lab.research.scenario_generator import DiffusionScheduler

    scheduler = DiffusionScheduler(num_timesteps=100)
    x0 = torch.randn(2, 4, 64)
    t = torch.tensor([10, 50])
    noisy = scheduler.add_noise(x0, t)

    assert noisy.shape == x0.shape


def test_diffusion_model_import() -> None:
    """Should import DiffusionModel."""
    from rdagent_lab.research.scenario_generator import DiffusionModel

    assert DiffusionModel is not None


def test_diffusion_model_forward() -> None:
    """Should run forward pass (noise prediction)."""
    from rdagent_lab.research.scenario_generator import DiffusionModel

    model = DiffusionModel(in_channels=4, model_dim=32, num_timesteps=100)
    x0 = torch.randn(2, 4, 64)
    t = torch.randint(0, 100, (2,))
    noise = torch.randn_like(x0)

    pred_noise = model(x0, t, noise)

    assert pred_noise.shape == x0.shape


@pytest.mark.slow
def test_diffusion_model_sample() -> None:
    """Should generate samples (slow - few steps only)."""
    from rdagent_lab.research.scenario_generator import DiffusionModel

    # Use very few timesteps for speed
    model = DiffusionModel(in_channels=2, model_dim=16, num_timesteps=5)
    samples = model.sample(shape=(1, 2, 32), device="cpu")

    assert samples.shape == (1, 2, 32)
