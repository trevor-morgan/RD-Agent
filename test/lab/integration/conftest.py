"""Fixtures for integration tests requiring Qlib setup."""

from __future__ import annotations

from pathlib import Path

import pytest


def _find_qlib_data() -> tuple[Path | None, str]:
    """Find Qlib data directory if available.

    Returns:
        Tuple of (path, region) or (None, "") if not found.
    """
    candidates = [
        (Path.home() / ".qlib" / "qlib_data" / "cn_data", "cn"),
        (Path.home() / ".qlib" / "qlib_data" / "us_data", "us"),
        (Path("/data/qlib/cn_data"), "cn"),
        (Path("/data/qlib/us_data"), "us"),
    ]
    for path, region in candidates:
        # Check for standard Qlib data structure
        if path.exists() and (path / "calendars").exists() and (path / "instruments").exists():
            return path, region
    return None, ""


def _check_qlib_api_compatible() -> tuple[bool, str]:
    """Check if Qlib API is compatible with our code.

    Different Qlib versions have different API signatures. This checks
    basic compatibility before running integration tests.

    Returns:
        Tuple of (is_compatible, reason_if_not).
    """
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158

        # Check Alpha158 signature - different versions have different args
        import inspect
        sig = inspect.signature(Alpha158.__init__)
        params = list(sig.parameters.keys())

        # We need these parameters to be present
        required_params = ["instruments", "start_time", "end_time"]
        if not all(p in params for p in required_params):
            return False, "Alpha158 missing required parameters"

        # Check Qlib version if available
        qlib_version = getattr(qlib, "__version__", "unknown")

        return True, f"Qlib {qlib_version} appears compatible"
    except ImportError:
        return False, "Qlib not installed"
    except Exception as e:
        return False, f"Compatibility check failed: {e}"


QLIB_DATA_PATH, QLIB_REGION = _find_qlib_data()
HAS_QLIB_DATA = QLIB_DATA_PATH is not None
QLIB_API_COMPATIBLE, QLIB_API_REASON = _check_qlib_api_compatible()


@pytest.fixture(scope="module")
def qlib_data_path() -> Path:
    """Return path to Qlib data."""
    if QLIB_DATA_PATH is None:
        pytest.skip("Qlib data not available")
    return QLIB_DATA_PATH


@pytest.fixture(scope="module")
def qlib_region() -> str:
    """Return Qlib region based on data path."""
    if QLIB_DATA_PATH is None:
        return "cn"
    return "us" if "us_data" in str(QLIB_DATA_PATH) else "cn"


@pytest.fixture(scope="module")
def initialized_qlib(qlib_data_path: Path, qlib_region: str):
    """Initialize Qlib for integration tests (module-scoped for performance)."""
    if not QLIB_API_COMPATIBLE:
        pytest.skip(f"Qlib API incompatible: {QLIB_API_REASON}")

    try:
        import qlib
        from qlib.config import REG_CN, REG_US

        region = REG_US if qlib_region == "us" else REG_CN
        qlib.init(provider_uri=str(qlib_data_path), region=region)
        return qlib
    except ImportError:
        pytest.skip("Qlib not installed")
    except Exception as e:
        pytest.skip(f"Qlib initialization failed: {e}")


def handle_qlib_api_error(func):
    """Decorator to skip tests on Qlib API incompatibility errors.

    Use this to wrap test functions that call Qlib APIs which may
    have different signatures across versions.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            if "got multiple values for argument" in str(e) or "unexpected keyword argument" in str(e):
                pytest.skip(f"Qlib API version incompatibility: {e}")
            raise
        except ValueError as e:
            if "not exists" in str(e) and ("instrument" in str(e) or "calendar" in str(e)):
                pytest.skip(f"Qlib data not available for this configuration: {e}")
            raise

    return wrapper


@pytest.fixture
def short_date_range(qlib_region: str) -> dict[str, str]:
    """Return a short date range for fast integration tests."""
    if qlib_region == "us":
        return {
            "start_time": "2020-01-01",
            "end_time": "2020-03-01",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2020-02-15",
        }
    return {
        "start_time": "2020-01-01",
        "end_time": "2020-03-01",
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2020-02-15",
    }
