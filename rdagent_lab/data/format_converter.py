"""Convert Alpaca data to proper Qlib format.

Fixes the format issue in the original alpaca_collector.py where:
1. The start_index was incorrectly written as a float32 prefix
2. Data was normalized relative to first close price

This converter:
1. Reads existing Alpaca data
2. Strips the erroneous start_index prefix
3. Optionally de-normalizes prices
4. Writes in correct Qlib format

Example:
    >>> from rdagent_lab.data import DataFormatConverter
    >>> converter = DataFormatConverter()
    >>> converter.convert()
    >>> converter.verify()
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def read_alpaca_binary(file_path: Path) -> tuple[int, np.ndarray]:
    """Read binary file in broken Alpaca format.

    The broken format has:
    - First 4 bytes: float32 start_index (should have been uint32)
    - Remaining bytes: float32 data

    Args:
        file_path: Path to binary file

    Returns:
        tuple of (start_index, data_array)
    """
    with open(file_path, "rb") as f:
        all_data = np.fromfile(f, dtype="<f4")

    # First value is the start_index (stored as float)
    start_index = int(all_data[0])
    data = all_data[1:]

    return start_index, data


def write_qlib_binary(file_path: Path, data: np.ndarray) -> None:
    """Write binary file in proper Qlib format.

    Qlib format is simply float32 values with no header.
    The data must be aligned with calendar dates.

    Args:
        file_path: Output file path
        data: Float32 data array
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.astype("<f4").tofile(str(file_path))


def convert_symbol(
    symbol: str,
    input_dir: Path,
    output_dir: Path,
    calendar_len: int,
    denormalize: bool = False,
    first_close: Optional[float] = None,
) -> np.ndarray:
    """Convert a single symbol's data files.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        input_dir: Input Alpaca data directory
        output_dir: Output Qlib data directory
        calendar_len: Expected calendar length
        denormalize: Whether to convert normalized prices back to raw
        first_close: First close price for de-normalization (if known)

    Returns:
        Converted close data array
    """
    input_features = input_dir / "features" / symbol
    output_features = output_dir / "features" / symbol
    output_features.mkdir(parents=True, exist_ok=True)

    fields = ["open", "high", "low", "close", "volume", "factor", "change", "vwap"]
    converted_close = None

    for field in fields:
        input_file = input_features / f"{field}.day.bin"
        if not input_file.exists():
            continue

        # Read broken format
        start_index, data = read_alpaca_binary(input_file)

        # Verify alignment
        if len(data) != calendar_len:
            logger.warning(
                f"{symbol}/{field}: data length {len(data)} != calendar {calendar_len}"
            )
            # Pad or truncate to match calendar
            if len(data) < calendar_len:
                # Pad with NaN at the end
                padded = np.full(calendar_len, np.nan, dtype="<f4")
                padded[start_index : start_index + len(data)] = data
                data = padded
            else:
                # Truncate
                data = data[:calendar_len]

        # De-normalize price fields if requested
        if denormalize and field in ["open", "high", "low", "close", "vwap"]:
            if first_close is not None:
                data = data * first_close
            else:
                logger.warning(f"{symbol}: No first_close provided for de-normalization")

        # Write in proper format
        output_file = output_features / f"{field}.day.bin"
        write_qlib_binary(output_file, data)

        if field == "close":
            converted_close = data

    return converted_close


def convert_all(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    denormalize: bool = False,
) -> None:
    """Convert all Alpaca data to Qlib format.

    Args:
        input_dir: Input Alpaca data directory (default: ~/.qlib/qlib_data/alpaca_us)
        output_dir: Output Qlib data directory (default: ~/.qlib/qlib_data/alpaca_us_fixed)
        symbols: List of symbols to convert (default: all)
        denormalize: Whether to de-normalize prices
    """
    if input_dir is None:
        input_dir = Path.home() / ".qlib/qlib_data/alpaca_us"
    else:
        input_dir = Path(input_dir).expanduser()

    if output_dir is None:
        output_dir = Path.home() / ".qlib/qlib_data/alpaca_us_fixed"
    else:
        output_dir = Path(output_dir).expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    logger.info(f"Converting: {input_dir} -> {output_dir}")

    # Read calendar
    calendar_file = input_dir / "calendars/day.txt"
    with open(calendar_file) as f:
        calendar = [line.strip() for line in f if line.strip()]
    calendar_len = len(calendar)
    logger.info(f"Calendar: {calendar_len} trading days ({calendar[0]} to {calendar[-1]})")

    # Copy calendar
    output_calendars = output_dir / "calendars"
    output_calendars.mkdir(parents=True, exist_ok=True)
    shutil.copy(calendar_file, output_calendars / "day.txt")

    # Copy instruments
    output_instruments = output_dir / "instruments"
    output_instruments.mkdir(parents=True, exist_ok=True)
    for inst_file in (input_dir / "instruments").glob("*.txt"):
        shutil.copy(inst_file, output_instruments / inst_file.name)

    # Get symbols to convert
    if symbols is None:
        features_dir = input_dir / "features"
        symbols = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
    elif isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    logger.info(f"Converting {len(symbols)} symbols...")

    # Convert each symbol
    success_count = 0
    for i, symbol in enumerate(symbols):
        try:
            convert_symbol(
                symbol=symbol,
                input_dir=input_dir,
                output_dir=output_dir,
                calendar_len=calendar_len,
                denormalize=denormalize,
            )
            success_count += 1
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(symbols)}")
        except Exception as e:
            logger.error(f"Error converting {symbol}: {e}")

    logger.info(f"Converted {success_count}/{len(symbols)} symbols")
    logger.info(f"Output: {output_dir}")


def verify(
    data_dir: Optional[str] = None,
) -> bool:
    """Verify Qlib can load the converted data.

    Args:
        data_dir: Data directory to verify (default: ~/.qlib/qlib_data/alpaca_us_fixed)

    Returns:
        True if verification passes
    """
    try:
        import qlib
        from qlib.data import D
    except ImportError:
        logger.error("Qlib not installed. Run: pip install qlib")
        return False

    if data_dir is None:
        data_dir = Path.home() / ".qlib/qlib_data/alpaca_us_fixed"
    else:
        data_dir = Path(data_dir).expanduser()

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False

    logger.info(f"Verifying: {data_dir}")

    # Initialize Qlib
    qlib.init(provider_uri=str(data_dir), region="us")

    # Test loading data
    df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume", "$high", "$low", "$open"],
        start_time="2020-01-02",
        end_time="2020-12-31",
    )

    if len(df) == 0:
        logger.error("Qlib returned empty DataFrame!")
        return False

    logger.info(f"Loaded {len(df)} rows for AAPL")
    logger.info(f"Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    logger.info(f"Close price range: ${df['$close'].min():.2f} - ${df['$close'].max():.2f}")

    # Check for NaN
    nan_pct = df.isna().sum().sum() / df.size * 100
    if nan_pct > 10:
        logger.warning(f"High NaN percentage: {nan_pct:.1f}%")

    logger.info("Verification passed!")
    return True


def compare(
    original_dir: Optional[str] = None,
    fixed_dir: Optional[str] = None,
    symbol: str = "AAPL",
) -> None:
    """Compare original and fixed data formats.

    Args:
        original_dir: Original Alpaca data directory
        fixed_dir: Fixed Qlib data directory
        symbol: Symbol to compare
    """
    if original_dir is None:
        original_dir = Path.home() / ".qlib/qlib_data/alpaca_us"
    else:
        original_dir = Path(original_dir).expanduser()

    if fixed_dir is None:
        fixed_dir = Path.home() / ".qlib/qlib_data/alpaca_us_fixed"
    else:
        fixed_dir = Path(fixed_dir).expanduser()

    print(f"\n=== Comparing {symbol} ===\n")

    # Original format
    orig_file = original_dir / f"features/{symbol}/close.day.bin"
    if orig_file.exists():
        with open(orig_file, "rb") as f:
            orig_data = np.fromfile(f, dtype="<f4")
        print(f"Original: {len(orig_data)} values")
        print(f"  First 5: {orig_data[:5]}")
        print(f"  Has start_index prefix: {orig_data[0] == 0.0}")

    # Fixed format
    fixed_file = fixed_dir / f"features/{symbol}/close.day.bin"
    if fixed_file.exists():
        with open(fixed_file, "rb") as f:
            fixed_data = np.fromfile(f, dtype="<f4")
        print(f"\nFixed: {len(fixed_data)} values")
        print(f"  First 5: {fixed_data[:5]}")

    # Calendar
    cal_file = original_dir / "calendars/day.txt"
    if cal_file.exists():
        with open(cal_file) as f:
            cal_len = len([line for line in f if line.strip()])
        print(f"\nCalendar: {cal_len} days")
        print(f"Original aligned: {len(orig_data) - 1 == cal_len}")
        print(f"Fixed aligned: {len(fixed_data) == cal_len}")


class DataFormatConverter:
    """CLI interface for the converter.

    Converts Alpaca data format to proper Qlib format.

    Example:
        >>> converter = DataFormatConverter()
        >>> converter.convert()
        >>> converter.verify()
    """

    def convert(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        symbols: Optional[str] = None,
        denormalize: bool = False,
    ) -> None:
        """Convert Alpaca data to Qlib format.

        Args:
            input_dir: Input directory (default: ~/.qlib/qlib_data/alpaca_us)
            output_dir: Output directory (default: ~/.qlib/qlib_data/alpaca_us_fixed)
            symbols: Comma-separated symbols (default: all)
            denormalize: De-normalize prices
        """
        symbol_list = None
        if symbols:
            if isinstance(symbols, (list, tuple)):
                symbol_list = list(symbols)
            else:
                symbol_list = [s.strip() for s in symbols.split(",")]

        convert_all(
            input_dir=input_dir,
            output_dir=output_dir,
            symbols=symbol_list,
            denormalize=denormalize,
        )

    def verify(self, data_dir: Optional[str] = None) -> bool:
        """Verify converted data works with Qlib.

        Args:
            data_dir: Data directory to verify

        Returns:
            True if verification passes
        """
        return verify(data_dir=data_dir)

    def compare(
        self,
        symbol: str = "AAPL",
        original_dir: Optional[str] = None,
        fixed_dir: Optional[str] = None,
    ) -> None:
        """Compare original and fixed data.

        Args:
            symbol: Symbol to compare
            original_dir: Original data directory
            fixed_dir: Fixed data directory
        """
        compare(
            original_dir=original_dir,
            fixed_dir=fixed_dir,
            symbol=symbol,
        )


__all__ = [
    "DataFormatConverter",
    "read_alpaca_binary",
    "write_qlib_binary",
    "convert_symbol",
    "convert_all",
    "verify",
    "compare",
]
