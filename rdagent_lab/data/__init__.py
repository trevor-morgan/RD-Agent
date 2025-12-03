"""Data collection and format conversion utilities.

This module provides tools for downloading market data from various
sources and converting to Qlib format.

Components:
    - AlpacaCollector: Download historical data from Alpaca Markets
    - DataFormatConverter: Fix/convert data to proper Qlib format
    - SP500_SYMBOLS: List of S&P 500 constituent symbols

Example:
    >>> from rdagent_lab.data import AlpacaCollector
    >>> collector = AlpacaCollector()
    >>> collector.download(start="2020-01-01", end="2025-11-26")

    >>> from rdagent_lab.data import DataFormatConverter
    >>> converter = DataFormatConverter()
    >>> converter.convert()
    >>> converter.verify()
"""

from rdagent_lab.data.alpaca_collector import (
    AlpacaCollector,
    SP500_SYMBOLS,
    convert_to_qlib_format,
    download_bars,
    get_alpaca_client,
)
from rdagent_lab.data.format_converter import (
    DataFormatConverter,
    compare,
    convert_all,
    convert_symbol,
    read_alpaca_binary,
    verify,
    write_qlib_binary,
)

__all__ = [
    # Alpaca collector
    "AlpacaCollector",
    "SP500_SYMBOLS",
    "download_bars",
    "convert_to_qlib_format",
    "get_alpaca_client",
    # Format converter
    "DataFormatConverter",
    "read_alpaca_binary",
    "write_qlib_binary",
    "convert_symbol",
    "convert_all",
    "verify",
    "compare",
]
