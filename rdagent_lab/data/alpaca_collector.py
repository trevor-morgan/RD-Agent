"""Alpaca Data Collector for Qlib.

Downloads historical stock data from Alpaca Markets and converts
to Qlib binary format. Alpaca provides free, clean, unlimited
historical data for US stocks.

Example:
    >>> from rdagent_lab.data import AlpacaCollector
    >>> collector = AlpacaCollector()
    >>> collector.download(start="2020-01-01", end="2025-11-26")
    >>> collector.info()

Requirements:
    pip install alpaca-py

Environment Variables:
    ALPACA_API_KEY: Your Alpaca API key
    ALPACA_SECRET_KEY: Your Alpaca secret key
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# S&P 500 symbols (as of 2024)
SP500_SYMBOLS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL",
    "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET",
    "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX",
    "BBWI", "BBY", "BDX", "BEN", "BF.B", "BG", "BIIB", "BIO", "BK", "BKNG",
    "BKR", "BLDR", "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BX",
    "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI",
    "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR",
    "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS",
    "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT",
    "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH",
    "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD", "DE", "DECK",
    "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV",
    "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY",
    "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH",
    "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "ETSY",
    "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX",
    "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FLT", "FMC",
    "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC",
    "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD",
    "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF",
    "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX",
    "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW",
    "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL",
    "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK",
    "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW",
    "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE",
    "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX",
    "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR",
    "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO",
    "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB",
    "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT",
    "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH",
    "ZBRA", "ZTS"
]


def get_alpaca_client():
    """Get Alpaca stock historical data client.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    Get free API keys at: https://alpaca.markets/

    Returns:
        StockHistoricalDataClient instance
    """
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required.\n"
            "Get free API keys at: https://alpaca.markets/"
        )

    return StockHistoricalDataClient(api_key, secret_key)


def download_bars(
    symbols: list[str],
    start: str,
    end: str,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Download historical bars from Alpaca.

    Args:
        symbols: List of stock symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (1Day, 1Hour, 1Min)

    Returns:
        DataFrame with OHLCV data
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = get_alpaca_client()

    tf_map = {
        "1Day": TimeFrame.Day,
        "1Hour": TimeFrame.Hour,
        "1Min": TimeFrame.Minute,
    }

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf_map.get(timeframe, TimeFrame.Day),
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
    )

    logger.info(f"Downloading {len(symbols)} symbols from {start} to {end}...")
    bars = client.get_stock_bars(request)

    # Convert to DataFrame
    df = bars.df
    if df.empty:
        logger.warning("No data returned from Alpaca")
        return df

    # Reset index to get symbol and timestamp as columns
    df = df.reset_index()
    df = df.rename(columns={
        "symbol": "instrument",
        "timestamp": "datetime",
        "trade_count": "volume_count",
        "vwap": "vwap",
    })

    # Ensure datetime is date only for daily data
    if timeframe == "1Day":
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.date

    logger.info(f"Downloaded {len(df):,} bars for {df['instrument'].nunique()} symbols")
    return df


def convert_to_qlib_format(
    df: pd.DataFrame,
    output_dir: Path,
    normalize: bool = True,
) -> None:
    """Convert DataFrame to Qlib binary format.

    Qlib expects:
    - features/{symbol}/ directory with binary files for each field
    - instruments/ directory with instrument lists
    - calendars/ directory with trading calendars

    Args:
        df: DataFrame with columns [datetime, instrument, open, high, low, close, volume]
        output_dir: Output directory for Qlib data
        normalize: Whether to normalize prices (required for Alpha158/Alpha360)
    """
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    instruments_dir = output_dir / "instruments"
    calendars_dir = output_dir / "calendars"

    # Create directories
    features_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    calendars_dir.mkdir(parents=True, exist_ok=True)

    # Get all dates and create calendar
    all_dates = sorted(df["datetime"].unique())
    calendar_file = calendars_dir / "day.txt"
    with open(calendar_file, "w") as f:
        for date in all_dates:
            f.write(f"{date}\n")
    logger.info(f"Created calendar with {len(all_dates)} trading days")

    # Get all instruments
    instruments = sorted(df["instrument"].unique())

    # Create instrument list
    min_date = min(all_dates)
    max_date = max(all_dates)
    instruments_file = instruments_dir / "all.txt"
    with open(instruments_file, "w") as f:
        for inst in instruments:
            f.write(f"{inst}\t{min_date}\t{max_date}\n")

    # Also create sp500.txt
    sp500_file = instruments_dir / "sp500.txt"
    with open(sp500_file, "w") as f:
        for inst in instruments:
            f.write(f"{inst}\t{min_date}\t{max_date}\n")
    logger.info(f"Created instrument lists with {len(instruments)} symbols")

    # Create date index for binary files
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

    # Sort and compute derived fields
    df_sorted = df.sort_values(["instrument", "datetime"]).copy()

    # Compute daily return (change)
    df_sorted["change"] = df_sorted.groupby("instrument")["close"].pct_change().fillna(0)

    # Normalize prices if requested
    if normalize:
        logger.info("Normalizing prices relative to first close (Qlib format)...")
        price_cols = ["open", "high", "low", "close"]
        if "vwap" in df_sorted.columns:
            price_cols.append("vwap")

        def normalize_prices(group):
            """Normalize prices relative to first close price."""
            first_close = group["close"].iloc[0]
            if first_close == 0 or pd.isna(first_close):
                first_close = 1.0
            for col in price_cols:
                if col in group.columns:
                    group[col] = group[col] / first_close
            return group

        normalized_dfs = []
        for inst, group in df_sorted.groupby("instrument", group_keys=False):
            normalized_group = normalize_prices(group.copy())
            normalized_dfs.append(normalized_group)
        df_sorted = pd.concat(normalized_dfs, ignore_index=False)
        df_sorted["factor"] = 1.0
    else:
        df_sorted["factor"] = 1.0

    df = df_sorted

    # Fields to save
    fields = ["open", "high", "low", "close", "volume", "factor", "change"]
    if "vwap" in df.columns:
        fields.append("vwap")

    # Process each instrument
    for inst in instruments:
        inst_dir = features_dir / inst
        inst_dir.mkdir(exist_ok=True)

        inst_df = df[df["instrument"] == inst].sort_values("datetime")

        # Find the starting index for this instrument in the calendar
        first_date = inst_df["datetime"].iloc[0]
        date_index = date_to_idx[first_date]

        for field in fields:
            if field not in inst_df.columns:
                continue

            # Create data array aligned to calendar
            values = []
            for _, row in inst_df.iterrows():
                values.append(float(row[field]))

            # Qlib binary format: [date_index, data...]
            bin_data = np.hstack([[date_index], values]).astype("<f")

            # Write binary file
            bin_file = inst_dir / f"{field}.day.bin"
            bin_data.tofile(str(bin_file))

    logger.info(f"Created Qlib binary data for {len(instruments)} instruments in {output_dir}")


class AlpacaCollector:
    """Alpaca data collector for Qlib.

    Downloads historical stock data from Alpaca Markets API
    and converts it to Qlib binary format.

    Attributes:
        output_dir: Output directory for Qlib data

    Example:
        >>> collector = AlpacaCollector()
        >>> collector.download(start="2020-01-01", end="2025-11-26")
        >>> collector.info()
    """

    def __init__(
        self,
        output_dir: str = "~/.qlib/qlib_data/alpaca_us",
    ):
        """Initialize collector.

        Args:
            output_dir: Output directory for Qlib data
        """
        self.output_dir = Path(output_dir).expanduser()

    def download(
        self,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        symbols: Optional[str] = None,
        use_sp500: bool = True,
    ) -> None:
        """Download historical data from Alpaca.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            symbols: Comma-separated list of symbols, or None for S&P 500
            use_sp500: Use S&P 500 symbols (default True)

        Example:
            >>> collector.download(start="2020-01-01", end="2025-11-26")
            >>> collector.download(symbols="AAPL,MSFT,GOOGL", start="2020-01-01")
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        elif use_sp500:
            symbol_list = SP500_SYMBOLS
        else:
            raise ValueError("Must specify --symbols or use --use_sp500")

        logger.info(f"Downloading {len(symbol_list)} symbols from {start} to {end}")

        # Download in batches (Alpaca allows up to 100 symbols per request)
        batch_size = 100
        all_dfs = []

        for i in range(0, len(symbol_list), batch_size):
            batch = symbol_list[i:i + batch_size]
            logger.info(f"Batch {i // batch_size + 1}: {len(batch)} symbols")

            try:
                df = download_bars(batch, start, end)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error downloading batch: {e}")
                continue

        if not all_dfs:
            logger.error("No data downloaded!")
            return

        # Combine all data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total: {len(combined_df):,} bars for {combined_df['instrument'].nunique()} symbols")

        # Convert to Qlib format
        convert_to_qlib_format(combined_df, self.output_dir)

        logger.info(f"\nData saved to: {self.output_dir}")
        logger.info(f"  Date range: {start} to {end}")
        logger.info(f"  Symbols: {combined_df['instrument'].nunique()}")
        logger.info(f"  Total bars: {len(combined_df):,}")
        logger.info(f"\nTo use with Qlib:")
        logger.info(f"  qlib.init(provider_uri='{self.output_dir}', region='us')")

    def update(self) -> None:
        """Update existing data to current date.

        Reads the existing calendar to find the last date and downloads
        new data from that date to today.
        """
        calendar_file = self.output_dir / "calendars" / "day.txt"

        if not calendar_file.exists():
            logger.error(f"No existing data found at {self.output_dir}")
            logger.info("Run 'download' first to create initial dataset")
            return

        # Read last date from calendar
        with open(calendar_file) as f:
            dates = [line.strip() for line in f if line.strip()]

        if not dates:
            logger.error("Calendar file is empty")
            return

        last_date = dates[-1]
        start_date = (datetime.strptime(str(last_date), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date >= end_date:
            logger.info("Data is already up to date!")
            return

        logger.info(f"Updating data from {start_date} to {end_date}")
        self.download(start=start_date, end=end_date)

    def info(self) -> None:
        """Show information about existing data."""
        calendar_file = self.output_dir / "calendars" / "day.txt"
        instruments_file = self.output_dir / "instruments" / "all.txt"

        if not self.output_dir.exists():
            logger.info(f"No data found at {self.output_dir}")
            return

        print(f"\nQlib Data Directory: {self.output_dir}")

        if calendar_file.exists():
            with open(calendar_file) as f:
                dates = [line.strip() for line in f if line.strip()]
            print(f"  Trading days: {len(dates)}")
            if dates:
                print(f"  Date range: {dates[0]} to {dates[-1]}")

        if instruments_file.exists():
            with open(instruments_file) as f:
                instruments = [line.split("\t")[0] for line in f if line.strip()]
            print(f"  Instruments: {len(instruments)}")

        features_dir = self.output_dir / "features"
        if features_dir.exists():
            symbols = list(features_dir.iterdir())
            print(f"  Feature directories: {len(symbols)}")


__all__ = [
    "AlpacaCollector",
    "SP500_SYMBOLS",
    "download_bars",
    "convert_to_qlib_format",
    "get_alpaca_client",
]
