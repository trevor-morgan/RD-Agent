import os

import qlib

# Support multiple data regions via QLIB_DATA_REGION environment variable
# Options: cn_data (default), us_data, alpaca_us
data_region = os.environ.get("QLIB_DATA_REGION", "cn_data")
provider_uri = f"~/.qlib/qlib_data/{data_region}"
print(f"Initializing Qlib with data region: {data_region} ({provider_uri})")

qlib.init(provider_uri=provider_uri)

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

# Adjust date range based on data region
# US data typically starts later than CN data
if data_region in ("us_data", "alpaca_us"):
    start_date = "2010-01-01"
    debug_start = "2018-01-01"
    debug_end = "2020-12-31"
else:
    start_date = "2008-12-29"
    debug_start = "2018-01-01"
    debug_end = "2019-12-31"

print(f"Fetching data from {start_date}...")
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc[start_date:].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")
print(f"Saved daily_pv_all.h5 with {len(data)} rows")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
print(f"Fetching debug data from {debug_start} to {debug_end}...")
data = (
    (
        D.features(instruments, fields, start_time=debug_start, end_time=debug_end, freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data.to_hdf("./daily_pv_debug.h5", key="data")
print(f"Saved daily_pv_debug.h5 with {len(data)} rows")
