# PJM Market Data Analysis

**Chat:** PJM Market Data Analysis
**Date:** 2/21/2026, 9:40:15 PM

---

## User

I have market data for the past month of PJM for various DA/RT LMPs, loads, some forecast, and uplift charges for American Electric Power. I also have generation by fuel types. I need you to do some analysis and give updates on whats been going on in the market, especially any notable changes/events, like oil being used, LMPs exploding, etc. 

### Sphinx

No data is loaded in the kernel yet. First step is to locate the user’s market data files in the working directory so we can load them and begin EDA (timestamps, nodes/zones, DA vs RT, uplift, fuel mix).

### Sphinx - Code

**Cell ID:** 3ecb04a8-5ea8-40b9-ab92-528023585cc3

**New:**
```python
# List files in the working directory (and common data subfolders) to locate the PJM/AEP market datasets
import os
from pathlib import Path

root = Path('.')
print('CWD:', root.resolve())

# Show top-level files/folders
entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
print('\nTop-level entries (first 200):')
for p in entries[:200]:
    print(('FILE ' if p.is_file() else 'DIR  '), p.name)

# If there are common data directories, list a bit more inside them
common_dirs = [p for p in entries if p.is_dir() and p.name.lower() in {'data','datasets','dataset','input','inputs','raw','files'}]
for d in common_dirs:
    print(f"\nContents of ./{d.name} (first 200):")
    sub = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for p in sub[:200]:
        print(('FILE ' if p.is_file() else 'DIR  '), f"{d.name}/{p.name}")
```

### Sphinx - Run Cell

**Cell ID:** 3ecb04a8-5ea8-40b9-ab92-528023585cc3

### Sphinx

We’ve only located the CSV files; next we must load each dataset and perform broad EDA (schema, date ranges, missingness, keys) across all files before planning any analysis narrative.

### Sphinx - Code

**Cell ID:** 54cb0126-f8ed-450b-b511-c8c885c361f0

**New:**
```python
# Load all provided PJM/AEP CSVs and perform basic EDA (shape, columns, dtypes, date range, missingness)
import pandas as pd

file_map = {
    'da_lmp': 'data/da_hrl_lmps.csv',
    'rt_lmp': 'data/rt_hrl_lmps.csv',
    'load_metered': 'data/hrl_load_metered.csv',
    'load_forecast_7d': 'data/load_frcstd_7_day.csv',
    'uplift_zone': 'data/uplift_charges_by_zone.csv',
    'gen_by_fuel': 'data/gen_by_fuel.csv',
}

dfs = {}
for k, fp in file_map.items():
    df = pd.read_csv(fp)
    dfs[k] = df
    print(f"\n=== {k} ===")
    print('file:', fp)
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print('dtypes (first 20):')
    print(df.dtypes.head(20))
    display(df.head(5))
    # missingness quick view
    na = (df.isna().mean().sort_values(ascending=False).head(10) * 100).round(2)
    print('top missing % (up to 10 cols):')
    print(na)

```

### Sphinx - Run Cell

**Cell ID:** 54cb0126-f8ed-450b-b511-c8c885c361f0

### Sphinx

EDA shows the CSVs cover mostly Feb 2026 hourly DA/RT LMPs (PJM-RTO zone only), hourly metered load (RTO), a single 7-day forecast snapshot (RTO_COMBINED), AEP uplift charges by day/category, and a ~month of hourly generation by fuel (includes Oil). To tailor the market “update” narrative and anomaly thresholds, we need the user’s preferred zones/nodes and what they consider “notable.”


---

## User

Focus on discussing whats been going on in PJM RTO in general. Specifically for uplift charges, it's just going to be AEP

### Sphinx

Need to convert all datetime strings to proper timestamps and quantify key market signals for PJM RTO: DA vs RT LMP spikes, volatility, load and fuel mix (esp Oil), then later link uplift (AEP) by day. First concrete step: parse/standardize timestamps and create cleaned copies with a common hourly index.

### Sphinx - Code

**Cell ID:** c403fef6-4b1c-440f-9c5b-3b31cb7974ec

**New:**
```python
# Parse/standardize timestamps across datasets and create cleaned copies keyed by hourly EPT
import pandas as pd

# Copy from dfs to avoid mutating raw frames
clean = {}

# Helper: parse EPT timestamp columns (provided as strings like '2/1/2026 12:00:00 AM')
def _to_dt(s):
    return pd.to_datetime(s, errors='coerce')

# DA / RT LMPs (PJM-RTO zone)
da = dfs['da_lmp'].copy()
rt = dfs['rt_lmp'].copy()
for _df in (da, rt):
    _df['datetime_beginning_ept'] = _to_dt(_df['datetime_beginning_ept'])
    _df['datetime_beginning_utc'] = _to_dt(_df['datetime_beginning_utc'])

clean['da_lmp'] = da
clean['rt_lmp'] = rt

# Metered load (RTO)
load = dfs['load_metered'].copy()
load['datetime_beginning_ept'] = _to_dt(load['datetime_beginning_ept'])
load['datetime_beginning_utc'] = _to_dt(load['datetime_beginning_utc'])
clean['load_metered'] = load

# 7-day forecast snapshot
fc = dfs['load_forecast_7d'].copy()
for c in [
    'evaluated_at_datetime_ept','evaluated_at_datetime_utc',
    'forecast_datetime_beginning_ept','forecast_datetime_beginning_utc',
    'forecast_datetime_ending_ept','forecast_datetime_ending_utc'
]:
    fc[c] = _to_dt(fc[c])
clean['load_forecast_7d'] = fc

# Uplift charges (AEP, daily granularity)
upl = dfs['uplift_zone'].copy()
upl['datetime_beginning_utc'] = _to_dt(upl['datetime_beginning_utc'])
upl['rundate_ept'] = _to_dt(upl['rundate_ept'])
clean['uplift_zone'] = upl

# Generation by fuel (hourly)
gen = dfs['gen_by_fuel'].copy()
gen['datetime_beginning_ept'] = _to_dt(gen['datetime_beginning_ept'])
gen['datetime_beginning_utc'] = _to_dt(gen['datetime_beginning_utc'])
clean['gen_by_fuel'] = gen

# Quick sanity checks: null timestamps + date ranges
for name, _df in clean.items():
    dt_cols = [c for c in _df.columns if 'datetime' in c]
    print(f"\n{name}:")
    for c in dt_cols:
        n_null = _df[c].isna().sum()
        if n_null:
            print(f"  {c}: null={n_null}/{len(_df)}")
    # choose a primary time column for range display
    primary = 'datetime_beginning_ept' if 'datetime_beginning_ept' in _df.columns else (
        'forecast_datetime_beginning_ept' if 'forecast_datetime_beginning_ept' in _df.columns else None
    )
    if primary is not None:
        print('  range (primary):', _df[primary].min(), '→', _df[primary].max())

```

### Sphinx - Run Cell

**Cell ID:** c403fef6-4b1c-440f-9c5b-3b31cb7974ec

### Sphinx

We have broad EDA but still need deeper, quantitative exploration: align hourly indices, check duplicates/gaps, summarize DA/RT LMP distributions and spikes, fuel mix (esp Oil) time series, and uplift aggregation for AEP. Next add a compact EDA cell to compute key stats and anomaly candidates to inform the eventual narrative plan.

### Sphinx - Code

**Cell ID:** 27a8dbd2-9032-40dd-9fd5-231ac7eace80

**New:**
```python
# Compute deeper EDA summaries needed for a PJM RTO market narrative (spikes, volatility, fuel mix, uplift)
import numpy as np
import pandas as pd

# Pull cleaned frames
_da = clean['da_lmp'].copy()
_rt = clean['rt_lmp'].copy()
_load = clean['load_metered'].copy()
_gen = clean['gen_by_fuel'].copy()
_upl = clean['uplift_zone'].copy()
_fc = clean['load_forecast_7d'].copy()

# 1) Key integrity checks: duplicates/gaps on hourly series

def hourly_integrity(df, time_col, value_cols=None, name=''):
    out = {}
    s = df[[time_col] + ([] if value_cols is None else value_cols)].copy()
    s = s.sort_values(time_col)
    out['n_rows'] = len(s)
    out['n_unique_timestamps'] = s[time_col].nunique()
    out['n_duplicate_timestamps'] = len(s) - s[time_col].nunique()
    # gaps based on unique sorted times
    t = pd.Series(sorted(s[time_col].dropna().unique()))
    if len(t) >= 2:
        deltas = t.diff().dropna()
        out['min_step'] = deltas.min()
        out['max_step'] = deltas.max()
        out['n_steps_not_1h'] = int((deltas != pd.Timedelta(hours=1)).sum())
    else:
        out['min_step'] = None
        out['max_step'] = None
        out['n_steps_not_1h'] = None
    print(f"\nIntegrity: {name}")
    for k, v in out.items():
        print(f"  {k}: {v}")

hourly_integrity(_da, 'datetime_beginning_ept', ['total_lmp_da'], 'DA LMP (PJM-RTO)')
hourly_integrity(_rt, 'datetime_beginning_ept', ['total_lmp_rt'], 'RT LMP (PJM-RTO)')
hourly_integrity(_load, 'datetime_beginning_ept', ['mw'], 'Metered load (RTO)')

# 2) Build an hourly PJM-RTO panel with DA/RT/load (inner join to avoid NA artifacts)
panel = (
    _da[['datetime_beginning_ept','system_energy_price_da','total_lmp_da','congestion_price_da','marginal_loss_price_da']]
    .merge(
        _rt[['datetime_beginning_ept','system_energy_price_rt','total_lmp_rt','congestion_price_rt','marginal_loss_price_rt']],
        on='datetime_beginning_ept', how='inner'
    )
    .merge(
        _load[['datetime_beginning_ept','mw']].rename(columns={'mw':'load_mw'}),
        on='datetime_beginning_ept', how='inner'
    )
    .sort_values('datetime_beginning_ept')
)

panel['rt_minus_da_lmp'] = panel['total_lmp_rt'] - panel['total_lmp_da']
panel['abs_rt_da_spread'] = panel['rt_minus_da_lmp'].abs()

print('\nPanel shape (DA/RT/load aligned):', panel.shape)
print('Panel time range:', panel['datetime_beginning_ept'].min(), '→', panel['datetime_beginning_ept'].max())
display(panel.head(3))

# 3) Summaries for narrative: distribution + top spike hours

def summarize_price(series, label):
    q = series.quantile([0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1]).to_frame('value')
    q.index = q.index.map(lambda x: f"p{int(x*100):02d}")
    print(f"\n{label} quantiles:")
    display(q)

summarize_price(panel['total_lmp_da'], 'DA total LMP ($/MWh)')
summarize_price(panel['total_lmp_rt'], 'RT total LMP ($/MWh)')
summarize_price(panel['abs_rt_da_spread'], '|RT-DA| spread ($/MWh)')

# Top 10 spike hours by RT LMP and by spread
spike_rt = panel.nlargest(10, 'total_lmp_rt')[['datetime_beginning_ept','total_lmp_rt','total_lmp_da','rt_minus_da_lmp','load_mw']]
spike_spread = panel.nlargest(10, 'abs_rt_da_spread')[['datetime_beginning_ept','total_lmp_rt','total_lmp_da','rt_minus_da_lmp','load_mw']]

print('\nTop 10 RT LMP hours:')
display(spike_rt)
print('\nTop 10 |RT-DA| spread hours:')
display(spike_spread)

# 4) Generation fuel mix: oil presence, renewables share, and whether totals look consistent
# Aggregate to hourly totals by fuel and overall

gen_hr = _gen.groupby(['datetime_beginning_ept','fuel_type'], as_index=False)['mw'].sum()
tot_hr = gen_hr.groupby('datetime_beginning_ept', as_index=False)['mw'].sum().rename(columns={'mw':'total_gen_mw'})

# Oil-specific
oil_hr = gen_hr[gen_hr['fuel_type'].str.lower().eq('oil')].merge(tot_hr, on='datetime_beginning_ept', how='right')
oil_hr['oil_mw'] = oil_hr['mw'].fillna(0)
oil_hr['oil_share'] = oil_hr['oil_mw'] / oil_hr['total_gen_mw']

print('\nOil usage summary (hourly):')
print('  hours in gen dataset:', oil_hr['datetime_beginning_ept'].nunique())
print('  hours with oil > 0 MW:', int((oil_hr['oil_mw']>0).sum()))
print('  max oil MW:', oil_hr['oil_mw'].max())
print('  max oil share:', oil_hr['oil_share'].max())

# Show top oil hours
print('\nTop 10 oil MW hours:')
display(oil_hr.sort_values('oil_mw', ascending=False).head(10)[['datetime_beginning_ept','oil_mw','oil_share','total_gen_mw']])

# Renewables share (based on is_renewable flag)
ren_hr = _gen.groupby(['datetime_beginning_ept','is_renewable'], as_index=False)['mw'].sum()
ren_pivot = ren_hr.pivot(index='datetime_beginning_ept', columns='is_renewable', values='mw').fillna(0)
ren_pivot.columns = ['non_renew_mw','renew_mw'] if set(ren_pivot.columns)=={False,True} else [str(c) for c in ren_pivot.columns]
ren_pivot['renew_share'] = ren_pivot['renew_mw'] / (ren_pivot['renew_mw'] + ren_pivot['non_renew_mw'])

print('\nRenewables share summary:')
display(ren_pivot['renew_share'].describe(percentiles=[.01,.05,.1,.5,.9,.95,.99]).to_frame('renew_share'))

# 5) Uplift (AEP): aggregate by day/category and identify biggest days & drivers
_upl2 = _upl.copy()
# uplift timestamps are daily at midnight UTC; create date key
_upl2['date_utc'] = pd.to_datetime(_upl2['datetime_beginning_utc']).dt.date

upl_day = _upl2.groupby(['date_utc','charge_category','charge_reason','region'], as_index=False)['uplift_charges'].sum()
upl_day_total = upl_day.groupby('date_utc', as_index=False)['uplift_charges'].sum().sort_values('uplift_charges', ascending=False)

print('\nAEP uplift: top 10 days by total charges:')
display(upl_day_total.head(10))

# For the top 3 days, show breakdown
for d in upl_day_total.head(3)['date_utc'].tolist():
    print(f"\nUplift breakdown for {d}:")
    display(upl_day[upl_day['date_utc']==d].sort_values('uplift_charges', ascending=False).head(15))

```

### Sphinx - Run Cell

**Cell ID:** 27a8dbd2-9032-40dd-9fd5-231ac7eace80

### Sphinx

We have sufficient EDA; now add a markdown cell summarizing key findings/assumptions that will inform the plan (time coverage, spikes, oil usage, uplift concentrations, data limitations).

### Sphinx - Code

**Cell ID:** 400400a7-45dd-4d05-80a6-741fd080c10c

**New:**
```python
### Key findings from EDA (for plan context)

**Data sources loaded (all CSVs in `./data/`):**
- `da_hrl_lmps.csv`: Hourly **Day-Ahead** LMP components for **PJM-RTO** (pnode_id=1, type=ZONE). 481 rows (~Feb 1 00:00 EPT → Feb 21 00:00 EPT).
- `rt_hrl_lmps.csv`: Hourly **Real-Time** LMP components for **PJM-RTO** (pnode_id=1, type=ZONE). 456 rows (Feb 1 00:00 → Feb 19 23:00 EPT).
- `hrl_load_metered.csv`: Hourly metered load for **RTO**. 456 rows (Feb 1 00:00 → Feb 19 23:00 EPT).
- `load_frcstd_7_day.csv`: A single 7-day forecast snapshot for **RTO_COMBINED** evaluated at **2026-02-21 16:47 EPT** (168 hourly points covering Feb 21 00:00 → Feb 27 23:00 EPT).
- `gen_by_fuel.csv`: Hourly generation by fuel type (480 hours × ~10 fuel types = 4,800 rows). Covers Feb 1 00:00 → Feb 21 00:00 EPT.
- `uplift_charges_by_zone.csv`: **AEP zone** uplift charges by day/category/reason/region (35 rows; daily timestamps in UTC). Top days are late Jan 2026.

**Joinable “hourly panel” created:** `panel` (456 hours) = inner-joined DA LMP, RT LMP, metered load on `datetime_beginning_ept`.

**Notable market signals (PJM-RTO):**
- Price distributions are extremely right-skewed with large spike hours.
  - DA total LMP quantiles: median ~59.6; p95 ~286.9; max ~670.5 $/MWh.
  - RT total LMP quantiles: median ~58.9; p95 ~299.6; p99 ~683.3; max ~1245.6 $/MWh.
- Largest RT spike hour observed in aligned panel: **2026-02-09 07:00 EPT** with RT ~1245.6 and DA ~524.3 (spread ~721.3).
- RT-DA absolute spread is usually modest but has heavy tails:
  - median ~12.8; p95 ~112.5; max ~721.3 $/MWh.

**Fuel mix / oil dispatch (system-level):**
- Oil appears **in every hour** of the generation-by-fuel dataset (oil_mw > 0 for all 480 hours).
- Maximum oil observed: **~6,248 MW** with maximum oil share **~4.56%** of total generation.
- Oil peaks coincide with high-stress hours around early Feb and notably Feb 9.
- Renewable share is relatively low on average in this dataset:
  - mean renew share ~7.6%, range ~1.2% to ~18.0%.

**AEP uplift charges (daily):**
- Uplift totals are highly concentrated in a few days:
  - Top day: **2026-01-25** total uplift ~**$5.63M**, dominated by **Balancing Operating Reserve: Lost Opportunity Cost** (reason: Deviations, region: RTO).
  - Next: **2026-01-27 ~$4.46M**, **2026-01-26 ~$2.82M** (same dominant category).

**Data limitations / assumptions to document in final writeup:**
- LMPs and load are only available through **Feb 19 23:00 EPT** in RT/load; DA extends to Feb 21 00:00.
- Fuel mix is system-level and may not align one-to-one with PJM-RTO price node/zone mechanics; treat as contextual.
- Uplift dataset is **AEP zone only** and covers late Jan dates; it does not directly overlap the Feb hourly LMP window.
- Timestamp parsing emitted warnings due to mixed string formats; parsed values look consistent after conversion, but plan should standardize parsing format if possible.

```

### Sphinx

**What I found from the available data (EDA summary)**

*Datasets present (all CSVs in `./data/`):*

- **Day-Ahead LMPs (`da_hrl_lmps.csv`)**: 481 hourly rows for **PJM-RTO zone** (pnode_id=1, type=ZONE). Key columns: `datetime_beginning_ept`, `system_energy_price_da`, `total_lmp_da`, `congestion_price_da`, `marginal_loss_price_da`. DA covers **2026-02-01 00:00 → 2026-02-21 00:00 EPT**.
- **Real-Time LMPs (`rt_hrl_lmps.csv`)**: 456 hourly rows for **PJM-RTO zone**. Key columns analogous to DA (`*_rt`). RT covers **2026-02-01 00:00 → 2026-02-19 23:00 EPT**.
- **Metered Load (`hrl_load_metered.csv`)**: 456 hourly rows for **RTO load** with `mw` and region/zone labels. Covers **2026-02-01 00:00 → 2026-02-19 23:00 EPT**.
- **7-day Load Forecast (`load_frcstd_7_day.csv`)**: 168 hourly rows for **RTO_COMBINED** for a single evaluation timestamp (**2026-02-21 16:47 EPT**). Forecast horizon **2026-02-21 00:00 → 2026-02-27 23:00 EPT**.
- **Generation by Fuel (`gen_by_fuel.csv`)**: 4,800 rows = **480 hours × fuel types**. Columns: `fuel_type`, `mw`, `fuel_percentage_of_total`, `is_renewable`. Covers **2026-02-01 00:00 → 2026-02-21 00:00 EPT**.
- **AEP uplift (`uplift_charges_by_zone.csv`)**: 35 rows, daily granularity, `zone` = AEP, with `charge_category`, `charge_reason`, `region`, `uplift_charges`. Largest days are **late January 2026**.

*Relationships / join paths identified:*

- **Hourly alignment key**: `datetime_beginning_ept` for DA LMP, RT LMP, metered load, and generation-by-fuel.
- Constructed an **inner-joined hourly panel** on `datetime_beginning_ept` for DA+RT+Load with **456 aligned hours**.
- Uplift is **daily** (UTC midnight) and **does not overlap** the Feb hourly price window in this sample; it must be discussed as a separate AEP-specific item.

*Data quality notes:*

- Timestamp parsing produced warnings (“format not inferred”); values appear to parse consistently but the final workflow should set an explicit datetime format to avoid silent parsing issues.
- `voltage`, `equipment`, and numeric `zone` columns in LMP files are 100% missing and can be dropped for this narrative.
- Hourly integrity checks for DA/RT/load show **no gaps/duplicates** at the hourly level within each series.

*Key quantitative signals already measured (to drive narrative):*

- **Price spikes / volatility (aligned 456-hour panel through Feb 19 23:00):**
  - DA total LMP: median ~59.6; p95 ~286.9; max ~670.5 $/MWh.
  - RT total LMP: median ~58.9; p95 ~299.6; p99 ~683.3; max ~1245.6 $/MWh.
  - |RT-DA| spread: median ~12.8; p95 ~112.5; max ~721.3 $/MWh.
  - Largest RT spike hour: **2026-02-09 07:00 EPT** (RT ~1245.6, DA ~524.3, spread ~721.3) with high load (~137.7 GW).
- **Oil usage (generation-by-fuel):**
  - Oil is >0 MW in **all 480 hours**; maximum oil ~6,248 MW; max share ~4.56%.
  - Biggest oil hours are around **Feb 9 morning** and early Feb high-stress periods.
- **Renewables share:** mean ~7.6% (min ~1.2%, max ~18.0%) over the gen dataset window.
- **AEP uplift (daily):** extremely concentrated in a few days.
  - Top day **2026-01-25**: ~$5.63M, almost entirely **Balancing Operating Reserve: Lost Opportunity Cost** (reason: Deviations, region: RTO).
  - Next: **2026-01-27 ~$4.46M**, **2026-01-26 ~$2.82M** (same dominant driver).


---

## User

Also, make sure these topics are explored.  


1) How has DA vs RT been? LMP spikes, trends, differences?


2) Load (hows it been, what we expect in the coming week)


3) Generation (oil use? renewables? etc)


4) Uplift charges (deviations). Any large charges or trends to notice

### Sphinx

Explored all 6 provided CSVs, inspected schemas/dtypes, parsed timestamps, built an hourly DA/RT/load panel (456 hours), computed quantiles and top spike/spread hours, summarized oil usage and renewables share from generation-by-fuel, and aggregated AEP uplift by day with top days and dominant deviation-related categories. This supports a focused plan addressing DA vs RT behavior, load + forecast, generation mix, and uplift deviations.


---

## User

Complete the analysis based on the plan

### Sphinx

Existing notebook has EDA tables but is missing key visuals and the forward-looking load forecast summary. Next I’ll add a dedicated cell to analyze and plot historical load vs the 7-day forecast, including peak/low and basic distribution to support the narrative.

### Sphinx - Code

**Cell ID:** ea17a6b4-da14-4452-9d3f-4d29da04241b

**New:**
```python
# Analyze historical metered load vs 7-day load forecast (PJM RTO context)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Historical load (aligned with RT through Feb 19 23:00 EPT)
hist_load = clean['load_metered'].copy()
hist_load = hist_load.sort_values('datetime_beginning_ept')

# Forecast (Feb 21-27, one evaluation snapshot)
fc = clean['load_forecast_7d'].copy()
fc = fc.sort_values('forecast_datetime_beginning_ept')

# Summaries
hist_desc = hist_load['mw'].describe(percentiles=[.01,.05,.1,.5,.9,.95,.99])
fc_desc = fc['forecast_load_mw'].describe(percentiles=[.01,.05,.1,.5,.9,.95,.99])

hist_peak = hist_load.loc[hist_load['mw'].idxmax(), ['datetime_beginning_ept','mw']]
hist_low  = hist_load.loc[hist_load['mw'].idxmin(), ['datetime_beginning_ept','mw']]

fc_peak = fc.loc[fc['forecast_load_mw'].idxmax(), ['forecast_datetime_beginning_ept','forecast_load_mw']]
fc_low  = fc.loc[fc['forecast_load_mw'].idxmin(), ['forecast_datetime_beginning_ept','forecast_load_mw']]

print("Historical metered load (RTO) summary (MW):")
display(hist_desc.to_frame('mw'))
print("\nHistorical peak / low:")
display(pd.DataFrame([hist_peak, hist_low]).rename(index={0:'peak',1:'low'}))

print("\n7-day forecast load (RTO_COMBINED) summary (MW):")
display(fc_desc.to_frame('forecast_load_mw'))
print("\nForecast peak / low:")
display(pd.DataFrame([fc_peak, fc_low]).rename(index={0:'peak',1:'low'}))

# Plots
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

axes[0].plot(hist_load['datetime_beginning_ept'], hist_load['mw'], color='tab:blue', linewidth=1.5)
axes[0].set_title('Historical metered load (RTO)')
axes[0].set_xlabel('Datetime (EPT)')
axes[0].set_ylabel('MW')
axes[0].axvline(hist_peak['datetime_beginning_ept'], color='tab:red', linestyle='--', linewidth=1)
axes[0].text(hist_peak['datetime_beginning_ept'], float(hist_peak['mw']), ' peak', color='tab:red', va='bottom')

axes[1].plot(fc['forecast_datetime_beginning_ept'], fc['forecast_load_mw'], color='tab:green', linewidth=1.5)
axes[1].set_title('7-day forecast load (RTO_COMBINED) — evaluated at ' + str(fc['evaluated_at_datetime_ept'].iloc[0]))
axes[1].set_xlabel('Datetime (EPT)')
axes[1].set_ylabel('MW')
axes[1].axvline(fc_peak['forecast_datetime_beginning_ept'], color='tab:red', linestyle='--', linewidth=1)
axes[1].text(fc_peak['forecast_datetime_beginning_ept'], float(fc_peak['forecast_load_mw']), ' peak', color='tab:red', va='bottom')

for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

plt.show()

# Quick compare: typical load level shift between hist vs forecast
print("\nMedian historical load vs forecast median (MW):", float(hist_desc['50%']), "vs", float(fc_desc['50%']))
print("Forecast peak - historical peak (MW):", float(fc_peak['forecast_load_mw'] - hist_peak['mw']))

```

### Sphinx - Run Cell

**Cell ID:** ea17a6b4-da14-4452-9d3f-4d29da04241b

### Sphinx

We already have core EDA but still need to complete the plan with clear DA-vs-RT plots, event window extraction, generation context joins, and uplift trend plots, then synthesize a final narrative. Next step: add a cell that creates the key DA/RT/LMP spread/load time-series plots with readable formatting and annotations of top spike hours.

### Sphinx - Code

**Cell ID:** ca1b8cf7-7ef7-457d-8895-0c7c0a03e154

**New:**
```python
# Create key PJM RTO time-series plots: DA vs RT LMP, RT-DA spread, and load context
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Use existing 'panel' built earlier (456 aligned hours)
plot_df = panel.copy().sort_values('datetime_beginning_ept')

# Identify a few notable hours to annotate (top 3 RT LMP)
ann = plot_df.nlargest(3, 'total_lmp_rt')[['datetime_beginning_ept','total_lmp_rt','total_lmp_da','load_mw']]

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

# 1) DA vs RT total LMP
axes[0].plot(plot_df['datetime_beginning_ept'], plot_df['total_lmp_da'], label='DA total LMP', lw=1.2, color='tab:blue')
axes[0].plot(plot_df['datetime_beginning_ept'], plot_df['total_lmp_rt'], label='RT total LMP', lw=1.2, color='tab:orange')
axes[0].set_title('PJM RTO: Day-Ahead vs Real-Time total LMP')
axes[0].set_ylabel('$/MWh')
axes[0].legend(loc='upper left', ncols=2)

# annotate
for _, r in ann.iterrows():
    axes[0].scatter([r['datetime_beginning_ept']], [r['total_lmp_rt']], color='tab:red', s=25, zorder=3)
    axes[0].annotate(
        r['datetime_beginning_ept'].strftime('%m-%d %H:%M'),
        (r['datetime_beginning_ept'], r['total_lmp_rt']),
        textcoords='offset points', xytext=(5, 5), fontsize=9, color='tab:red'
    )

# 2) Spread
axes[1].plot(plot_df['datetime_beginning_ept'], plot_df['rt_minus_da_lmp'], lw=1.2, color='tab:purple')
q95 = plot_df['abs_rt_da_spread'].quantile(0.95)
q99 = plot_df['abs_rt_da_spread'].quantile(0.99)
axes[1].axhline(q99, color='tab:red', ls='--', lw=1, label=f'|spread| p99 ≈ {q99:,.0f}')
axes[1].axhline(-q99, color='tab:red', ls='--', lw=1)
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_title('RT - DA total LMP (positive = RT above DA)')
axes[1].set_ylabel('$/MWh')
axes[1].legend(loc='upper left')

# 3) Load
axes[2].plot(plot_df['datetime_beginning_ept'], plot_df['load_mw'], lw=1.2, color='tab:green')
axes[2].set_title('PJM RTO: Metered load (context)')
axes[2].set_ylabel('MW')
axes[2].set_xlabel('Datetime (EPT)')

# Formatting
locator = mdates.DayLocator(interval=2)
formatter = mdates.DateFormatter('%m-%d')
axes[2].xaxis.set_major_locator(locator)
axes[2].xaxis.set_major_formatter(formatter)
for ax in axes:
    ax.grid(True, alpha=0.25)

plt.show()

# Print the annotated spike hours for reference
print('Top 3 RT LMP hours (annotated):')
display(ann.sort_values('total_lmp_rt', ascending=False))
print(f"Spread thresholds: |spread| p95≈{q95:,.1f}, p99≈{q99:,.1f} $/MWh")

```

### Sphinx - Run Cell

**Cell ID:** ca1b8cf7-7ef7-457d-8895-0c7c0a03e154

### Sphinx

We have core DA/RT/load plots and summary tables, but we still need event-window detection (grouping extreme hours), generation context joined to those windows, an uplift trend plot, and a final written market update. Next cell will compute objective event flags and cluster them into windows with summary stats.

### Sphinx - Code

**Cell ID:** 8e14d812-7181-4c7a-8ac4-96f689098cb3

**New:**
```python
# Identify extreme-price / extreme-spread / high-load hours and cluster them into event windows
import pandas as pd
import numpy as np

panel_ev = panel.copy().sort_values('datetime_beginning_ept').reset_index(drop=True)

# Thresholds (objective, distribution-based)
thr_rt_q99 = panel_ev['total_lmp_rt'].quantile(0.99)
thr_spread_q99 = panel_ev['abs_rt_da_spread'].quantile(0.99)
thr_load_q95 = panel_ev['load_mw'].quantile(0.95)

panel_ev['flag_rt_q99'] = panel_ev['total_lmp_rt'] >= thr_rt_q99
panel_ev['flag_spread_q99'] = panel_ev['abs_rt_da_spread'] >= thr_spread_q99
panel_ev['flag_load_q95'] = panel_ev['load_mw'] >= thr_load_q95
panel_ev['flag_any'] = panel_ev[['flag_rt_q99','flag_spread_q99','flag_load_q95']].any(axis=1)

flagged = panel_ev.loc[panel_ev['flag_any'], [
    'datetime_beginning_ept','total_lmp_da','total_lmp_rt','rt_minus_da_lmp','abs_rt_da_spread','load_mw',
    'flag_rt_q99','flag_spread_q99','flag_load_q95'
]].copy()

print('Thresholds used:')
print(f"  RT LMP q99: {thr_rt_q99:,.1f} $/MWh")
print(f"  |RT-DA| spread q99: {thr_spread_q99:,.1f} $/MWh")
print(f"  Load q95: {thr_load_q95:,.0f} MW")
print('\nFlagged hours:', len(flagged), 'out of', len(panel_ev))
display(flagged.head(10))

# Cluster flagged hours into windows (gap > 1 hour starts a new window)
flagged = flagged.sort_values('datetime_beginning_ept').reset_index(drop=True)
if len(flagged) > 0:
    flagged['gap_hr'] = flagged['datetime_beginning_ept'].diff().dt.total_seconds().div(3600)
    flagged['new_window'] = flagged['gap_hr'].isna() | (flagged['gap_hr'] > 1)
    flagged['window_id'] = flagged['new_window'].cumsum()

    windows = flagged.groupby('window_id').agg(
        start=('datetime_beginning_ept','min'),
        end=('datetime_beginning_ept','max'),
        n_hours=('datetime_beginning_ept','count'),
        max_rt=('total_lmp_rt','max'),
        max_da=('total_lmp_da','max'),
        max_abs_spread=('abs_rt_da_spread','max'),
        max_load=('load_mw','max'),
        min_spread=('rt_minus_da_lmp','min'),
        max_spread=('rt_minus_da_lmp','max'),
        any_rt_q99=('flag_rt_q99','any'),
        any_spread_q99=('flag_spread_q99','any'),
        any_load_q95=('flag_load_q95','any'),
    ).reset_index(drop=True)

    # Convenience: add a label for reporting
    windows['label'] = windows.apply(
        lambda r: f"{r['start']:%Y-%m-%d %H:%M} → {r['end']:%Y-%m-%d %H:%M} ({int(r['n_hours'])}h)", axis=1
    )

    print('\nEvent windows (sorted by max RT):')
    display(windows.sort_values('max_rt', ascending=False).head(15))
else:
    windows = pd.DataFrame()
    print('No flagged hours found with current thresholds.')

```

### Sphinx - Run Cell

**Cell ID:** 8e14d812-7181-4c7a-8ac4-96f689098cb3

### Sphinx

We have DA/RT/load event windows but still need (1) generation context joined to those windows, (2) uplift daily trend plot/table focused on deviations, and (3) a final narrative summary. Next add a cell to augment the event windows with oil/renewables metrics by joining hourly generation summaries to the price/load panel and aggregating over each window.

### Sphinx - Code

**Cell ID:** ad2c68d1-5041-4780-8045-c84268c5d386

**New:**
```python
# Join price/load panel to generation context (oil + renewables) and summarize by event window
import pandas as pd

# Build hourly oil + renewables series aligned on datetime_beginning_ept
# oil_hr and ren_pivot were already computed earlier; ensure they have a column we can join on
_oil = oil_hr[['datetime_beginning_ept', 'oil_mw', 'oil_share', 'total_gen_mw']].copy()
_ren = ren_pivot.reset_index()[['datetime_beginning_ept', 'renew_share']].copy()

# Combine generation context
_gen_ctx = _oil.merge(_ren, on='datetime_beginning_ept', how='inner')

# Join with panel (limits to the overlapping time range Feb 1–Feb 19 23:00 EPT)
panel_ctx = panel.merge(_gen_ctx, on='datetime_beginning_ept', how='left')

# Sanity check join coverage
print('Panel hours:', panel_ctx['datetime_beginning_ept'].nunique())
print('Gen context missing hours in panel:', int(panel_ctx['oil_mw'].isna().sum()))

# Summarize generation context inside each previously computed window
windows_ctx = windows.copy()
rows = []
for _, w in windows_ctx.iterrows():
    m = (panel_ctx['datetime_beginning_ept'] >= w['start']) & (panel_ctx['datetime_beginning_ept'] <= w['end'])
    sub = panel_ctx.loc[m].copy()
    rows.append({
        'label': w['label'],
        'start': w['start'],
        'end': w['end'],
        'n_hours': int(w['n_hours']),
        'max_rt_lmp': float(w['max_rt']),
        'max_abs_spread': float(w['max_abs_spread']),
        'max_load_mw': float(w['max_load']),
        'max_oil_mw': float(sub['oil_mw'].max()),
        'mean_oil_mw': float(sub['oil_mw'].mean()),
        'max_oil_share': float(sub['oil_share'].max()),
        'mean_oil_share': float(sub['oil_share'].mean()),
        'mean_renew_share': float(sub['renew_share'].mean()),
    })

windows_ctx2 = pd.DataFrame(rows).sort_values('max_rt_lmp', ascending=False)
print('\nEvent windows augmented with generation context (sorted by max RT LMP):')
display(windows_ctx2)

# Baseline context for comparison (all hours in panel)
baseline = panel_ctx[['oil_mw','oil_share','renew_share']].describe(percentiles=[.05,.1,.5,.9,.95,.99])
print('\nBaseline (all panel hours) generation context summary:')
display(baseline)
```

### Sphinx - Run Cell

**Cell ID:** ad2c68d1-5041-4780-8045-c84268c5d386

### Sphinx

We have computed event windows and generation context, but we still need an uplift deviations-focused trend plot/table and then a final written market update synthesizing DA vs RT, load+forecast, generation, and uplift. Next add a cell to summarize and visualize AEP uplift charges over time, emphasizing deviations and top charge categories.

### Sphinx - Code

**Cell ID:** 284858c6-5992-4cd9-8734-abae821f9ed3

**New:**
```python
# Analyze AEP uplift charges: daily totals, deviations focus, and simple trend plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

upl_aep = clean['uplift_zone'].copy()
# Ensure we're only looking at AEP
upl_aep = upl_aep[upl_aep['zone'].astype(str).str.upper().eq('AEP')].copy()

# Date key (UTC-based; source timestamp appears daily)
upl_aep['date_utc'] = pd.to_datetime(upl_aep['datetime_beginning_utc']).dt.date

# Daily totals
upl_day_total2 = (
    upl_aep.groupby('date_utc', as_index=False)['uplift_charges']
    .sum()
    .sort_values('date_utc')
)

# Deviations vs non-deviations
upl_day_reason = (
    upl_aep.assign(is_deviations=upl_aep['charge_reason'].astype(str).str.lower().eq('deviations'))
    .groupby(['date_utc', 'is_deviations'], as_index=False)['uplift_charges']
    .sum()
)

upl_reason_pivot = (
    upl_day_reason
    .pivot(index='date_utc', columns='is_deviations', values='uplift_charges')
    .fillna(0)
    .rename(columns={False: 'non_deviations', True: 'deviations'})
    .reset_index()
)

upl_reason_pivot['total'] = upl_reason_pivot['deviations'] + upl_reason_pivot['non_deviations']
upl_reason_pivot['dev_share'] = upl_reason_pivot['deviations'] / upl_reason_pivot['total'].replace(0, pd.NA)

# Top categories overall and for deviations
upl_cat_total = (
    upl_aep.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

upl_dev = upl_aep[upl_aep['charge_reason'].astype(str).str.lower().eq('deviations')].copy()
upl_cat_dev = (
    upl_dev.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

print('AEP uplift date range (UTC):', upl_aep['date_utc'].min(), '→', upl_aep['date_utc'].max())
print('Total AEP uplift ($):', f"{upl_aep['uplift_charges'].sum():,.2f}")

print('\nTop 10 uplift days (AEP, total):')
display(upl_day_total2.sort_values('uplift_charges', ascending=False).head(10))

print('\nDaily deviations vs non-deviations (AEP):')
display(upl_reason_pivot.sort_values('total', ascending=False).head(10))

print('\nTop uplift charge categories (AEP, all reasons):')
display(upl_cat_total.head(10))

print('\nTop uplift charge categories (AEP, deviations only):')
display(upl_cat_dev.head(10))

# Plot daily totals + deviations overlay
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)

axes[0].plot(pd.to_datetime(upl_day_total2['date_utc']), upl_day_total2['uplift_charges'], color='tab:blue', lw=2)
axes[0].set_title('AEP uplift charges by day (UTC)')
axes[0].set_ylabel('Uplift charges ($)')

axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['deviations'], color='tab:red', lw=2, label='Deviations')
axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['non_deviations'], color='tab:gray', lw=1.5, label='Non-deviations')
axes[1].set_title('AEP uplift: deviations vs non-deviations (UTC)')
axes[1].set_ylabel('Uplift charges ($)')
axes[1].legend(loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.show()

# For the top 3 uplift days, print breakdown for deviations categories
for d in upl_day_total2.sort_values('uplift_charges', ascending=False).head(3)['date_utc']:
    print(f"\nBreakdown for {d} (top 15 rows by $):")
    display(
        upl_aep[upl_aep['date_utc'] == d]
        .sort_values('uplift_charges', ascending=False)
        .head(15)
        [['date_utc','charge_category','charge_reason','region','uplift_charges','rundate_ept']]
    )

```

### Sphinx - Run Cell

**Cell ID:** 284858c6-5992-4cd9-8734-abae821f9ed3

### Sphinx

Cell 9 fails because the deviations pivot only has the True column (all rows are deviations), so `non_deviations` doesn't exist. I’ll add a small, robust step to ensure both `deviations` and `non_deviations` columns exist (filled with 0) before computing totals and plotting.

### Sphinx - Code

**Cell ID:** 284858c6-5992-4cd9-8734-abae821f9ed3

**Original:**
```python
# Analyze AEP uplift charges: daily totals, deviations focus, and simple trend plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

upl_aep = clean['uplift_zone'].copy()
# Ensure we're only looking at AEP
upl_aep = upl_aep[upl_aep['zone'].astype(str).str.upper().eq('AEP')].copy()

# Date key (UTC-based; source timestamp appears daily)
upl_aep['date_utc'] = pd.to_datetime(upl_aep['datetime_beginning_utc']).dt.date

# Daily totals
upl_day_total2 = (
    upl_aep.groupby('date_utc', as_index=False)['uplift_charges']
    .sum()
    .sort_values('date_utc')
)

# Deviations vs non-deviations
upl_day_reason = (
    upl_aep.assign(is_deviations=upl_aep['charge_reason'].astype(str).str.lower().eq('deviations'))
    .groupby(['date_utc', 'is_deviations'], as_index=False)['uplift_charges']
    .sum()
)

upl_reason_pivot = (
    upl_day_reason
    .pivot(index='date_utc', columns='is_deviations', values='uplift_charges')
    .fillna(0)
    .rename(columns={False: 'non_deviations', True: 'deviations'})
    .reset_index()
)

upl_reason_pivot['total'] = upl_reason_pivot['deviations'] + upl_reason_pivot['non_deviations']
upl_reason_pivot['dev_share'] = upl_reason_pivot['deviations'] / upl_reason_pivot['total'].replace(0, pd.NA)

# Top categories overall and for deviations
upl_cat_total = (
    upl_aep.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

upl_dev = upl_aep[upl_aep['charge_reason'].astype(str).str.lower().eq('deviations')].copy()
upl_cat_dev = (
    upl_dev.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

print('AEP uplift date range (UTC):', upl_aep['date_utc'].min(), '→', upl_aep['date_utc'].max())
print('Total AEP uplift ($):', f"{upl_aep['uplift_charges'].sum():,.2f}")

print('\nTop 10 uplift days (AEP, total):')
display(upl_day_total2.sort_values('uplift_charges', ascending=False).head(10))

print('\nDaily deviations vs non-deviations (AEP):')
display(upl_reason_pivot.sort_values('total', ascending=False).head(10))

print('\nTop uplift charge categories (AEP, all reasons):')
display(upl_cat_total.head(10))

print('\nTop uplift charge categories (AEP, deviations only):')
display(upl_cat_dev.head(10))

# Plot daily totals + deviations overlay
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)

axes[0].plot(pd.to_datetime(upl_day_total2['date_utc']), upl_day_total2['uplift_charges'], color='tab:blue', lw=2)
axes[0].set_title('AEP uplift charges by day (UTC)')
axes[0].set_ylabel('Uplift charges ($)')

axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['deviations'], color='tab:red', lw=2, label='Deviations')
axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['non_deviations'], color='tab:gray', lw=1.5, label='Non-deviations')
axes[1].set_title('AEP uplift: deviations vs non-deviations (UTC)')
axes[1].set_ylabel('Uplift charges ($)')
axes[1].legend(loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.show()

# For the top 3 uplift days, print breakdown for deviations categories
for d in upl_day_total2.sort_values('uplift_charges', ascending=False).head(3)['date_utc']:
    print(f"\nBreakdown for {d} (top 15 rows by $):")
    display(
        upl_aep[upl_aep['date_utc'] == d]
        .sort_values('uplift_charges', ascending=False)
        .head(15)
        [['date_utc','charge_category','charge_reason','region','uplift_charges','rundate_ept']]
    )

```

**New:**
```python
# Analyze AEP uplift charges: daily totals, deviations focus, and simple trend plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

upl_aep = clean['uplift_zone'].copy()
# Ensure we're only looking at AEP
upl_aep = upl_aep[upl_aep['zone'].astype(str).str.upper().eq('AEP')].copy()

# Date key (UTC-based; source timestamp appears daily)
upl_aep['date_utc'] = pd.to_datetime(upl_aep['datetime_beginning_utc']).dt.date

# Daily totals
upl_day_total2 = (
    upl_aep.groupby('date_utc', as_index=False)['uplift_charges']
    .sum()
    .sort_values('date_utc')
)

# Deviations vs non-deviations
upl_day_reason = (
    upl_aep.assign(is_deviations=upl_aep['charge_reason'].astype(str).str.lower().eq('deviations'))
    .groupby(['date_utc', 'is_deviations'], as_index=False)['uplift_charges']
    .sum()
)

upl_reason_pivot = (
    upl_day_reason
    .pivot(index='date_utc', columns='is_deviations', values='uplift_charges')
    .fillna(0)
)

# Robustly ensure both columns exist (in this dataset, all rows may be deviations)
if True not in upl_reason_pivot.columns:
    upl_reason_pivot[True] = 0.0
if False not in upl_reason_pivot.columns:
    upl_reason_pivot[False] = 0.0

upl_reason_pivot = (
    upl_reason_pivot
    .rename(columns={False: 'non_deviations', True: 'deviations'})
    .reset_index()
)

upl_reason_pivot['total'] = upl_reason_pivot['deviations'] + upl_reason_pivot['non_deviations']
upl_reason_pivot['dev_share'] = upl_reason_pivot['deviations'] / upl_reason_pivot['total'].replace(0, pd.NA)

# Top categories overall and for deviations
upl_cat_total = (
    upl_aep.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

upl_dev = upl_aep[upl_aep['charge_reason'].astype(str).str.lower().eq('deviations')].copy()
upl_cat_dev = (
    upl_dev.groupby('charge_category', as_index=False)['uplift_charges']
    .sum()
    .sort_values('uplift_charges', ascending=False)
)

print('AEP uplift date range (UTC):', upl_aep['date_utc'].min(), '→', upl_aep['date_utc'].max())
print('Total AEP uplift ($):', f"{upl_aep['uplift_charges'].sum():,.2f}")

print('\nTop 10 uplift days (AEP, total):')
display(upl_day_total2.sort_values('uplift_charges', ascending=False).head(10))

print('\nDaily deviations vs non-deviations (AEP):')
display(upl_reason_pivot.sort_values('total', ascending=False).head(10))

print('\nTop uplift charge categories (AEP, all reasons):')
display(upl_cat_total.head(10))

print('\nTop uplift charge categories (AEP, deviations only):')
display(upl_cat_dev.head(10))

# Plot daily totals + deviations overlay
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)

axes[0].plot(pd.to_datetime(upl_day_total2['date_utc']), upl_day_total2['uplift_charges'], color='tab:blue', lw=2)
axes[0].set_title('AEP uplift charges by day (UTC)')
axes[0].set_ylabel('Uplift charges ($)')

axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['deviations'], color='tab:red', lw=2, label='Deviations')
axes[1].plot(pd.to_datetime(upl_reason_pivot['date_utc']), upl_reason_pivot['non_deviations'], color='tab:gray', lw=1.5, label='Non-deviations')
axes[1].set_title('AEP uplift: deviations vs non-deviations (UTC)')
axes[1].set_ylabel('Uplift charges ($)')
axes[1].legend(loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.show()

# For the top 3 uplift days, print breakdown for deviations categories
for d in upl_day_total2.sort_values('uplift_charges', ascending=False).head(3)['date_utc']:
    print(f"\nBreakdown for {d} (top 15 rows by $):")
    display(
        upl_aep[upl_aep['date_utc'] == d]
        .sort_values('uplift_charges', ascending=False)
        .head(15)
        [['date_utc','charge_category','charge_reason','region','uplift_charges','rundate_ept']]
    )

```

### Sphinx - Run Cell

**Cell ID:** 284858c6-5992-4cd9-8734-abae821f9ed3

### Sphinx

We have all required analyses computed and plotted, but we still need to synthesize them into a clear written PJM RTO market update covering DA vs RT, load + forecast, generation (oil/renewables), and AEP uplift deviations. Next add a markdown cell that references the computed tables/thresholds and summarizes the notable event windows with quantitative context.

### Sphinx - Code

**Cell ID:** eff87926-42fd-42ec-9832-00d0c8924ddd

**New:**
```python
## PJM RTO market update (from provided datasets)

### 1) DA vs RT LMPs — spikes, trends, differences
**Coverage used for aligned comparisons:** 456 hourly points from **2026-02-01 00:00 → 2026-02-19 23:00 EPT** (inner join of DA + RT + metered load).

**Typical levels (distribution):**
- **DA total LMP ($/MWh):** median **~59.6**, p95 **~286.9**, max **~670.5**.
- **RT total LMP ($/MWh):** median **~58.9**, p95 **~299.6**, p99 **~683.3**, max **~1,245.6**.
- **DA–RT divergence (|RT−DA|, $/MWh):** median **~12.8**, p95 **~112.5**, p99 **~316.8**, max **~721.3**.

**What this implies:**
- Most hours look “normal” (medians near ~$60), but **both DA and RT exhibit heavy right tails**.
- **RT has the more extreme tail** (max ~1,246) and **can substantially exceed DA** in the most stressed hours.

**Most notable spike windows (objective event windows):**
(Windows built from any of: RT≥q99, |spread|≥q99, load≥q95.)
- **2026-02-09 06:00 → 09:00 (4h)**
  - **Max RT:** ~**1,245.6**; **Max DA:** ~**524.3**
  - **Max |RT−DA| spread:** ~**721.3**
  - Also coincides with the **highest load** in the panel (see Load section).
- **2026-02-01 01:00 → 03:00 (3h)**
  - **Max RT:** ~**937.3**; **Max DA:** ~**300.0**
  - **Max |RT−DA| spread:** ~**637.3**
- **2026-02-02 06:00 → 09:00 (4h)**
  - **Max RT:** ~**713.0**; **Max DA:** ~**670.5**
  - This window features **high prices in both DA and RT**, but **less extreme divergence** than the Feb 1 / Feb 9 events.

**Note on sign of spread:** In the most extreme windows, **RT is above DA** (large positive RT−DA).

---

### 2) Load — recent conditions and coming-week expectations
**Historical metered load (RTO) summary (Feb 1 → Feb 19):**
- Median: **~109.5 GW**, p95: **~128.1 GW**, max: **~137.7 GW**.
- **Peak historical hour:** **2026-02-09 07:00 EPT** at **~137.7 GW**.
- **Lowest historical hour:** **2026-02-19 03:00 EPT** at **~82.2 GW**.

**7-day load forecast snapshot (RTO_COMBINED, evaluated 2026-02-21 16:47 EPT):**
- Forecast median: **~102.5 GW**, max: **~123.7 GW**.
- **Forecast peak:** **2026-02-24 07:00 EPT** at **~123.7 GW**.

**Interpretation (with the limited forecast context available):**
- The forecast week shows a **typical diurnal cycle** with a predicted peak around **Feb 24 morning**.
- Compared with the recent historical sample, the **forecast median and peak are lower** (forecast peak ~123.7 GW vs. observed peak ~137.7 GW), suggesting **less extreme demand stress** than the Feb 9 event—assuming conditions evolve similarly.

---

### 3) Generation mix — oil use and renewables
**System generation context comes from `gen_by_fuel.csv` (hourly), treated as contextual to PJM RTO pricing.**

**Oil dispatch:**
- In this dataset, **Oil is present every hour** (oil_mw > 0 across all 480 hours in the generation file).
- **Max oil output:** **~6,248 MW**.
- **Max oil share of total generation:** **~4.56%**.

**Renewables share (is_renewable flag):**
- Mean renewables share: **~7.6%**.
- Range: **~1.2% → ~18.0%** (hourly).

**How oil/renewables looked during the top price events (event-window augmentation):**
Baseline across all panel hours (Feb 1 → Feb 19):
- Median oil: **~351 MW** (mean ~916 MW)
- Oil share median: **~0.31%** (mean ~0.75%)

Key event windows:
- **Feb 9 06:00–09:00 (max RT ~1,246)**
  - **Max oil:** ~**6,248 MW**, **mean oil:** ~**5,059 MW**
  - **Max oil share:** ~**4.56%**, **mean oil share:** ~**3.73%**
  - **Mean renew share:** ~**6.75%**
  - Oil is **far above baseline** (well above the panel p99 for oil MW).
- **Feb 1 01:00–03:00 (max RT ~937)**
  - **Mean oil:** ~**3,318 MW**, **max oil share:** ~**2.85%**
  - **Mean renew share:** ~**2.73%** (notably low vs baseline median ~6.9%).

**Interpretation (non-causal):**
- The largest price spike window (Feb 9 morning) coincides with **(i) the highest load in the sample and (ii) unusually high oil dispatch**.
- Early Feb 1 spike window coincides with **elevated oil** and **very low renewables share** in this dataset.

---

### 4) AEP uplift charges (deviations) — large charges and patterns
**AEP uplift data coverage:** **2026-01-02 → 2026-01-31 (UTC date)**.
- **Total uplift (AEP):** **~$14.81M** over the sample.

**Concentration in a few days:**
- **2026-01-25:** **~$5.63M**
- **2026-01-27:** **~$4.46M**
- **2026-01-26:** **~$2.82M**

**Deviations dominance:**
- In the provided dataset, **100% of the uplift charges are labeled `charge_reason = Deviations`**.

**Dominant category:**
- **Balancing Operating Reserve: Lost Opportunity Cost** is the primary driver:
  - ~**$13.66M** of **$14.81M** total.
  - The largest days are essentially single-category, deviations-driven charges (region shows as RTO on those peak days).

**Interpretation / what to watch:**
- The uplift series is **highly episodic**—most days are small, but a few days dominate totals.
- Since all charges are deviations in this extract, if you later pull a broader uplift dataset (more zones or longer horizon), it’s worth checking whether this “all deviations” pattern persists or is a sampling artifact.

---

### Data limitations to keep in mind
- **RT LMP and metered load stop at 2026-02-19 23:00 EPT** in the provided files; DA extends to Feb 21 00:00.
- **Uplift (AEP) is late Jan and does not overlap the Feb hourly LMP window** in this sample, so we cannot directly align uplift days to the Feb LMP spikes using only these files.
- Generation-by-fuel is **system-level context**, not a nodal/zonal dispatch explanation; treat it as contextual rather than causal.

```

### Sphinx

Delivered a complete PJM RTO market update using the provided datasets: aligned DA/RT/load panel, quantified distributions and spike windows, compared historical load to 7-day forecast, summarized generation context with oil and renewables (including event-window augmentation), and analyzed AEP uplift deviations with daily trends and top-day/category breakdowns. Added readable supporting plots and a final narrative markdown section.


---

## User

Can you convert this to narrtive form to be presented to clients?

### Sphinx

Convert the existing bullet-based market update into a client-ready narrative with clear sections, quantified highlights, and explicit caveats. Add as a markdown cell without rerunning analysis.

### Sphinx - Code

**Cell ID:** d62b896d-6cee-46a5-b24e-69f77daa936c

**New:**
```python
## Client-ready narrative: PJM RTO market update (from provided datasets)

### Executive summary
Over the **Feb 1–Feb 19** period available for aligned analysis (456 hourly observations), **PJM RTO pricing was highly bifurcated**: most hours cleared around **~$60/MWh**, but the distribution had a **heavy right tail** with **episodic, very large price spikes**—particularly in **Real-Time (RT)**. The most stressed interval occurred on **Feb 9 in the morning hours**, when **RT prices exceeded $1,200/MWh** and the **RT–Day-Ahead (DA) spread widened dramatically**, coinciding with the **highest load** observed in the sample and **unusually high oil dispatch** in the system fuel mix data.

Separately, the **AEP uplift** sample (daily, **Jan 2–Jan 31**) shows **highly episodic deviations-related uplift**, dominated by **Balancing Operating Reserve: Lost Opportunity Cost**, with a handful of late-January days driving the majority of total uplift dollars.

---

### 1) Day-Ahead vs Real-Time LMPs: spikes, trends, and divergence
**Data used:** An inner-joined hourly panel of **PJM RTO DA LMP**, **PJM RTO RT LMP**, and **RTO metered load**, covering **2026-02-01 00:00 → 2026-02-19 23:00 EPT**.

**Typical price levels were moderate, but tail risk was extreme.** Across the sample, the median DA and RT total LMPs were both near **$60/MWh**. However, both markets exhibited pronounced right skew:
- **DA total LMP:** median **~$59.6/MWh**, p95 **~$286.9/MWh**, max **~$670.5/MWh**
- **RT total LMP:** median **~$58.9/MWh**, p95 **~$299.6/MWh**, p99 **~$683.3/MWh**, max **~$1,245.6/MWh**

**RT volatility was the defining feature.** The RT market showed the more extreme tail (max near **$1,246/MWh**), and the most stressed hours were characterized by RT clearing far above DA.

**DA–RT divergence was usually modest, but with rare, very large exceptions.** The absolute RT–DA spread had a median of **~$12.8/MWh**, but widened materially in the tail (p95 **~$112.5/MWh**, p99 **~$316.8/MWh**, max **~$721.3/MWh**).

**Most notable event windows (objective thresholds):** We flagged hours meeting any of the following conditions: **RT LMP ≥ 99th percentile**, **|RT–DA| spread ≥ 99th percentile**, or **load ≥ 95th percentile**, then grouped consecutive hours into windows. The most important windows were:

- **Feb 9, 06:00–09:00 EPT (4 hours):**
  - **Max RT:** ~**$1,245.6/MWh**
  - **Max DA:** ~**$524.3/MWh**
  - **Max |RT–DA| spread:** ~**$721.3/MWh**
  - This window represents the clearest “stress event” in the sample, combining extreme RT pricing, unusually wide DA–RT divergence, and peak load.

- **Feb 1, 01:00–03:00 EPT (3 hours):**
  - **Max RT:** ~**$937.3/MWh**
  - **Max DA:** ~**$300.0/MWh**
  - **Max |RT–DA| spread:** ~**$637.3/MWh**

- **Feb 2, 06:00–09:00 EPT (4 hours):**
  - **Max RT:** ~**$713.0/MWh**
  - **Max DA:** ~**$670.5/MWh**
  - Here, prices were high in both DA and RT, but **divergence was smaller** than the Feb 1 and Feb 9 events.

**Client takeaway:** The period is best described as **“normal medians with extreme, episodic spikes”**, where **RT risk** dominates and **DA–RT basis can widen sharply** in the most stressed hours.

---

### 2) Load: how it’s been, and what we expect in the coming week
**Recent load conditions (Feb 1–Feb 19, metered):** Load ranged from a low around **~82.2 GW** to a peak around **~137.7 GW**, with a median of **~109.5 GW**.
- **Median:** ~**109.5 GW**
- **p95:** ~**128.1 GW**
- **Max:** ~**137.7 GW** (at **2026-02-09 07:00 EPT**)

This is notable because the **system peak load hour coincided with the highest RT price hour** in the aligned panel.

**Coming-week expectations (7-day forecast snapshot):** The forecast provided is a single evaluation snapshot for **RTO_COMBINED**, evaluated at **2026-02-21 16:47 EPT**, covering **Feb 21–Feb 27**.
- **Forecast median:** ~**102.5 GW**
- **Forecast peak:** ~**123.7 GW** at **2026-02-24 07:00 EPT**

**Interpretation:** Based on this snapshot alone, the next week’s forecasted load profile exhibits a normal diurnal pattern and is **materially below** the recent observed peak (forecast peak **~123.7 GW** vs. observed peak **~137.7 GW**). All else equal, this points toward **less demand-driven stress than the Feb 9 event**, although weather, outages, and operational conditions can still drive volatility.

---

### 3) Generation: oil use, renewables share, and stress-period context
**Data context:** Fuel mix comes from the system-level `gen_by_fuel.csv` series (hourly). It should be treated as **market context** rather than a nodal causal explanation for PJM RTO LMP outcomes.

**Oil usage was persistent in the sample and surged during stress periods.** Oil was present at a non-zero level in **every hour** of the generation dataset. While typical oil levels were relatively low, oil output increased sharply in stressed windows.
- **Max oil output:** ~**6,248 MW**
- **Max oil share:** ~**4.56%** of total generation

Across all hours overlapping the DA/RT/load panel, the baseline (all hours) oil levels were much lower:
- **Median oil:** ~**351 MW** (mean ~**916 MW**)
- **Median oil share:** ~**0.31%** (mean ~**0.75%**)

**During the key Feb 9 price spike window (06:00–09:00 EPT):**
- **Max oil:** ~**6,248 MW**; **mean oil:** ~**5,059 MW**
- **Max oil share:** ~**4.56%**; **mean oil share:** ~**3.73%**

This indicates that the largest price spike window coincided with **oil dispatch far above typical levels**, consistent with a system operating closer to the margin and calling on higher-cost resources.

**Renewables share was relatively low on average, with variability.** Using the dataset’s `is_renewable` flag, renewables averaged **~7–8%** of hourly generation, with a range from roughly **~1% to ~18%**. Notably, the **Feb 1 early-morning spike window** coincided with **very low renewable share** (mean renew share ~**2.7%** in that event window), while the **Feb 9 spike window** had a renew share closer to the baseline.

**Client takeaway:** The fuel mix data supports a clear qualitative message: **stress hours coincide with markedly higher oil output** (and, in some cases, low renewables share), reinforcing that the system was operating under conditions where **higher-cost resources were more likely to set price**.

---

### 4) AEP uplift charges (deviations): large charges and patterns
**Data used:** AEP uplift sample covering **2026-01-02 → 2026-01-31** (UTC date). Total uplift in the extract is approximately **$14.81M**.

**Uplift was highly episodic, concentrated in a few days.** The top three uplift days dominate the period:
- **2026-01-25:** ~**$5.63M**
- **2026-01-27:** ~**$4.46M**
- **2026-01-26:** ~**$2.82M**

**Deviations were the entire story in this sample.** In the provided extract, **100% of uplift charges** are labeled with `charge_reason = Deviations`.

**Dominant charge category:**
- **Balancing Operating Reserve: Lost Opportunity Cost** accounts for approximately **$13.66M** of the **$14.81M** total in this sample.

**Client takeaway:** For AEP, the uplift sample indicates a **few extreme deviation-related days** can drive the majority of uplift exposure, with Lost Opportunity Cost being the principal driver in this extract.

---

### Data limitations and how to interpret these results
- **Coverage mismatch:** RT LMP and metered load end at **2026-02-19 23:00 EPT** in the provided files; DA and generation extend later.
- **Uplift timing mismatch:** AEP uplift data is **late January** and does not overlap the **February hourly LMP** window in this sample; we therefore discuss uplift as a separate (AEP-only) observation.
- **Fuel mix is system-level context:** Generation-by-fuel is useful for understanding broader operating conditions (e.g., oil on the margin) but should not be interpreted as a direct nodal driver.

If you’d like, I can package the above into a one-page PDF-style brief (with the charts embedded) and a separate appendix with the event tables.
```

### Sphinx

Added a client-ready narrative markdown section summarizing DA vs RT behavior, load + forecast, generation (oil/renewables) context, and AEP uplift deviations, with quantified highlights and clear limitations, based on previously computed tables/plots.


---

## User

yes, package the above into a one-page PDF-style brief (with the charts embedded) and a separate appendix with the event tables.

### Sphinx

We need to generate a one-page PDF-style brief with embedded charts plus a separate appendix of event tables. Best is to export existing figures (recreate deterministically in one cell), then build HTML via Jinja2 and convert to PDF using WeasyPrint if available; otherwise use nbconvert to PDF via HTML+PDF. First step: create an output directory and regenerate/save the key charts as image files for embedding.

### Sphinx - Code

**Cell ID:** e8e10b70-c680-4a0a-af56-ff14edc087db

**New:**
```python
# Create output folder and (re)generate key charts as image files for embedding in a 1-page brief
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid")

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# 1) PJM RTO: DA vs RT LMP, spread, load (3-panel)
plot_df = panel.copy().sort_values('datetime_beginning_ept')
ann = plot_df.nlargest(3, 'total_lmp_rt')[['datetime_beginning_ept','total_lmp_rt','total_lmp_da','load_mw']]

fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)

axes[0].plot(plot_df['datetime_beginning_ept'], plot_df['total_lmp_da'], label='DA total LMP', lw=1.2, color='tab:blue')
axes[0].plot(plot_df['datetime_beginning_ept'], plot_df['total_lmp_rt'], label='RT total LMP', lw=1.2, color='tab:orange')
axes[0].set_title('PJM RTO: Day-Ahead vs Real-Time total LMP')
axes[0].set_ylabel('$/MWh')
axes[0].legend(loc='upper left', ncols=2, frameon=True)
for _, r in ann.iterrows():
    axes[0].scatter([r['datetime_beginning_ept']], [r['total_lmp_rt']], color='tab:red', s=18, zorder=3)
    axes[0].annotate(r['datetime_beginning_ept'].strftime('%m-%d %H:%M'),
                     (r['datetime_beginning_ept'], r['total_lmp_rt']),
                     textcoords='offset points', xytext=(4, 4), fontsize=8, color='tab:red')

axes[1].plot(plot_df['datetime_beginning_ept'], plot_df['rt_minus_da_lmp'], lw=1.2, color='tab:purple')
q99 = plot_df['abs_rt_da_spread'].quantile(0.99)
axes[1].axhline(q99, color='tab:red', ls='--', lw=1, label=f'|spread| p99 ≈ {q99:,.0f}')
axes[1].axhline(-q99, color='tab:red', ls='--', lw=1)
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_title('RT − DA total LMP (positive = RT above DA)')
axes[1].set_ylabel('$/MWh')
axes[1].legend(loc='upper left', frameon=True)

axes[2].plot(plot_df['datetime_beginning_ept'], plot_df['load_mw'], lw=1.2, color='tab:green')
axes[2].set_title('PJM RTO: Metered load (context)')
axes[2].set_ylabel('MW')
axes[2].set_xlabel('Datetime (EPT)')

locator = mdates.DayLocator(interval=2)
formatter = mdates.DateFormatter('%m-%d')
axes[2].xaxis.set_major_locator(locator)
axes[2].xaxis.set_major_formatter(formatter)
for ax in axes:
    ax.grid(True, alpha=0.25)

fig_path_1 = out_dir / "pjm_da_rt_spread_load.png"
fig.savefig(fig_path_1, dpi=200)
plt.close(fig)

# 2) Load: historical vs 7-day forecast (2-panel)
hist_load = clean['load_metered'].copy().sort_values('datetime_beginning_ept')
fc = clean['load_forecast_7d'].copy().sort_values('forecast_datetime_beginning_ept')

hist_peak = hist_load.loc[hist_load['mw'].idxmax(), ['datetime_beginning_ept','mw']]
fc_peak = fc.loc[fc['forecast_load_mw'].idxmax(), ['forecast_datetime_beginning_ept','forecast_load_mw']]

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
axes[0].plot(hist_load['datetime_beginning_ept'], hist_load['mw'], color='tab:blue', lw=1.4)
axes[0].set_title('Historical metered load (RTO)')
axes[0].set_ylabel('MW')
axes[0].axvline(hist_peak['datetime_beginning_ept'], color='tab:red', ls='--', lw=1)
axes[0].annotate('peak', (hist_peak['datetime_beginning_ept'], float(hist_peak['mw'])),
                 textcoords='offset points', xytext=(4, 4), fontsize=8, color='tab:red')

axes[1].plot(fc['forecast_datetime_beginning_ept'], fc['forecast_load_mw'], color='tab:green', lw=1.4)
axes[1].set_title('7-day forecast load (RTO_COMBINED) — evaluated at ' + str(fc['evaluated_at_datetime_ept'].iloc[0]))
axes[1].set_ylabel('MW')
axes[1].set_xlabel('Datetime (EPT)')
axes[1].axvline(fc_peak['forecast_datetime_beginning_ept'], color='tab:red', ls='--', lw=1)
axes[1].annotate('peak', (fc_peak['forecast_datetime_beginning_ept'], float(fc_peak['forecast_load_mw'])),
                 textcoords='offset points', xytext=(4, 4), fontsize=8, color='tab:red')

for ax in axes:
    ax.grid(True, alpha=0.25)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

fig_path_2 = out_dir / "load_hist_vs_forecast.png"
fig.savefig(fig_path_2, dpi=200)
plt.close(fig)

# 3) AEP uplift daily totals plot
upl_aep = clean['uplift_zone'].copy()
upl_aep = upl_aep[upl_aep['zone'].astype(str).str.upper().eq('AEP')].copy()
upl_aep['date_utc'] = upl_aep['datetime_beginning_utc'].dt.date
upl_day_total2 = upl_aep.groupby('date_utc', as_index=False)['uplift_charges'].sum().sort_values('date_utc')

fig, ax = plt.subplots(1, 1, figsize=(11, 3.6), constrained_layout=True)
ax.plot(pd.to_datetime(upl_day_total2['date_utc']), upl_day_total2['uplift_charges'], color='tab:blue', lw=2)
ax.set_title('AEP uplift charges by day (UTC)')
ax.set_ylabel('Uplift charges ($)')
ax.grid(True, alpha=0.25)

fig_path_3 = out_dir / "aep_uplift_daily.png"
fig.savefig(fig_path_3, dpi=200)
plt.close(fig)

print('Saved figures:')
print('-', fig_path_1)
print('-', fig_path_2)
print('-', fig_path_3)

```

### Sphinx - Run Cell

**Cell ID:** e8e10b70-c680-4a0a-af56-ff14edc087db

### Sphinx

We have charts saved to outputs/, but still need to generate (1) a one-page PDF-style brief embedding those images and key bullets, and (2) a separate appendix with event tables. Next step is to create an HTML brief and attempt PDF rendering (WeasyPrint if available; otherwise fall back to saving HTML).

### Sphinx - Code

**Cell ID:** 66f90e3f-846a-4ad3-b309-9b603d9144cd

**New:**
```python
# Create a 1-page PDF-style brief (HTML->PDF if possible) and an appendix (HTML + CSVs)
from pathlib import Path
import pandas as pd
import textwrap

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# --- Inputs already produced earlier in the notebook ---
# Figures saved in Cell 12
fig1 = out_dir / "pjm_da_rt_spread_load.png"
fig2 = out_dir / "load_hist_vs_forecast.png"
fig3 = out_dir / "aep_uplift_daily.png"

# Tables already computed
# windows: event windows (start/end/max_rt/max_abs_spread/max_load + flags)
# windows_ctx2: event windows augmented with oil/renewables context
# spike_rt / spike_spread: top 10 hours

# --- Save appendix tables as CSVs for client portability ---
(windows.sort_values('max_rt', ascending=False)
 .to_csv(out_dir / "appendix_event_windows.csv", index=False))
(windows_ctx2.sort_values('max_rt_lmp', ascending=False)
 .to_csv(out_dir / "appendix_event_windows_with_gen.csv", index=False))
(spike_rt.to_csv(out_dir / "appendix_top10_rt_spike_hours.csv", index=False))
(spike_spread.to_csv(out_dir / "appendix_top10_spread_hours.csv", index=False))

# --- Build compact HTML (brief) ---
# Keep it truly "one page" by using compact typography and 3 embedded charts.
# Use relative paths; we will also write the HTML into outputs/.

def _fmt_money(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

# Pull key headline numbers from existing computed stats in the notebook (hard-coded from markdown is OK,
# but we recompute from panel to ensure consistency).
q = panel[['total_lmp_da','total_lmp_rt','abs_rt_da_spread']].quantile([0.5,0.95,0.99,1.0])
load_q = panel['load_mw'].quantile([0.5,0.95,1.0])

# Biggest window by RT
w_top = windows.sort_values('max_rt', ascending=False).iloc[0]

brief_html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>PJM RTO Market Brief</title>
  <style>
    @page {{ size: letter; margin: 0.55in; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; }}
    h1 {{ font-size: 16pt; margin: 0 0 6px 0; }}
    h2 {{ font-size: 11.5pt; margin: 10px 0 4px 0; }}
    p, li {{ font-size: 9.5pt; line-height: 1.25; margin: 0 0 6px 0; }}
    .meta {{ font-size: 9pt; color:#333; margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .card {{ border: 1px solid #ddd; padding: 8px; border-radius: 6px; }}
    .kpi {{ font-size: 9.5pt; }}
    .kpi b {{ font-weight: 700; }}
    img {{ width: 100%; height: auto; border: 1px solid #eee; }}
    .small {{ font-size: 8.5pt; color:#444; }}
    .foot {{ font-size: 8.5pt; color:#444; margin-top: 6px; }}
  </style>
</head>
<body>

<h1>PJM RTO Market Brief (from provided datasets)</h1>
<div class='meta'>Coverage (aligned DA/RT/Load panel): <b>{panel['datetime_beginning_ept'].min():%Y-%m-%d %H:%M}</b> → <b>{panel['datetime_beginning_ept'].max():%Y-%m-%d %H:%M}</b> (EPT), {len(panel):,} hours.</div>

<div class='grid'>
  <div class='card'>
    <h2>Pricing: DA vs RT</h2>
    <ul>
      <li class='kpi'>Median LMPs: <b>{q.loc[0.5,'total_lmp_da']:.1f}</b> DA vs <b>{q.loc[0.5,'total_lmp_rt']:.1f}</b> RT ($/MWh)</li>
      <li class='kpi'>Tail risk: RT p99 <b>{q.loc[0.99,'total_lmp_rt']:.1f}</b>, max <b>{q.loc[1.0,'total_lmp_rt']:.1f}</b> ($/MWh)</li>
      <li class='kpi'>|RT−DA| spread: median <b>{q.loc[0.5,'abs_rt_da_spread']:.1f}</b>, p99 <b>{q.loc[0.99,'abs_rt_da_spread']:.1f}</b>, max <b>{q.loc[1.0,'abs_rt_da_spread']:.1f}</b> ($/MWh)</li>
      <li class='kpi'>Largest event window (by RT): <b>{w_top['start']:%Y-%m-%d %H:%M}</b> → <b>{w_top['end']:%H:%M}</b> (EPT), max RT <b>{w_top['max_rt']:.0f}</b>, max spread <b>{w_top['max_abs_spread']:.0f}</b></li>
    </ul>
  </div>

  <div class='card'>
    <h2>Load: recent vs 7-day forecast snapshot</h2>
    <ul>
      <li class='kpi'>Recent metered load: median <b>{load_q.loc[0.5]:,.0f}</b> MW; p95 <b>{load_q.loc[0.95]:,.0f}</b> MW; max <b>{load_q.loc[1.0]:,.0f}</b> MW</li>
      <li class='kpi'>Forecast snapshot (RTO_COMBINED): Feb 21–27; peak ~<b>{int(fc['forecast_load_mw'].max()):,}</b> MW</li>
    </ul>
    <p class='small'>Note: forecast is a single evaluation (no backtest possible with provided files).</p>
  </div>
</div>

<h2>Key Charts</h2>
<div class='grid'>
  <div class='card'>
    <p><b>DA vs RT LMP, spread, and load</b></p>
    <img src='{fig1.name}' />
  </div>
  <div class='card'>
    <p><b>Load history vs 7-day forecast</b></p>
    <img src='{fig2.name}' />
  </div>
</div>

<div class='card' style='margin-top:10px;'>
  <p><b>AEP uplift (daily)</b></p>
  <img src='{fig3.name}' />
  <p class='small'>In this extract, uplift is 100% labeled <i>Deviations</i> and is dominated by Balancing Operating Reserve: Lost Opportunity Cost.</p>
</div>

<p class='foot'><b>Data limitations:</b> RT LMP and metered load end 2026-02-19 23:00 EPT in provided files; DA and generation extend later. AEP uplift is late Jan and does not overlap the Feb hourly LMP window in this sample; treat uplift as a separate observation. Generation-by-fuel is system-level context (not nodal causal attribution).</p>

</body>
</html>"""

brief_html_path = out_dir / "pjm_rto_one_page_brief.html"
brief_html_path.write_text(brief_html, encoding="utf-8")

# --- Build appendix HTML with embedded tables (readable, multi-page ok) ---

def df_to_html(df: pd.DataFrame, max_rows=50):
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_html(index=False, border=0, classes="tbl")

appendix_html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>PJM RTO Appendix</title>
  <style>
    @page {{ size: letter; margin: 0.6in; }}
    body {{ font-family: Arial, sans-serif; color:#111; }}
    h1 {{ font-size: 16pt; margin: 0 0 10px 0; }}
    h2 {{ font-size: 12pt; margin: 14px 0 6px 0; }}
    p {{ font-size: 9.5pt; margin: 0 0 8px 0; }}
    table.tbl {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; }}
    .tbl th, .tbl td {{ border: 1px solid #ddd; padding: 4px 6px; }}
    .tbl th {{ background: #f4f6f8; text-align: left; }}
    .note {{ font-size: 9pt; color:#444; }}
  </style>
</head>
<body>
<h1>Appendix: Event Tables (from provided datasets)</h1>
<p class='note'>CSV exports are also saved alongside this appendix for reuse in slide decks or spreadsheets.</p>

<h2>Event windows (flags: RT q99, |spread| q99, load q95)</h2>
{df_to_html(windows.sort_values('max_rt', ascending=False), max_rows=50)}

<h2>Event windows with generation context (oil + renewables)</h2>
{df_to_html(windows_ctx2.sort_values('max_rt_lmp', ascending=False), max_rows=50)}

<h2>Top 10 RT LMP spike hours</h2>
{df_to_html(spike_rt, max_rows=20)}

<h2>Top 10 |RT−DA| spread hours</h2>
{df_to_html(spike_spread, max_rows=20)}

</body>
</html>"""

appendix_html_path = out_dir / "pjm_rto_appendix.html"
appendix_html_path.write_text(appendix_html, encoding="utf-8")

# --- Attempt HTML -> PDF conversion ---
# Prefer WeasyPrint if installed; otherwise leave HTML as deliverable.
brief_pdf_path = out_dir / "pjm_rto_one_page_brief.pdf"
appendix_pdf_path = out_dir / "pjm_rto_appendix.pdf"

pdf_status = {"weasyprint": False, "brief_pdf": False, "appendix_pdf": False}
try:
    import weasyprint  # type: ignore
    pdf_status["weasyprint"] = True

    # Ensure images resolve by setting base_url to out_dir
    weasyprint.HTML(filename=str(brief_html_path), base_url=str(out_dir.resolve())).write_pdf(str(brief_pdf_path))
    weasyprint.HTML(filename=str(appendix_html_path), base_url=str(out_dir.resolve())).write_pdf(str(appendix_pdf_path))
    pdf_status["brief_pdf"] = True
    pdf_status["appendix_pdf"] = True
except Exception as e:
    pdf_status["error"] = str(e)

print("Wrote:")
print("-", brief_html_path)
print("-", appendix_html_path)
print("-", out_dir / "appendix_event_windows.csv")
print("-", out_dir / "appendix_event_windows_with_gen.csv")
print("-", out_dir / "appendix_top10_rt_spike_hours.csv")
print("-", out_dir / "appendix_top10_spread_hours.csv")
print("\nPDF conversion status:", pdf_status)
if pdf_status.get("brief_pdf"):
    print("-", brief_pdf_path)
if pdf_status.get("appendix_pdf"):
    print("-", appendix_pdf_path)

```

### Sphinx - Run Cell

**Cell ID:** 66f90e3f-846a-4ad3-b309-9b603d9144cd

### Sphinx

Generated a one-page brief and appendix as HTML with embedded charts; exported key tables to CSV; attempted PDF conversion but WeasyPrint isn’t installed, so deliverables are the HTML files plus CSV appendix tables and PNG figures in outputs/.


---

## User

give another analysis, but include market opportunities/potential actions to take 

### Sphinx

We already have descriptive market analysis; now we need to extend it with actionable market opportunities. I’ll add a markdown cell that translates observed DA/RT volatility, spreads, load/oil stress signals, and uplift episodicity into potential hedging, bidding, and operational actions, while clearly labeling them as non-advice and constrained by dataset limits.

### Sphinx - Code

**Cell ID:** 0bf912e6-0497-4bf4-b04c-c05cc677bb9c

**New:**
```python
## Add-on: Market opportunities / potential actions (based on observed PJM RTO conditions)

**Important (scope & caveats):** The opportunities below are *hypothesis-driven actions* derived from the patterns in the provided sample (Feb 1–Feb 19 hourly DA/RT/load; Feb 1–Feb 21 system fuel mix; Jan 2–Jan 31 AEP uplift). They are **not investment advice** and should be validated with additional drivers not included here (weather/temperature, outages, gas prices, transmission constraints, reserve scarcity pricing, unit commitment, load forecasting error, and node/zone basis).

### 1) DA vs RT volatility and large positive RT–DA spreads → strengthen real-time exposure management
**What the data shows:**
- Median prices are ~\$60/MWh, but **tail events are extreme**, especially in RT (max ~\$1,246/MWh).
- The largest stress window (**Feb 9 06:00–09:00 EPT**) had **RT far above DA** with max spread ~\$721/MWh.

**Opportunities / actions:**
- **Reduce unhedged RT purchase exposure during “stress hours.”**
  - If you are structurally short in RT (e.g., load-serving with forecast error), consider **adding DA coverage** for the high-risk morning peak hours when the system is tight.
- **Formalize a DA-to-RT “tail hedge” trigger policy** using objective thresholds that resemble what we observed:
  - Example trigger signals: prior-day DA > p95, load forecast above a threshold, tight reserves, or known constraints.
- **Use RT/DA divergence as an operational KPI.** Large positive RT–DA indicates exposure to forecast/dispatch uncertainty; track and attribute these hours to drivers (forecast error, outages, congestion).

### 2) “Stress window playbook” around high load + high oil dispatch → pre-emptive hedging and operational readiness
**What the data shows:**
- The largest RT spike coincided with the **highest load** in sample (~137.7 GW) and **very elevated oil dispatch** (mean ~5.1 GW; max ~6.25 GW vs baseline median ~0.35 GW).

**Opportunities / actions:**
- **Treat oil dispatch as a *system tightness proxy*.** When oil ramps materially above baseline, it can signal expensive marginal resources and scarcity risk.
  - Build an internal alert: if oil MW or oil share crosses a percentile threshold (e.g., > p95 or > p99), escalate hedging/ops review.
- **Demand response / load flexibility monetization:**
  - If you have flexible load, target **Feb 9–type morning ramps** where marginal value of curtailment can be very high.
- **For generators / tolling / dispatchable assets:**
  - During high-load mornings, consider **conservative RT offer strategies** (within market rules) and ensure operational readiness (fuel assurance, start capability) for scarcity-type events.

### 3) Basis between DA and RT suggests value in forecast improvement and intraday re-optimization
**What the data shows:**
- Typical |RT–DA| is modest (median ~\$12.8/MWh) but has a **fat tail** (p99 ~\$317; max ~\$721).

**Opportunities / actions:**
- **Forecast-error monetization/mitigation:**
  - If you manage a portfolio (load/DERs/renewables), invest in short-horizon improvements (intraday weather nowcasts, ramp prediction) to reduce worst-case RT exposure.
- **Intraday position management:**
  - Operationally, treat unusually high DA price levels as a prompt to review RT risk posture (e.g., tighten risk limits, adjust schedules).

### 4) Load outlook (single 7-day snapshot) implies lower peak than recent stress → opportunity to “re-risk” selectively
**What the data shows:**
- Forecast peak (~123.7 GW) is **below** the recent observed peak (~137.7 GW), suggesting *potentially* less demand-driven stress than Feb 9.

**Opportunities / actions:**
- **Selective reduction of expensive hedges** if corroborated by weather/outage data:
  - If multiple independent signals point to looser system conditions, there may be room to **shift from defensive RT hedging** back toward more normal positioning.
- **But keep tail protection:** Given observed fat tails, maintain at least a minimal “tail hedge” strategy for morning/evening ramps.

### 5) AEP uplift (deviations) is highly episodic and LOC-dominated → tighten deviation controls and uplift budgeting
**What the data shows:**
- AEP uplift totals are concentrated in a few late-Jan days (\$5.63M, \$4.46M, \$2.82M).
- In this extract, **100% is labeled Deviations** and dominated by **Balancing Operating Reserve: Lost Opportunity Cost**.

**Opportunities / actions:**
- **Deviation reduction program:**
  - Review drivers of deviations (schedule accuracy, load forecast error, intermittent generation variance, telemetry issues).
  - Implement tighter controls/alerts on deviation magnitude during high-risk conditions.
- **Uplift “shock budgeting” and governance:**
  - Because costs are spiky, use **stress-case uplift adders** in budgeting/settlement forecasting.
  - Establish thresholds that trigger executive review (e.g., daily uplift above \$X or percentile-based).
- **Validate whether the “all deviations” pattern is sampling-driven:**
  - Pull a broader uplift dataset (more zones or longer period) to confirm if non-deviation charges appear and whether LOC remains dominant.

### 6) Quick, concrete next steps to operationalize (low lift)
1. **Create an “Event Monitor” dashboard** updated daily:
   - RT LMP percentile vs trailing window
   - |RT–DA| spread percentile
   - Load percentile vs trailing window
   - Oil MW and oil share percentiles
2. **Define 3–5 action triggers** (e.g., if RT p99 risk is elevated, increase DA cover for morning hours; if oil share > p95, tighten RT risk limits).
3. **Run a small attribution study** on the top 2–3 stress windows (Feb 1, Feb 9):
   - Correlate with temperature, wind/solar output, outages, reserve prices, congestion indicators (requires additional datasets).

---

### Questions to tailor “actions” to your book (optional)
- Are you **net long or net short** (energy) in PJM RTO during these hours?
- Is your exposure primarily **RTO-wide** or tied to specific zones/nodes?
- Do you have **dispatchable generation**, **demand response**, or **storage** that can be optimized during spike windows?

```

### Sphinx

Added a client-ready add-on section translating observed PJM RTO volatility, stress windows (high RT spikes / wide RT–DA spreads / high load), oil dispatch surges, and episodic AEP uplift deviations into hypothesis-driven opportunities and concrete next steps, with clear caveats about data limits and non-causality.
