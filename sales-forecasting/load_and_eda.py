
import pandas as pd
import matplotlib.pyplot as plt
import sys

FILENAME = "monthly_sales.csv"   


try:
    df = pd.read_csv(FILENAME, parse_dates=['date','Date','month','Month'], infer_datetime_format=True)
except Exception as e:
    
    print("parse_dates failed, loading without parse:", e)
    df = pd.read_csv(FILENAME)


date_col = None
for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]) or 'date' in c.lower() or 'month' in c.lower():
        date_col = c
        break
if date_col is None:
    
    for c in df.columns:
        try:
            tmp = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            if tmp.notna().sum() > 0:
                date_col = c
                df[c] = tmp
                break
        except:
            continue

if date_col is None:
    print("ERROR: no date-like column found. Columns:", df.columns.tolist())
    sys.exit(1)

df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
df = df.set_index(date_col).sort_index()


sales_col = None
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ('id',):
        sales_col = c
        break

if sales_col is None:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
        if pd.api.types.is_numeric_dtype(df[c]):
            sales_col = c
            break

if sales_col is None:
    print("ERROR: no numeric sales column found. Columns:", df.columns.tolist())
    sys.exit(1)

df = df[[sales_col]].rename(columns={sales_col: 'sales'})


print("\n===== HEAD =====")
print(df.head(10).to_string())
print("\n===== INFO =====")
print(df.info())
print("\n===== MISSING =====")
print(df['sales'].isna().sum(), "missing sales values out of", len(df))
print("\n===== INDEX SAMPLE =====")
print("Index dtype:", df.index.dtype)
print("Index first 5:", list(df.index[:5]))
print("Index freq:", getattr(df.index, 'freq', None))
print("Index diffs sample:", pd.Series(df.index).diff().unique()[:5])


plt.figure(figsize=(10,4))
plt.plot(df.index, df['sales'])
plt.title("Sales over time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Step 2: Feature Engineering
# ---------------------------

print("\n🔧 Starting feature engineering...")

# Make a copy to avoid modifying original unexpectedly
df_fe = df.copy()

# 1) Lag features (previous periods)
lags = [1, 2, 3, 6, 12]   # adjust to your needs
for lag in lags:
    df_fe[f"sales_lag_{lag}"] = df_fe['sales'].shift(lag)

# 2) Rolling window features (use shift(1) so rolling doesn't leak current target)
df_fe["rolling_mean_3"] = df_fe['sales'].shift(1).rolling(window=3, min_periods=1).mean()
df_fe["rolling_std_3"]  = df_fe['sales'].shift(1).rolling(window=3, min_periods=1).std()
df_fe["rolling_mean_6"] = df_fe['sales'].shift(1).rolling(window=6, min_periods=1).mean()
df_fe["rolling_std_6"]  = df_fe['sales'].shift(1).rolling(window=6, min_periods=1).std()

# 3) Expanding features (cumulative)
df_fe["expanding_mean"] = df_fe['sales'].shift(1).expanding(min_periods=1).mean()

# 4) Time-based features
df_fe["month"] = df_fe.index.month
df_fe["quarter"] = df_fe.index.quarter
df_fe["year"] = df_fe.index.year
# if your data is daily or weekly these may help:
df_fe["day"] = df_fe.index.day
df_fe["dayofweek"] = df_fe.index.dayofweek
df_fe["is_month_start"] = df_fe.index.is_month_start.astype(int)
df_fe["is_month_end"] = df_fe.index.is_month_end.astype(int)

# 5) Cyclical encoding for month (helps tree & linear models capture seasonality)
import numpy as np
df_fe["month_sin"] = np.sin(2 * np.pi * (df_fe["month"] / 12))
df_fe["month_cos"] = np.cos(2 * np.pi * (df_fe["month"] / 12))

# 6) Optionally create one-hot month (uncomment if you want dummies)
# df_fe = pd.get_dummies(df_fe, columns=["month"], prefix="m", drop_first=True)

# 7) Handle missing values created by lag/rolling
print("Rows before dropna():", len(df_fe))
df_fe = df_fe.dropna()
print("Rows after dropna():", len(df_fe))

# 8) Quick sanity checks / save
print("\n✅ Feature engineering finished. Columns now:")
print(list(df_fe.columns))

print("\nSample after FE:")
print(df_fe.head(10).to_string())

# Save processed dataset for modeling
OUTFILE = "processed_sales.csv"
df_fe.to_csv(OUTFILE, index=True)
print(f"\n📁 Processed data saved to: {OUTFILE}")
