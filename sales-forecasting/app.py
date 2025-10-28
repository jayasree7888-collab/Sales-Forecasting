import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import MonthBegin
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecast Dashboard (Enhanced)", layout="wide")

# ----------------- Helpers -----------------
def make_features_from_df(df):
    df = df.sort_index().copy()
    df_fe = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        df_fe[f"sales_lag_{lag}"] = df_fe["sales"].shift(lag)
    df_fe["rolling_mean_3"] = df_fe["sales"].shift(1).rolling(window=3, min_periods=1).mean()
    df_fe["rolling_mean_6"] = df_fe["sales"].shift(1).rolling(window=6, min_periods=1).mean()
    df_fe["month"] = df_fe.index.month
    df_fe["year"] = df_fe.index.year
    df_fe["quarter"] = df_fe.index.quarter
    df_fe = df_fe.dropna()
    return df_fe


def forecast_with_model(model, df_fe, months=6, return_simulations=False, n_sims=1000, residuals=None, method='normal', zscore=1.96):
    all_sales = list(df_fe["sales"].astype(float).values)
    lags = [1, 2, 3, 6, 12]
    future_dates = pd.date_range(df_fe.index[-1] + MonthBegin(1), periods=months, freq="MS")
    preds = []
    sims = []
    for date in future_dates:
        feat = {}
        for lag in lags:
            feat[f"sales_lag_{lag}"] = all_sales[-lag] if len(all_sales) >= lag else np.mean(all_sales)
        feat["rolling_mean_3"] = np.mean(all_sales[-3:])
        feat["rolling_mean_6"] = np.mean(all_sales[-6:])
        feat["month"] = date.month
        feat["year"] = date.year
        feat["quarter"] = (date.month - 1) // 3 + 1

        X_new = pd.DataFrame([feat]).reindex(columns=df_fe.drop(columns="sales").columns, fill_value=0)
        point = float(model.predict(X_new)[0])
        preds.append(point)

        # Simulate for CI if requested
        if return_simulations:
            if method == 'bootstrap' and residuals is not None and len(residuals) > 0:
                sim_vals = [point + np.random.choice(residuals) for _ in range(n_sims)]
                sims.append(sim_vals)
            elif method == 'normal' and residuals is not None:
                s = np.std(residuals)
                sims.append(list(point + np.random.normal(0, s, size=n_sims)))
            else:
                sims.append([point] * n_sims)

        all_sales.append(point)

    forecast_df = pd.DataFrame({"Predicted_Sales": preds}, index=future_dates)
    if return_simulations:
        sims_arr = np.array(sims).T
        return forecast_df, sims_arr
    return forecast_df


# ----------------- Streamlit UI -----------------
st.title("📊 Sales Forecast Dashboard — Visuals + Retrain")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    months_to_forecast = st.slider("Months to forecast", 1, 12, 6)
    show_intervals = st.checkbox("Show prediction intervals (CI)", value=True)
    ci_method = st.selectbox("CI method", ["normal (z*std)", "residual bootstrap"], index=0)
    if ci_method == "normal (z*std)":
        z_val = st.number_input("z-value (for CI)", value=1.96, step=0.1)
    else:
        n_sims = st.number_input("Bootstrap sims", min_value=200, max_value=5000, value=1000, step=100)

    st.markdown("---")
    st.subheader("Retrain Model")
    uploaded_file = st.file_uploader("Upload CSV (date,sales)", type=["csv"])
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=2000, value=300, step=50)
    learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f")
    max_depth = st.number_input("max_depth", min_value=1, max_value=15, value=4)
    save_new_model = st.checkbox("Save trained model as sales_forecast_model.pkl", value=True)
    retrain_btn = st.button("🔁 Retrain model from uploaded CSV")

# Load saved model
model = None
try:
    model = joblib.load("sales_forecast_model.pkl")
except Exception as e:
    st.warning(f"Existing model not found or failed to load: {e}")

# Retrain logic
if retrain_btn:
    if uploaded_file is None:
        st.error("Please upload a CSV to retrain.")
    else:
        try:
            data = pd.read_csv(uploaded_file)
            # Identify date and sales columns
            date_col = None
            for c in data.columns:
                if 'date' in c.lower() or 'month' in c.lower():
                    date_col = c
                    break
            if date_col is None:
                st.error("CSV must contain a date-like column.")
                st.stop()

            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            if 'sales' not in data.columns:
                for c in data.columns:
                    if pd.api.types.is_numeric_dtype(data[c]):
                        data = data.rename(columns={c: 'sales'})
                        break
            data = data.dropna(subset=[date_col, 'sales']).set_index(date_col).sort_index()

            st.info(f"Data loaded with {len(data)} rows. Starting feature engineering and training...")

            df_fe_new = make_features_from_df(data)
            X = df_fe_new.drop(columns="sales")
            y = df_fe_new["sales"]
            test_size = max(3, int(0.15 * len(df_fe_new)))
            train_len = len(df_fe_new) - test_size
            X_train, X_test = X.iloc[:train_len], X.iloc[train_len:]
            y_train, y_test = y.iloc[:train_len], y.iloc[train_len:]

            with st.spinner("Training XGBoost model..."):
                new_model = XGBRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate),
                    max_depth=int(max_depth),
                    subsample=0.9,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
                new_model.fit(X_train, y_train)

            y_pred = new_model.predict(X_test)
            rmse_new = np.sqrt(mean_squared_error(y_test, y_pred))
            st.success(f"✅ Model retrained successfully! Test RMSE = {rmse_new:.2f}")

            model = new_model
            residuals = (y_test.values - y_pred).tolist()

            if save_new_model:
                joblib.dump(new_model, "sales_forecast_model.pkl")
                st.info("💾 New model saved as sales_forecast_model.pkl")

        except Exception as e:
            st.error(f"Retrain failed: {e}")
            st.stop()

# Load data
if uploaded_file is not None and model is None:
    try:
        raw = pd.read_csv(uploaded_file)
        date_col = None
        for c in raw.columns:
            if 'date' in c.lower() or 'month' in c.lower():
                date_col = c
                break
        raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
        if 'sales' not in raw.columns:
            for c in raw.columns:
                if pd.api.types.is_numeric_dtype(raw[c]):
                    raw = raw.rename(columns={c: 'sales'})
                    break
        raw = raw.dropna(subset=[date_col, 'sales']).set_index(date_col).sort_index()
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
        st.stop()
else:
    try:
        raw = pd.read_csv("processed_sales.csv", index_col=0, parse_dates=True)
    except Exception as e:
        st.error(f"Could not load processed_sales.csv and no uploaded data: {e}")
        st.stop()

if model is None:
    st.error("No model available. Either load a saved model or retrain from a CSV.")
    st.stop()

# Build features for forecasting/backtest
df_fe = make_features_from_df(raw)

# Backtest RMSE
TEST_SIZE = min(6, len(df_fe)//4)
if TEST_SIZE >= 1:
    X = df_fe.drop(columns="sales")
    y = df_fe["sales"]
    X_train, X_test = X.iloc[:-TEST_SIZE], X.iloc[-TEST_SIZE:]
    y_train, y_test = y.iloc[:-TEST_SIZE], y.iloc[-TEST_SIZE:]
    y_pred = model.predict(X_test)
    backtest_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.metric("Backtest RMSE", f"{backtest_rmse:.2f}")
    residuals_for_ci = (y_test.values - y_pred).tolist()
else:
    residuals_for_ci = []

# Forecast
if show_intervals and len(residuals_for_ci) > 0:
    if ci_method.startswith("normal"):
        forecast_df, sims = forecast_with_model(model, df_fe, months=months_to_forecast,
                                                return_simulations=True, n_sims=1000,
                                                residuals=np.array(residuals_for_ci), method='normal', zscore=z_val)
        lower = np.percentile(sims, 2.5, axis=0)
        upper = np.percentile(sims, 97.5, axis=0)
    else:
        forecast_df, sims = forecast_with_model(model, df_fe, months=months_to_forecast,
                                                return_simulations=True, n_sims=int(n_sims),
                                                residuals=np.array(residuals_for_ci), method='bootstrap')
        lower = np.percentile(sims, 2.5, axis=0)
        upper = np.percentile(sims, 97.5, axis=0)
else:
    forecast_df = forecast_with_model(model, df_fe, months=months_to_forecast)
    lower = None
    upper = None

# ----------------- Plotting -----------------
st.subheader("Historical + Forecast")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(raw.index, raw["sales"], label="Historical", marker='o', linewidth=1.6)
ax.plot(forecast_df.index, forecast_df["Predicted_Sales"], label="Forecast", marker='x', color="orange", linewidth=1.8)

if show_intervals and lower is not None and upper is not None:
    ax.fill_between(forecast_df.index, lower, upper, color='orange', alpha=0.2, label="95% CI")

ax.set_title("Sales: Historical & Forecast", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

# Forecast table + download
st.subheader("Forecast Table")
st.dataframe(forecast_df.style.format({"Predicted_Sales": "{:.0f}"}))
csv_bytes = forecast_df.to_csv().encode("utf-8")
st.download_button("📥 Download forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")

if TEST_SIZE >= 1:
    st.subheader("Backtest: Actual vs Predicted")
    comp = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index)
    st.table(comp.style.format({"Actual": "{:.0f}", "Predicted": "{:.0f}"}))

st.info("✅ Use the sidebar to upload a CSV and retrain the model with new hyperparameters.")
