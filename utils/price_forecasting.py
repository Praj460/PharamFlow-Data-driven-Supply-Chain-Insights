import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    MODEL_BACKEND = "xgb"
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    MODEL_BACKEND = "rf"


# ──────────────────────────────────────────────
# 1. DATA CLEANING
# ──────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the SCMS dataset.
    Returns a clean DataFrame ready for feature engineering.
    """
    df = df.copy()

    # --- Unit Price ---
    df["Unit Price"] = pd.to_numeric(df["Unit Price"], errors="coerce")
    df = df[df["Unit Price"] > 0]           # drop zero / negative prices
    df = df[df["Unit Price"] < df["Unit Price"].quantile(0.99)]  # remove extreme outliers

    # --- Date ---
    df["Delivered to Client Date"] = pd.to_datetime(
        df["Delivered to Client Date"], errors="coerce", dayfirst=True
    )
    df = df.dropna(subset=["Delivered to Client Date", "Unit Price"])

    # --- Line Item Quantity ---
    df["Line Item Quantity"] = pd.to_numeric(df["Line Item Quantity"], errors="coerce")
    df["Line Item Quantity"] = df["Line Item Quantity"].fillna(df["Line Item Quantity"].median())

    # --- Weight: "Weight Captured Separately" and similar non-numeric → NaN → median ---
    df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")
    df["Weight (Kilograms)"] = df["Weight (Kilograms)"].fillna(df["Weight (Kilograms)"].median())

    # --- Freight Cost: text values → NaN → 0 (often means included / not applicable) ---
    df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")
    df["Freight Cost (USD)"] = df["Freight Cost (USD)"].fillna(0)

    # --- Categorical cleaning ---
    for col in ["Country", "Product Group", "Sub Classification", "Shipment Mode", "Vendor"]:
        df[col] = df[col].astype(str).str.strip()

    # --- Vendor: keep top N, group rest as "Other" ---
    top_vendors = df["Vendor"].value_counts().nlargest(10).index
    df["Vendor"] = df["Vendor"].where(df["Vendor"].isin(top_vendors), other="Other")

    # --- Shipment Mode: keep known modes, group rest ---
    known_modes = {"Air", "Truck", "Air Charter", "Ocean"}
    df["Shipment Mode"] = df["Shipment Mode"].where(df["Shipment Mode"].isin(known_modes), other="Other")

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. FILTER HELPERS (used by UI dropdowns)
# ──────────────────────────────────────────────

def get_countries(df: pd.DataFrame) -> list:
    return sorted(df["Country"].dropna().unique().tolist())

def get_product_groups(df: pd.DataFrame, country: str) -> list:
    subset = df[df["Country"] == country] if country else df
    return sorted(subset["Product Group"].dropna().unique().tolist())

def get_sub_classifications(df: pd.DataFrame, country: str, product_group: str) -> list:
    subset = df.copy()
    if country:
        subset = subset[subset["Country"] == country]
    if product_group:
        subset = subset[subset["Product Group"] == product_group]
    return sorted(subset["Sub Classification"].dropna().unique().tolist())


# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix used by the ML model.
    All encoding is done here so the model only sees numbers.
    """
    df = df.copy()

    # --- Time features from delivery date ---
    df["year"]  = df["Delivered to Client Date"].dt.year
    df["month"] = df["Delivered to Client Date"].dt.month
    df["quarter"] = df["Delivered to Client Date"].dt.quarter

    # --- Label-encode categoricals ---
    le = LabelEncoder()
    for col in ["Shipment Mode", "Vendor", "Country", "Product Group", "Sub Classification"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # --- Log-transform skewed numerics (helps tree models too) ---
    df["log_quantity"] = np.log1p(df["Line Item Quantity"])
    df["log_weight"]   = np.log1p(df["Weight (Kilograms)"])
    df["log_freight"]  = np.log1p(df["Freight Cost (USD)"])

    return df


FEATURE_COLS = [
    "year", "month", "quarter",
    "Shipment Mode_enc", "Vendor_enc",
    "Country_enc", "Product Group_enc", "Sub Classification_enc",
    "log_quantity", "log_weight", "log_freight",
]


# ──────────────────────────────────────────────
# 4. MODEL TRAINING
# ──────────────────────────────────────────────

def train_price_model(df_engineered: pd.DataFrame):
    """
    Train XGBoost (or RF fallback) on the engineered feature set.
    Uses a chronological train/test split (no data leakage).

    Returns: model, metrics dict, feature importance dict
    """
    df = df_engineered.dropna(subset=FEATURE_COLS + ["Unit Price"]).copy()

    if len(df) < 20:
        raise ValueError("Not enough data to train a model (need ≥ 20 rows after filtering).")

    # Chronological split
    df = df.sort_values("Delivered to Client Date")
    split = int(len(df) * 0.8)
    X = df[FEATURE_COLS]
    y = df["Unit Price"]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if MODEL_BACKEND == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    

    metrics = {"mae": mae, "rmse": rmse, "n_train": split, "n_test": len(df) - split}

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return model, metrics, importance


# ──────────────────────────────────────────────
# 5. FORECASTING
# ──────────────────────────────────────────────

def build_forecast_rows(
    last_row: pd.Series,
    periods: int,
    freq: str = "QS",          # quarterly by default
) -> pd.DataFrame:
    """
    Build synthetic future rows by advancing the date and keeping
    all other features constant (i.e. same country/product/vendor/mode).
    """
    rows = []
    current_date = last_row["Delivered to Client Date"]
    offset = pd.tseries.frequencies.to_offset(freq)

    for _ in range(periods):
        current_date = current_date + offset
        row = last_row.copy()
        row["Delivered to Client Date"] = current_date
        row["year"]    = current_date.year
        row["month"]   = current_date.month
        row["quarter"] = current_date.quarter
        rows.append(row)

    return pd.DataFrame(rows)


def forecast_prices(
    df_clean: pd.DataFrame,
    country: str,
    product_group: str,
    sub_classification: str,
    periods: int = 6,
    freq: str = "QS",
):
    """
    End-to-end: filter → engineer → train → forecast.

    Returns
    -------
    history_df   : monthly aggregated historical unit prices for the selection
    forecast_df  : DataFrame with columns [date, forecast_price]
    metrics      : dict with mae, rmse, mape, n_train, n_test
    importance   : dict of feature importances
    model_used   : "XGBoost" or "RandomForest"
    """
    # --- Filter to selection ---
    mask = (
        (df_clean["Country"] == country) &
        (df_clean["Product Group"] == product_group) &
        (df_clean["Sub Classification"] == sub_classification)
    )
    subset = df_clean[mask].copy()

    if len(subset) < 20:
        raise ValueError(
            f"Only {len(subset)} rows found for {country} / {product_group} / {sub_classification}. "
            "Need at least 20 to train a model. Try a broader selection."
        )

    # --- Engineer features ---
    subset_eng = engineer_features(subset)

    # --- Train ---
    model, metrics, importance = train_price_model(subset_eng)

    # --- Historical monthly average (for chart) ---
    history_df = (
        subset.set_index("Delivered to Client Date")["Unit Price"]
        .resample("MS")
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"Delivered to Client Date": "date", "Unit Price": "avg_unit_price"})
    )

    # --- Build future rows from the last known data point ---
    subset_eng = subset_eng.sort_values("Delivered to Client Date")
    last_row = subset_eng.iloc[-1].copy()

    future_df = build_forecast_rows(last_row, periods=periods, freq=freq)
    forecast_prices_arr = model.predict(future_df[FEATURE_COLS])

    forecast_df = pd.DataFrame({
        "date": future_df["Delivered to Client Date"].values,
        "forecast_price": forecast_prices_arr,
    })

    model_used = "XGBoost" if MODEL_BACKEND == "xgb" else "RandomForest"

    return history_df, forecast_df, metrics, importance, model_used
