import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.price_forecasting import (
    clean_dataset,
    get_countries,
    get_product_groups,
    get_sub_classifications,
    forecast_prices,
)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return clean_dataset(df)


def _friendly_feature_name(col: str) -> str:
    mapping = {
        "year":                      "Year",
        "month":                     "Month",
        "quarter":                   "Quarter",
        "Shipment Mode_enc":         "Shipment Mode",
        "Vendor_enc":                "Vendor",
        "Country_enc":               "Country",
        "Product Group_enc":         "Product Group",
        "Sub Classification_enc":    "Sub Classification",
        "log_quantity":              "Line Item Quantity (log)",
        "log_weight":                "Weight (log)",
        "log_freight":               "Freight Cost (log)",
    }
    return mapping.get(col, col)


# ──────────────────────────────────────────────
# MAIN TAB RENDERER
# ──────────────────────────────────────────────

def render_price_prediction_tab(df: pd.DataFrame):
    st.header("💰 Unit Price Prediction")
    st.write(
        "Select a country, product group, and sub-classification to train a price "
        "forecasting model using shipment mode, vendor, quantity, weight, and freight cost as features."
    )

    # --- Clean data once ---
    with st.spinner("Preparing dataset..."):
        df_clean = get_clean_df(df)

    # ── Cascading dropdowns ──────────────────────
    st.subheader("🔍 Filter Selection")
    col1, col2, col3 = st.columns(3)

    with col1:
        countries = get_countries(df_clean)
        country = st.selectbox("Country", countries)

    with col2:
        product_groups = get_product_groups(df_clean, country)
        product_group = st.selectbox("Product Group", product_groups)

    with col3:
        sub_classes = get_sub_classifications(df_clean, country, product_group)
        sub_classification = st.selectbox("Sub Classification", sub_classes)

    # ── Forecast settings ────────────────────────
    st.subheader("⚙️ Forecast Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        periods = st.slider("Forecast periods ahead", min_value=2, max_value=12, value=6)
    with col_b:
        freq_label = st.selectbox(
            "Forecast frequency",
            ["Monthly", "Quarterly"],
            index=1,
        )
    freq = "MS" if freq_label == "Monthly" else "QS"

    # ── Run ─────────────────────────────────────
    if st.button("🚀 Run Price Forecast", type="primary"):
        with st.spinner("Training model and forecasting..."):
            try:
                history_df, forecast_df, metrics, importance, model_used = forecast_prices(
                    df_clean=df_clean,
                    country=country,
                    product_group=product_group,
                    sub_classification=sub_classification,
                    periods=periods,
                    freq=freq,
                )
                _display_results(
                    history_df, forecast_df, metrics, importance,
                    model_used, country, product_group, sub_classification,
                )
            except ValueError as ve:
                st.warning(f"⚠️ {ve}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ──────────────────────────────────────────────
# RESULTS DISPLAY
# ──────────────────────────────────────────────

def _display_results(
    history_df, forecast_df, metrics, importance,
    model_used, country, product_group, sub_classification,
):
    title_str = f"{country} · {product_group} · {sub_classification}"

    st.success(f"✅ Model trained successfully using **{model_used}**")

    # ── 1. Summary metrics ───────────────────────
    st.subheader("📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  f"${metrics['mae']:.3f}")
    c2.metric("RMSE", f"${metrics['rmse']:.3f}")
    c3.metric("Training rows", f"{metrics['n_train']:,}")
    c3.metric("Test rows",     f"{metrics['n_test']:,}")

    # ── 2. Historical stats ──────────────────────
    st.subheader("📈 Historical Price Summary")
    avg = history_df["avg_unit_price"].mean()
    mn  = history_df["avg_unit_price"].min()
    mx  = history_df["avg_unit_price"].max()
    vol = history_df["avg_unit_price"].std() / avg * 100 if avg else 0

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Avg Unit Price", f"${avg:.3f}")
    h2.metric("Min",            f"${mn:.3f}")
    h3.metric("Max",            f"${mx:.3f}")
    h4.metric("Volatility (CV%)", f"{vol:.1f}%")

    # ── 3. Combined chart ────────────────────────
    st.subheader(f"📉 Price History & Forecast — {title_str}")

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=history_df["date"],
        y=history_df["avg_unit_price"],
        mode="lines+markers",
        name="Historical Avg Price",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecast_price"],
        mode="lines+markers",
        name=f"{model_used} Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))

    # Shade forecast region
    if not forecast_df.empty:
        fig.add_vrect(
            x0=str(forecast_df["date"].iloc[0]),
            x1=str(forecast_df["date"].iloc[-1]),
            fillcolor="rgba(255,127,14,0.08)",
            line_width=0,
            annotation_text="Forecast Period",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Unit Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Forecast table ────────────────────────
    st.subheader("📋 Forecast Values")
    display_fc = forecast_df.copy()
    display_fc["date"] = pd.to_datetime(display_fc["date"]).dt.strftime("%b %Y")
    display_fc = display_fc.rename(columns={
        "date": "Period",
        "forecast_price": "Forecast Unit Price (USD)",
    })
    display_fc["Forecast Unit Price (USD)"] = display_fc["Forecast Unit Price (USD)"].round(4)
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    # ── 5. Feature importance ────────────────────
    st.subheader("🔑 Feature Importance")
    fi_df = pd.DataFrame({
        "Feature":    [_friendly_feature_name(k) for k in importance],
        "Importance": list(importance.values()),
    })
    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="What drives unit price?",
    )
    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig_fi, use_container_width=True)
