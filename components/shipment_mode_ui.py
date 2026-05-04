import streamlit as st
import pandas as pd
import plotly.express as px

# Import cleaning function
from utils.freight_utils import clean_freight_cost_column_with_id_priority

def render_shipment_mode_tab(df):
    st.header("🚛 Shipment Mode Analysis")
    st.subheader("Analyze Freight Costs and Trends by Shipment Mode")

    # 🚛 Clean Freight Cost data
    filtered_df = clean_freight_cost_column_with_id_priority(df)

    # 🔎 Filters: Country and Product Group
    country_list = sorted(filtered_df["Country"].dropna().unique())
    selected_countries = st.multiselect(
        "Select Country(s)", country_list, default=country_list, key="ship_country"
    )
    product_list = sorted(filtered_df["Product Group"].dropna().unique())
    selected_products = st.multiselect(
        "Select Product Group(s)", product_list, default=product_list, key="ship_product"
    )

    # Apply filters
    filtered_df = filtered_df[
        filtered_df["Country"].isin(selected_countries) &
        filtered_df["Product Group"].isin(selected_products)
    ]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # Normalize and parse
    filtered_df["Shipment Mode"] = filtered_df["Shipment Mode"].fillna("Unknown")
    filtered_df["Freight Cost (USD)"] = filtered_df["Freight Cost (USD)"].fillna(0)
    filtered_df["Delivered to Client Date"] = pd.to_datetime(
        filtered_df["Delivered to Client Date"], errors="coerce"
    )

    st.divider()

    # 📋 Shipment Mode KPIs
    st.subheader("📋 Shipment Mode KPIs")
    total_shipments = len(filtered_df)
    mode_counts = filtered_df["Shipment Mode"].value_counts(normalize=True) * 100
    air_pct = mode_counts.get("Air", 0) + mode_counts.get("Air charter", 0)
    ocean_pct = mode_counts.get("Ocean", 0)
    truck_pct = mode_counts.get("Truck", 0)
    unknown_pct = 100 - (air_pct + ocean_pct + truck_pct)

    cols = st.columns(5)
    cols[0].metric("Total Shipments", f"{total_shipments}")
    cols[1].metric("Air (%)", f"{air_pct:.1f}%")
    cols[2].metric("Ocean (%)", f"{ocean_pct:.1f}%")
    cols[3].metric("Truck (%)", f"{truck_pct:.1f}%")
    cols[4].metric("Unknown (%)", f"{unknown_pct:.1f}%")

    st.divider()

    # 📦 Average Freight Cost by Shipment Mode (Bar Chart)
    st.subheader("📦 Average Freight Cost by Shipment Mode")
    mode_cost = (
        filtered_df.groupby("Shipment Mode")["Freight Cost (USD)"]
        .mean()
        .reset_index()
    )
    fig1 = px.bar(
        mode_cost,
        x="Shipment Mode",
        y="Freight Cost (USD)",
        color="Shipment Mode",
        text_auto=".2s",
        title="Average Freight Cost by Shipment Mode"
    )
    fig1.update_layout(hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # 📈 Freight Cost Trends for Each Shipment Mode (One Chart per Mode)
    st.subheader("📈 Freight Cost Trends for Each Shipment Mode")
    df_time = (
        filtered_df.dropna(subset=["Delivered to Client Date"])  
        .groupby([pd.Grouper(key="Delivered to Client Date", freq="W"), "Shipment Mode"] )
        ["Freight Cost (USD)"].mean().reset_index()
    )
    unique_modes = df_time["Shipment Mode"].unique()
    for mode in unique_modes:
        st.markdown(f"#### {mode} Shipments")
        mode_df = df_time[df_time["Shipment Mode"] == mode]
        fig_mode = px.line(
            mode_df,
            x="Delivered to Client Date",
            y="Freight Cost (USD)",
            title=f"Freight Cost Trend - {mode}",
        )
        fig_mode.update_layout(xaxis_title="Date", yaxis_title="Avg Freight Cost (USD)", hovermode="x unified")
        st.plotly_chart(fig_mode, use_container_width=True)

    st.divider()

    # 🗓️ Monthly Seasonality: Avg Freight Cost per Month for Each Mode
    st.subheader("🗓️ Monthly Seasonality: Avg Freight Cost per Month by Shipment Mode")
    monthly = (
        filtered_df.assign(
            Month_Num=filtered_df["Delivered to Client Date"].dt.month,
            Month=filtered_df["Delivered to Client Date"].dt.month_name()
        )
        .groupby(["Shipment Mode", "Month_Num", "Month"])  
        ["Freight Cost (USD)"].mean().reset_index(name="Avg Freight Cost")
        .sort_values(["Shipment Mode", "Month_Num"])
    )
    modes = monthly["Shipment Mode"].unique()
    for mode in modes:
        st.markdown(f"#### {mode} Seasonality")
        mode_month = monthly[monthly["Shipment Mode"] == mode]
        fig_m = px.line(
            mode_month,
            x="Month",
            y="Avg Freight Cost",
            markers=True,
            title=f"Avg Freight Cost by Month - {mode}"
        )
        fig_m.update_layout(xaxis_title="Month", yaxis_title="Avg Freight Cost (USD)", hovermode="x unified")
        st.plotly_chart(fig_m, use_container_width=True)
