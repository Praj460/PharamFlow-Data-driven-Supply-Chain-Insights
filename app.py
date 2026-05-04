import streamlit as st
from components.chatbot_ui import render_chatbot_tab
from components.forecast_ui import render_forecast_tab
from components.visualization_ui import render_visualization_tab
from components.price_forecasting_ui import render_price_prediction_tab
from components.shipment_mode_ui import render_shipment_mode_tab
from components.data_entry_ui import render_data_entry_tab
from components.Freight_Cost_Analysis import render_freight_cost_tab
from components.homepage_ui import render_homepage
from utils.google_sheets_loader import load_data_from_sheets

st.set_page_config(
    page_title="PharmaFlow",
    page_icon="💊",
    layout="wide"
)

# Load dataset
df, date_columns = load_data_from_sheets()

# Initialize page
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def navigate(target):
    st.session_state.page = target

# App Router
if st.session_state.page == "home":
    render_homepage()

elif st.session_state.page == "forecast":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_forecast_tab(df)

elif st.session_state.page == "visualization":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_visualization_tab(df, date_columns)

elif st.session_state.page == "price":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_price_prediction_tab(df)

elif st.session_state.page == "shipment":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_shipment_mode_tab(df)

elif st.session_state.page == "data_entry":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_data_entry_tab()

elif st.session_state.page == "freight":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_freight_cost_tab(df)

elif st.session_state.page == "chatbot":
    st.button("⬅️ Back to Home", on_click=go_home)
    render_chatbot_tab(df)
