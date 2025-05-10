import streamlit as st

def render_optimization(data, strategy_type):
    st.header(f"📈 Tối ưu đầu tư - Chiến lược: {strategy_type}")
    st.info("Tính năng tối ưu đang được phát triển...")

def handle_optimization_menu(data):
    strategy_type = st.sidebar.radio("CHIẾN LƯỢC", ["Mua/bán", "Vào lệnh"])
    render_optimization(data, strategy_type)