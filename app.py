from data.loader import load_real_data
from views.analysis import handle_analysis_menu
from views.components import render_sidebar_header, render_footer, local_css
from data.loader import load_stock_transaction_data
from views.prediction import render_prediction_tab  # ✅from views.optimization import handle_optimization_menu



import streamlit as st

def main():
    local_css()
    data = load_real_data()
    render_sidebar_header()
    # 🔁 Thêm dữ liệu phân tích kỹ thuật
    transaction_data = load_stock_transaction_data()
    data.update(transaction_data)  # Hợp nhất vào dict `data`

    menu_option = st.sidebar.radio("MENU CHÍNH", ["Phân tích", "Dự đoán", "Tối ưu đầu tư"])
    st.sidebar.markdown("---")

    if menu_option == "Phân tích":
        handle_analysis_menu(data)
    elif menu_option == "Dự đoán":
        render_prediction_tab()
    elif menu_option == "Tối ưu đầu tư":
        handle_optimization_menu(data)

    render_footer()

if __name__ == "__main__":
    main()