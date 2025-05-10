import streamlit as st
from services.predictor_fpt import run_prediction, predict_future_days


def render_prediction_tab():
    st.header("🔮 Dự đoán giá cổ phiếu")

    stock = st.selectbox("Chọn mã cổ phiếu", ["FPT", "CMG"], index=0)

    n_days = st.slider("Chọn số ngày muốn dự báo", min_value=7, max_value=60, value=14)

    if st.button("📈 Dự báo"):
        with st.spinner("Đang chạy mô hình..."):
            df_forecast = predict_future_days(stock, n_days)

        st.success("✅ Dự báo hoàn tất!")
        st.dataframe(df_forecast)

