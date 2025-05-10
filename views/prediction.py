import streamlit as st
from services.predictor_fpt import run_prediction, predict_future_days


def render_prediction_tab():
    st.header("🔮 Dự đoán giá cổ phiếu")

    stock = st.selectbox("Chọn mã cổ phiếu", ["FPT", "CMG"], index=0)

    if st.button("Chạy mô hình dự đoán"):
        with st.spinner("Đang chạy mô hình..."):
            image_path, metrics = run_prediction(stock)

        st.success("✅ Dự đoán hoàn tất!")
        st.image(image_path, caption="Kết quả dự báo", use_column_width=True)

        st.subheader("📊 Độ chính xác mô hình:")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['MAE']}")
        col2.metric("RMSE", f"{metrics['RMSE']}")
        col3.metric("R²", f"{metrics['R2']}")

    n_days = st.slider("Chọn số ngày muốn dự báo", min_value=7, max_value=60, value=14)

    if st.button("📈 Dự báo"):
        with st.spinner("Đang chạy mô hình..."):
            df_forecast = predict_future_days(stock, n_days)

        st.success("✅ Dự báo hoàn tất!")
        st.dataframe(df_forecast)

