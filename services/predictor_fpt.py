
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import matplotlib.pyplot as plt

def run_prediction(stock_id='FPT'):
    # 1. Đọc dữ liệu giao dịch
    file_path = os.path.join("assets", "data", f"4.2.3 (TARGET) (live & his) {stock_id}_detail_transactions_processed.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
    df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)

    # 2. Đặc trưng kỹ thuật
    df['Return%'] = df['Closing Price'].pct_change() * 100
    df['MA5'] = df['Closing Price'].rolling(window=5).mean()
    df['MA10'] = df['Closing Price'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Total Volume'] / df['Total Volume'].rolling(5).mean()
    df['Volatility'] = df['Closing Price'].pct_change().rolling(window=5).std() * 100
    df['Price_Momentum'] = df['Closing Price'].diff(5)
    df = df.fillna(0)

    # 3. Sự kiện cổ tức & họp ĐHĐCĐ
    df_dividend = pd.read_csv(os.path.join("assets", "data", "3.2 (live & his) news_dividend_issue (FPT_CMG)_processed.csv"))
    df_meeting = pd.read_csv(os.path.join("assets", "data", "3.3 (live & his) news_shareholder_meeting (FPT_CMG)_processed.csv"))

    df_dividend = df_dividend[df_dividend['StockID'] == stock_id].copy()
    df_meeting = df_meeting[df_meeting['StockID'] == stock_id].copy()
    df_dividend['Execution Date'] = pd.to_datetime(df_dividend['Execution Date'], format='%d/%m/%Y', errors='coerce')
    df_meeting['Execution Date'] = pd.to_datetime(df_meeting['Execution Date'], format='%d/%m/%Y')
    df['Dividend_Event'] = df['Date'].isin(df_dividend['Execution Date']).astype(int)
    df['Meeting_Event'] = df['Date'].isin(df_meeting['Execution Date']).astype(int)

    # 4. Dữ liệu tài chính
    df_financial = pd.read_csv(os.path.join("assets", "data", "6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv"))
    df_financial = df_financial[df_financial['Stocks'].str.contains(stock_id)].copy()
    df_financial['Indicator'] = df_financial['Indicator'].str.replace('\n', '', regex=True).str.strip()
    for col in df_financial.columns[3:]:
        df_financial[col] = df_financial[col].astype(str).str.replace(',', '').astype(float, errors='ignore')

    indicators = [
        'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
        'Tỷ lệ lãi EBIT%',
        'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
        'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
        'P/SLần',
        'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
        'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
    ]

    df_financial = df_financial[df_financial['Indicator'].isin(indicators)]
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    df_melted = df_financial.melt(id_vars=['Indicator'], value_vars=quarters, var_name='Quarter', value_name='Value')
    df_melted['Date'] = pd.to_datetime(df_melted['Quarter'].map({
        'Q1_2023': '2023-01-01', 'Q2_2023': '2023-04-01', 'Q3_2023': '2023-07-01', 'Q4_2023': '2023-10-01',
        'Q1_2024': '2024-01-01', 'Q2_2024': '2024-04-01', 'Q3_2024': '2024-07-01', 'Q4_2024': '2024-10-01'
    }))
    df_pivot = df_melted.pivot(index='Date', columns='Indicator', values='Value')
    df = df.merge(df_pivot, on='Date', how='left')
    df[indicators] = df[indicators].ffill()
    df.dropna(inplace=True)

    # 5. Chuẩn bị dữ liệu
    features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum'] + indicators
    X_xgb = df[features_xgb]
    y = df['Closing Price']

    y_log = np.log1p(y)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(y_log.values.reshape(-1, 1))
    lookback = 7
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled)):
        X_lstm.append(scaled[i - lookback:i, 0])
        y_lstm.append(scaled[i, 0])
    X_lstm = np.array(X_lstm).reshape(-1, lookback, 1)
    y_lstm = np.array(y_lstm)

    X_xgb = X_xgb.iloc[lookback:].reset_index(drop=True)
    y = y.iloc[lookback:].reset_index(drop=True)
    mask = ~(X_xgb.isna().any(axis=1) | y.isna())
    X_xgb, y = X_xgb[mask], y[mask].reset_index(drop=True)
    X_lstm, y_lstm = X_lstm[mask], y_lstm[mask]

    split = int(len(y) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
    y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]
    X_xgb_train, X_xgb_test = X_xgb[:split], X_xgb[split:]
    y_train, y_test = y[:split], y[split:]

    # 6. LSTM
    model_lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(loss='mse', optimizer=Adam(0.001))
    model_lstm.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=16, verbose=0)
    pred_lstm_test = model_lstm.predict(X_lstm_test)
    pred_lstm_test = np.expm1(scaler.inverse_transform(pred_lstm_test)).flatten()

    # 7. XGBoost
    model_xgb = xgb.XGBRegressor()
    model_xgb.fit(X_xgb_train, y_train)
    pred_xgb_test = model_xgb.predict(X_xgb_test)

    # 8. Meta model
    X_meta = np.vstack((pred_lstm_test, pred_xgb_test)).T
    meta_model = Sequential([
        Dense(64, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    meta_model.compile(optimizer=Adam(0.005), loss='mse')
    meta_model.fit(X_meta, y_test, epochs=150, batch_size=32, verbose=0)

    y_pred = meta_model.predict(X_meta).flatten()
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), neginf=0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 9. Plot
    fig_path = f"assets/outputs/{stock_id.lower()}_stock_prediction.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'].iloc[-len(y_test):], y_test, label='Giá thật')
    plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Dự báo', linestyle='--')
    plt.title(f"Dự báo giá cổ phiếu {stock_id}")
    plt.xlabel("Ngày")
    plt.ylabel("Giá đóng cửa")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path, {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)}

def predict_future_days(stock_id='FPT', n_days=14):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # 1. Đọc dữ liệu
    df = pd.read_csv(f"assets/data/4.2.3 (TARGET) (live & his) {stock_id}_detail_transactions_processed.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df = df.sort_values("Date")
    df['Closing Price'] = df['Closing Price'].str.replace(",", "").astype(float)

    # 2. Chuẩn bị dữ liệu cho LSTM
    prices = df['Closing Price'].values
    y_log = np.log1p(prices)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(y_log.reshape(-1, 1))

    lookback = 7
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled)):
        X_lstm.append(scaled[i - lookback:i, 0])
        y_lstm.append(scaled[i, 0])

    X_lstm = np.array(X_lstm).reshape(-1, lookback, 1)
    y_lstm = np.array(y_lstm)

    # 3. Tạo và huấn luyện model LSTM
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0)

    # 4. Dự báo tương lai
    recent_seq = scaled[-lookback:].reshape(1, lookback, 1)
    forecast_scaled = []

    for _ in range(n_days):
        pred = model.predict(recent_seq, verbose=0)[0][0]
        forecast_scaled.append(pred)
        recent_seq = np.append(recent_seq[0, 1:, 0], pred).reshape(1, lookback, 1)

    forecast_prices = np.expm1(scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))).flatten()

    # 5. Trả kết quả
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    return pd.DataFrame({
        'Ngày': future_dates,
        'Giá dự báo': forecast_prices
    })
