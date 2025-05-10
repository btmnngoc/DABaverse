import pandas as pd
import re

def clean_indicator_name(indicator):
    """Clean indicator name by removing unit"""
    return re.sub(r'\n.+$', '', indicator)

def extract_unit(indicator_list):
    """Extract unit from indicator names"""
    units = set()
    for ind in indicator_list:
        match = re.search(r'\n(.+)', ind)
        if match:
            units.add(match.group(1))
    return ', '.join(units) if units else ''

import re

def clean_indicator_name(ind):
    """Remove trailing units (e.g. '\n%' or '\nLần')"""
    return re.sub(r'\n.+$', '', ind)

def get_indicator_groups():
    return {
        'Khả năng sinh lời': [
            'Tỷ suất sinh lợi trên tổng tài sản bình quân (ROAA)\n%',
            'Tỷ suất lợi nhuận trên vốn chủ sở hữu bình quân (ROEA)\n%',
            'Tỷ suất lợi nhuận gộp biên\n%',
            'Tỷ suất sinh lợi trên doanh thu thuần\n%'
        ],
        'Khả năng thanh toán': [
            'Tỷ số thanh toán hiện hành (ngắn hạn)\nLần',
            'Tỷ số thanh toán nhanh\nLần',
            'Tỷ số thanh toán bằng tiền mặt\nLần'
        ],
        'Hiệu quả hoạt động': [
            'Vòng quay tổng tài sản (Hiệu suất sử dụng toàn bộ tài sản)\nVòng',
            'Vòng quay hàng tồn kho\nVòng',
            'Vòng quay phải thu khách hàng\nVòng'
        ],
        'Chỉ số thị trường': [
            'Chỉ số giá thị trường trên thu nhập (P/E)\nLần',
            'Chỉ số giá thị trường trên giá trị sổ sách (P/B)\nLần',
            'Beta\nLần'
        ]
    }

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

import pandas as pd
import numpy as np

def advanced_preprocess(df):
    df = df.copy()

    # Tìm cột ngày
    if 'Date' not in df.columns:
        possible = [col for col in df.columns if 'date' in col.lower() or 'ngày' in col.lower()]
        if possible:
            df.rename(columns={possible[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors="coerce")

    numeric_cols = [
        "Total Volume", "Total Value", "Market Cap",
        "Closing Price", "Price Change", "Matched Volume", "Matched Value"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Date', 'Closing Price'], inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi