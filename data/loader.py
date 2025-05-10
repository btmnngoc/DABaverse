import pandas as pd
import os
import streamlit as st

def load_real_data():
    data_paths = {
        'FPT': 'assets/data/6.2 (his) financialreport_metrics_FPT_CMG_processed.csv',
        'CMG': 'assets/data/6.2 (his) financialreport_metrics_FPT_CMG_processed.csv',
        'Thị trường': 'assets/data/6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv',
        'Ngành CNTT': 'assets/data/6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv'
    }

    data = {}
    for key, path in data_paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            st.error(f"Không tìm thấy file dữ liệu: {path}")
            data[key] = pd.DataFrame()

    return data


import pandas as pd
import re
from pathlib import Path
from pandas.api.types import CategoricalDtype


def load_financial_data():
    """Load and process financial data from CSV"""
    data_dir = Path(__file__).parent.parent / "assets" / "data"
    file_path = data_dir / "6.2 (his) financialreport_metrics_FPT_CMG_processed.csv"

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # Clean Indicator column
        df['Indicator'] = df['Indicator'].astype(str).str.strip()

        # Melt to long format
        time_cols = [col for col in df.columns if col.startswith('Q')]
        df_long = df.melt(
            id_vars=['Indicator', 'StockID'],
            value_vars=time_cols,
            var_name='Period',
            value_name='Value'
        )

        # Clean Value column
        df_long['Value'] = (
            df_long['Value']
            .astype(str)
            .str.replace(',', '')
            .str.replace('\n', '')
            .replace('', pd.NA)
            .astype(float)
        )
        df_long.dropna(subset=['Value'], inplace=True)

        # Cập nhật phần chuẩn hóa định dạng thời gian
        period_order = [
            'Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023',
            'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'  # Thêm các quý 2024 nếu có
        ]

        # Chuyển Period thành kiểu Categorical với thứ tự đúng
        period_type = CategoricalDtype(categories=period_order, ordered=True)
        df_long['Period'] = df_long['Period'].astype(period_type)
        df_long = df_long.sort_values(['Period'])

        return df_long

    except Exception as e:
        raise Exception(f"Error loading financial data: {str(e)}")



def get_indicator_groups():
    """Define financial indicator groups"""
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
        'Đòn bẩy tài chính': [
            'Tỷ số Nợ trên Tổng tài sản\n%',
            'Tỷ số Nợ trên Vốn chủ sở hữu\n%'
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

def extract_unit(indicator_list):
    """Lấy đơn vị từ chỉ số"""
    units = set()
    for ind in indicator_list:
        match = re.search(r'\n(.+)', ind)
        if match:
            units.add(match.group(1))
    return ', '.join(units) if units else ''


def load_financial_data1():
    """Load and process financial data from CSV"""
    data_dir = Path(__file__).parent.parent / "assets" / "data"
    file_path = data_dir / "6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv"

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # Clean Indicator column
        df['Indicator'] = df['Indicator'].astype(str).str.strip()

        # Melt to long format
        time_cols = [col for col in df.columns if col.startswith('Q')]
        df_long = df.melt(
            id_vars=['Indicator', 'StockID'],
            value_vars=time_cols,
            var_name='Period',
            value_name='Value'
        )

        # Clean Value column
        df_long['Value'] = (
            df_long['Value']
            .astype(str)
            .str.replace(',', '')
            .str.replace('\n', '')
            .replace('', pd.NA)
            .astype(float)
        )
        df_long.dropna(subset=['Value'], inplace=True)

        # Cập nhật phần chuẩn hóa định dạng thời gian
        period_order = [
            'Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023',
            'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024'  # Thêm các quý 2024 nếu có
        ]

        # Chuyển Period thành kiểu Categorical với thứ tự đúng
        period_type = CategoricalDtype(categories=period_order, ordered=True)
        df_long['Period'] = df_long['Period'].astype(period_type)
        df_long = df_long.sort_values(['Period'])

        return df_long

    except Exception as e:
        raise Exception(f"Error loading financial data: {str(e)}")

from services.financial_utils import advanced_preprocess

def load_stock_transaction_data():
    file_paths = {
        'FPT': 'assets/data/4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv',
        'CMG': 'assets/data/4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv',
    }

    stock_data = {}
    for stock, path in file_paths.items():
        if os.path.exists(path):
            raw_df = pd.read_csv(path)
            stock_data[stock] = advanced_preprocess(raw_df)
        else:
            stock_data[stock] = pd.DataFrame()
    return stock_data