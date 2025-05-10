import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import re
import plotly.express as px

def plot_financial_metrics(df, stock, indicator_group):
    """Create interactive financial metrics plot"""

    # Check type of indicator_group
    if isinstance(indicator_group, dict):
        indicator_list = list(indicator_group.values())[0]
        title_text = list(indicator_group.keys())[0]
    else:
        indicator_list = indicator_group
        title_text = "Chỉ số tài chính"

    # Filter data
    sub = df[(df['StockID'] == stock) &
             (df['Indicator'].isin(indicator_list))]

    if sub.empty:
        return None

    # Clean indicator names for display
    sub['Indicator_clean'] = sub['Indicator'].apply(
        lambda x: re.sub(r'\n.+$', '', x)
    )

    # Create figure
    fig = px.line(
        sub,
        x='Period',
        y='Value',
        color='Indicator_clean',
        markers=True,

        labels={
            'Value': 'Giá trị',
            'Period': 'Kỳ báo cáo',
            'Indicator_clean': 'Chỉ số'
        },
        title = f"{title_text} - {stock}",
    )
    # Trong hàm plot_financial_metrics()
    fig.update_xaxes(
        categoryorder='array',
        categoryarray=df['Period'].unique().sort_values())

    # Customize layout
    fig.update_layout(
        hovermode="x unified",
        legend_title_text='Chỉ số',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=0, r=0, b=50, t=130)
    )

    return fig


