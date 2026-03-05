import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns

df = pd.read_parquet('/data/quant14/EBY/day1.parquet/')

df = df[['Time', 'Price']]
fig = px.line(df, x='Time', y='Price')
fig.update_layout(title='Price Plot', xaxis_title='Time', yaxis_title='Price')
fig.show()
