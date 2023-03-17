import mplfinance as mpf
import pandas as pd

ch = pd.read_csv('raw_1.csv', index_col=0, parse_dates=True)
print(ch.index)
mpf.plot(ch, type='candle', savefig='data/chart1m.png')