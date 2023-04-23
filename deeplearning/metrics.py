import mplfinance as mpf
import numpy as np
import pandas as pd

# from datetime import timedelta as td

def metrics(chart_before_trade: pd.DataFrame, chart_after_trade: pd.DataFrame, is_volume: bool, is_long: bool, pos_rate: float, tp_rate: float, is_make_chart: bool=False):
    """
    chart_before_trade, chart_after_trade:
        colums = ['Open', 'High', 'Low', 'Close', 'Volume']
        index : datetime
    """
    chart = pd.concat([chart_before_trade, chart_after_trade])
    pos_signal = [np.nan] * 200
    pos_signal[179] = chart.iloc[179]['Close']
    # lines = dict(hlines=[pos_rate], colors=['black'], linestyle='-')
    tp_time, tp_rate = calc_tp(chart_after_trade, pos_rate) # type: ignore
    if is_make_chart:
        if not tp_rate:
            tp_time = chart_after_trade.iloc[-1].name
            tp_rate = chart_after_trade.iloc[-1]['Close']
        res_price = (tp_rate - pos_rate) * (int(is_long) * 2 - 1)
        tp_index = int((tp_time - chart_before_trade.iloc[0].name).seconds / 60) # type: ignore
        tp_signal = [np.nan] * 200
        tp_signal[tp_index] = tp_rate
        lines = dict(hlines=[pos_rate, tp_rate], colors=['black', 'blue'], linestyle='-.')
        if is_long:
            apds = [mpf.make_addplot(pos_signal, type='scatter', markersize=200, marker='^'), 
                    mpf.make_addplot(tp_signal, type='scatter', markersize=200, marker='v')]
        else:
            apds = [mpf.make_addplot(pos_signal, type='scatter', markersize=200, marker='v'), 
                    mpf.make_addplot(tp_signal, type='scatter', markersize=200, marker='^')]
        mpf.plot(chart, addplot=apds, hlines=lines, type='candle', volume=is_volume, savefig='trade_result.png', title=str(res_price))
        # print(chart_before_trade)
    return tp_time, tp_rate


def calc_tp(chart_after_trade, pos_rate):
    tp_time = None
    tp_rate = None
    for ohcl in chart_after_trade.iterrows():
        if pos_rate - ohcl[1]['Low'] >= 0.100:
            tp_rate = pos_rate - 0.100
            tp_time = ohcl[0]
        elif ohcl[1]['High'] - pos_rate >= 0.100:
            tp_rate = pos_rate + 0.100
            tp_time = ohcl[0]
    return tp_time, tp_rate


if __name__ == '__main__':
    """ test """
    chart = pd.read_csv('/raidc/m-kawabt/system_trade/data/USDJPY/raw/DAT_XLSX_USDJPY_M1_202302.csv', header=None, index_col=0, parse_dates=True)
    chart_before_trade = chart[0:180]
    chart_after_trade = chart[180:200]
    chart_before_trade.index.name = 'Time'
    chart_before_trade.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    chart_before_trade = chart_before_trade.drop('Volume', axis=1)
    chart_after_trade.index.name = 'Time'
    chart_after_trade.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    chart_after_trade = chart_after_trade.drop('Volume', axis=1)

    pos_rate = chart_before_trade.iloc[-1]['Close']
    is_long = False
    is_volume = False
    tp_rate = chart_after_trade.iloc[-1]['Close']
    is_make_chart = True

    metrics(chart_before_trade=chart_before_trade, chart_after_trade=chart_after_trade, is_volume=is_volume, is_long=is_long, pos_rate=pos_rate, tp_rate=tp_rate, is_make_chart=is_make_chart)