import pandas as pd

def metrics(chart_before_trade: pd.DataFrame, chart_after_trade: pd.DataFrame, is_long: bool, pos_rate: float):
    pass



if __name__ == '__main__':
    chart = pd.read_csv('/raidc/m-kawabt/system_trade/data/USDJPY/raw/DAT_XLSX_USDJPY_M1_202302.csv', header=None, index_col=0)
    chart_before_trade = chart[0:180]
    chart_after_trade = chart[180:199]
    print(chart[0:179])
