import random

import mplfinance as mpf
import numpy as np
# import pandas as pd
import torch
import torch.backends.cudnn
from dataset import FXDataset
from torch.utils import data
from tqdm import tqdm

# from datetime import timedelta as td

def metrics(pos_rate, chart_after_trade, output):
    """
    is_long: 1:long, 0:no_pos, -1:short
    """
    tp_rate = None
    is_long = 0
    if output[0][0] > 0.5:
        if output[1][0] > 0.5:
            is_long = 1
        else:
            is_long = -1
    if is_long != 0:
        for ohlc in chart_after_trade:
            # -0.100
            if ohlc[3] <= pos_rate - 0.100:
                tp_rate = (ohlc[3] - pos_rate) * is_long
            # +0.100
            elif ohlc[3] >= pos_rate + 0.100:
                tp_rate = (ohlc[3] - pos_rate) * is_long
        if not tp_rate:
            tp_rate = (chart_after_trade[-1][3] - pos_rate) * is_long
    return tp_rate


# def make_chart():
#     lines = dict(hlines=[pos_rate], colors=['black'], linestyle='-')
#     if is_make_chart:
#         if not tp_rate:
#             tp_time = chart_after_trade.iloc[-1].name
#             tp_rate = chart_after_trade.iloc[-1]['Close']
#         res_price = (tp_rate - pos_rate) * (int(is_long) * 2 - 1)
#         tp_index = int((tp_time - chart_before_trade.iloc[0].name).seconds / 60) # type: ignore
#         tp_signal = [np.nan] * 200
#         tp_signal[tp_index] = tp_rate
#         lines = dict(hlines=[pos_rate, tp_rate], colors=['black', 'blue'], linestyle='-.')
#         if is_long:
#             apds = [mpf.make_addplot(pos_signal, type='scatter', markersize=200, marker='^'), 
#                     mpf.make_addplot(tp_signal, type='scatter', markersize=200, marker='v')]
#         else:
#             apds = [mpf.make_addplot(pos_signal, type='scatter', markersize=200, marker='v'), 
#                     mpf.make_addplot(tp_signal, type='scatter', markersize=200, marker='^')]
#         mpf.plot(chart, addplot=apds, hlines=lines, type='candle', volume=is_volume, savefig='trade_result.png', title=str(res_price))


if __name__ == '__main__':
    """ test """
    # 乱数を固定
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    # データローダーを作成
    gt_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/ground_truth.csv'
    data_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/input.csv'
    dataset = FXDataset(gt_file_path, data_file_path)
    train_size = int(dataset.__len__() * 0.8)
    test_size = dataset.__len__() - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    batch_size = 1
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for input_data, result_data, targets in tqdm(train_data_loader, desc='train', ncols=80):
        print(input_data[-1])
        exit(0)
        metrics = metrics(pos_rate=input_data[-1], chart_after_trade=result_data, is_volume=False, is_long=False)