from datetime import datetime as dt
from datetime import timedelta as td

import pandas as pd
from torch.utils import data


class FXDataset(data.Dataset):
    def __init__(self, gt_file_path, data_file_path):
        self.gt_df = pd.read_csv(gt_file_path, header=None)
        self.data_df = pd.read_csv(data_file_path, header=None, index_col=0)

    def __len__(self):
        return len(self.gt_df)

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        gt = self.gt_df.iloc[index]
        start_time = dt.strptime(gt[0], '%Y-%m-%d %H:%M:%S')
        end_time = start_time + td(minutes=179)
        label = gt[1]
        input = self.data_df.loc[start_time.strftime('%Y-%m-%d %H:%M:%S') : end_time.strftime('%Y-%m-%d %H:%M:%S')]
        return input, label



if __name__ == '__main__':
    test_gt_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/ground_truth.csv'
    test_data_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/input.csv'
    dataset = FXDataset(test_gt_file_path, test_data_file_path)