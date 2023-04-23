from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np

import pandas as pd
import torch
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
        # index = 361
        gt = self.gt_df.iloc[index]
        input_start_time = dt.strptime(gt[0], '%Y-%m-%d %H:%M:%S')
        input_end_time = input_start_time + td(minutes=179)
        result_start_time = input_start_time + td(minutes=180)
        result_end_time = input_start_time + td(minutes=199)
        input_df = self.data_df.loc[input_start_time.strftime('%Y-%m-%d %H:%M:%S') : input_end_time.strftime('%Y-%m-%d %H:%M:%S')]
        input_tensor = torch.tensor(np.array(input_df), dtype=torch.float)
        input_tensor_transformed = torch.reshape(input_tensor - input_tensor[-1][-1], [1, -1]).squeeze(0)
        result_df = self.data_df.loc[result_start_time.strftime('%Y-%m-%d %H:%M:%S') : result_end_time.strftime('%Y-%m-%d %H:%M:%S')]
        # print(len(result_df))
        result_tensor = torch.tensor(np.array(result_df), dtype=torch.float)
        result_tensor_transformed = torch.reshape(result_tensor - result_tensor[-1][-1], [1, -1]).squeeze(0)
        target_tensor = torch.tensor(gt[1], dtype=torch.float)
        return input_tensor_transformed, result_tensor_transformed, target_tensor

# class DataTransformer():
#     def __init__(self):
#         pass
#     def __call__(self):
#         pass



if __name__ == '__main__':
    test_gt_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/ground_truth.csv'
    test_data_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/input.csv'
    dataset = FXDataset(test_gt_file_path, test_data_file_path)
    train_size = int(dataset.__len__() * 0.8)
    test_size = dataset.__len__() - train_size

    """test dataloader"""
    # train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    # train_data_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    # test_data_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    # # print(test_dataset.indices)
    # test_dataset.__getitem__(idx=0)[1]

    """test dataset"""
    dataset.pull_item(0)