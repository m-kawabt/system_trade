import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
from dataset import FXDataset
from model import MyModel0
from torch.utils import data
from tqdm import tqdm


def train(max_epoch=100):
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
    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # モデル
    net = MyModel0()

    # 重みを初期化
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    net.modulelist.apply(weights_init)

    # GPU 設定
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス：', device)
    print('ネットワーク設定完了')

    # 最適化手法
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


    for epoch in range(max_epoch):
        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, max_epoch))

        # モデルを訓練モードに
        # net.train()

        for input_data, targets in tqdm(train_data_loader, desc='train', ncols=80):
            print(input_data)
            print(targets)


if __name__ == '__main__':
    train(1)