import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
from dataset import FXDataset
from loss import MyLoss
from model import MyModel0
from torch.utils import data
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_value_

def train(max_epoch=100, batch_size=16):
    # 乱数を固定
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # GPU 設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス：', device)
    print('ネットワーク設定完了')
    
    # データローダーを作成
    gt_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/ground_truth.csv'
    data_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/input.csv'
    dataset = FXDataset(gt_file_path, data_file_path)
    train_size = int(dataset.__len__() * 0.8)
    test_size = dataset.__len__() - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
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

    # 損失関数
    criterion = MyLoss(device=device)
    criterion.to(device=device)
    
    # 最適化手法
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # モデルをGPUへ
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    net.to(device)

    for epoch in range(max_epoch):
        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, max_epoch))

        # モデルを訓練モードに
        # net.train()

        epoch_loss = 0.0

        for input_data, targets in tqdm(train_data_loader, desc='train', ncols=80):
            # データをGPUへ
            input_data = input_data.to(device)
            targets = targets.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(True):
                output = net(input_data)
                is_trade_loss, longshort_loss = criterion(output, targets)
                loss = is_trade_loss + longshort_loss
                # 勾配の計算 
                loss.backward()
                epoch_loss += loss
                # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                clip_grad_value_(net.parameters(), clip_value=2.0)
                # パラメータ更新
                optimizer.step()

        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), 'deeplearning/weights/mymodel_' + str(epoch+1) + '.pth')
            torch.save(net.state_dict(), 'deeplearning/weights/latest.pth')


        # epochのphaseごとのloss
        print('Epoch_TRAIN_Loss:{:.4f}'.format(epoch_loss))



if __name__ == '__main__':
    train(max_epoch=100, batch_size=128)