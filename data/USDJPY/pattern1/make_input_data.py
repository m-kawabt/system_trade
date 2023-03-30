import csv
import glob
import os
from datetime import datetime as dt
from datetime import timedelta as td

file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/raw'
files = sorted(glob.glob(os.path.join(file_path, '*')))
dest_file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/pattern1/dataset/ground_truth.txt'
gt_list = []

for file_name in files:
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
        cluster = []
        for l in lines:
            l = l.split(',')
            t = dt.strptime(l[0], '%Y-%m-%d %H:%M') + td(hours=14)
            if t.hour >= 3 and t.hour <= 8:
                continue
            oPen = l[1]
            high = l[2]
            low = l[3]
            close = l[4]
            if cluster:
                t_p = cluster[-1][0]
                time_lapse = t - t_p
                if time_lapse > td(minutes=1):
                    if time_lapse == td(minutes=2):
                        t_tmp = t - td(minutes=1)
                        price_tmp = cluster[-1][4]
                        cluster.append(list([t_tmp, price_tmp, price_tmp, price_tmp, price_tmp]))
                    else:
                        if len(cluster) > 200:
                            # 入力データの作成
                            for s in range(0, len(cluster) - 200):
                                data200 = cluster[s:s+200]
                                close = float(data200[179][4])
                                up = 0
                                down = 0
                                criterion = 0.100
                                label = 0
                                for d in data200[180:]:
                                    up = max(up, float(d[2]) - close)
                                    down = max(down, close - float(d[3]))
                                    if up >= criterion or down >= criterion:
                                        if up >= criterion:
                                            label = (0.100 - down) / 0.100
                                        elif down >= criterion:
                                            label = -(0.100 - up) / 0.100
                                        break
                                gt_list.append([data200[0][0], label])
                        cluster = []
            cluster.append(list([t, oPen, high, low, close]))
        with open(dest_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(gt_list)
        break