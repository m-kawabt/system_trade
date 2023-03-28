import glob
import os
from datetime import datetime as dt


file_path = '/raidc/m-kawabt/system_trade/data/USDJPY/raw'
files = sorted(glob.glob(os.path.join(file_path, '*')))

for file_name in files:
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
        for l in lines:
            l = l.split(',')
            t = l[0]
            t = dt.strptime(t, '\ufeff%Y-%m-%d %H:%M')
            print(type(t), t)
        break