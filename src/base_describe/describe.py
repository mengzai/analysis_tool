#encoding=utf-8

import csv
import scipy.stats as stats
import argparse
import pandas as pd
import time
import sys

def load_data(path):
    return pd.read_csv(path, error_bad_lines=False)

def des(data,start,savename ):
    decrib = ['feature', 'count（not nullall_count:761539', 'min', 'max', 'mean', '方差', '偏度', '峰度',
              '25%', '50%', '75%', '90%', '99.7%', '99.97%所占人数', '覆盖率', '划分（针对离散型变量）"	含义	"连续1:离散：0', 'meaning']
    file0 = open(savename, 'wb+')  # 'wb'
    output = csv.writer(file0, dialect='excel')
    output.writerow(decrib)
    total = len(data)
    fea_list = data.columns
    print fea_list

    for m in range(int(start), len(fea_list)):
        colom = fea_list[m]
        data_notnull = data[-data[colom].isnull()][colom]
        g_dist = sorted(data_notnull)
        lenth = len(g_dist)
        info = stats.describe(data_notnull)
        listdes = [colom, str(info[0]), str(info[1][0]), str(info[1][1]), str(info[2]),
                   str(info[3]), str(info[4]), str(info[5]), g_dist[int(0.25 * lenth)],
                   g_dist[int(0.5 * lenth)], g_dist[int(0.75 * lenth)], g_dist[int(0.9 * lenth)],
                   g_dist[int(0.9997 * lenth)], int(lenth - int(0.9997 * lenth)), float(int(info[0]) * 1.0 / total)]
        output.writerow(listdes)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        USAGE = "python describe.py data file_name start savename "
        print "[Error] Expect 6 args, but got %d." % (len(sys.argv) - 1)
        print USAGE
        sys.exit(1)
    data_dir = sys.argv[1]
    file_name = sys.argv[2]
    start = sys.argv[3]
    savename = data_dir + sys.argv[4]

    dataname = data_dir+file_name
    data = load_data(dataname)
    des(data, start ,savename)