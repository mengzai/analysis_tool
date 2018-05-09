#encoding=utf-8
from __future__ import division
from matplotlib import rcParams
import os
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import os
import os.path
from matplotlib.lines import Line2D
import argparse
import numpy as np
import argparse
import pandas as pd
import math
import numpy
import csv
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_train_all',help='training data in csv format')
parser.add_argument('--fig_save_path', type=str, default='figtrainall/',help='the path of saving the figs')
parser.add_argument('--null_fig_save_path', type=str, default='nullfigtrainall/',help='the path of saving the figs')
parser.add_argument('--has_label', type=int, default=0, required=False, help='the start index of features in dataset(if no labels in dataset, this is 1; else this is 2)')
parser.add_argument('--feature_file', type=str, help='the file of feature list')
parser.add_argument('--feature_name', type=str, help= 'the feature name ')
parser.add_argument('--plot_null', type=int, default=1,help= 'the feature name ')
parser.add_argument('--savename',type=str,default='ks_score.csv',help='the sort of ksscore')
args = parser.parse_args()



"""
有三个参数必须设置
1) data_path : 指定数据路径
2) has_label : 指定特征的起始位置. 如果数据中有label, 则has_label=1; 如果没有label, 则has_label=1
3:plot_null   :是否画出null    ,plot_null=1 表示画出null图  ,plot_null=0表示不画null  且保存路径也不同

注意:
1) fig_save_path 这个最好设置, 不然图像默认放到当前路径的fig/目录下. 如果图像很多不容易归档,表示不包含nul的ks图
2:null_fig_save_path:表示包含bull的画图
3) feature_file feature_name不可同时设置


举例:
1) 绘制全部特征(不设置feature_file和feature_name)
python plot_features_ks.py --data_path datasets1126/all_baoxiao_safe_1 --has_label 0   --fig_save_path "fig/fig_baoxiao_safe_1"

2) 绘制指定feature_file 中的特征
python plot_features_ks.py --data_path datasets1126/all_baoxiao_safe_1 --has_label 0   --feature_file "fea_file" --fig_save_path "fig/fig_baoxiao_safe_1"

3) 绘制指定feature_name 的特征
python plot_features_ks.py --data_path datasets1126/33w --has_label 0   --feature_name "n1" --fig_save_path "fig/fig_baoxiao_safe_1"



"""
# get the line from the result_list
#input good_ks_result_list,bad_ks_result_list,feature_name,max_ks_gap_index,fig_save_path
#out_put
def plot_features(good_ks_result_list,bad_ks_result_list,feature_name,max_ks_gap_index,fig_save_path):
    length = len(good_ks_result_list)
    index = range(0, length)
    labels = index
    title_name=feature_name
    fig = plt.figure(figsize=(10, 10))
    axes = fig.gca()
    axes.plot(index, good_ks_result_list, 'r', linewidth=2, label='bad')
    axes.plot(index, bad_ks_result_list, 'g', linewidth=2, label='good')
    axes.legend()
    axes.set_xticklabels(labels)
    axes.set_title(title_name)
    max_ks_gap_good_value = good_ks_result_list[max_ks_gap_index]
    max_ks_gap_bad_value = bad_ks_result_list[max_ks_gap_index]
    annotate_text_y_index = abs(max_ks_gap_bad_value - max_ks_gap_good_value) / 2 + \
                            min(max_ks_gap_good_value, max_ks_gap_bad_value)
    max_ks_gap_value = abs(max_ks_gap_bad_value - max_ks_gap_good_value)
    axes.annotate(str(round(max_ks_gap_value, 2)), xy=(max_ks_gap_index, annotate_text_y_index))
    xytext_value = str(max_ks_gap_index)
    axes.annotate(xytext_value, xy=(max_ks_gap_index, 0), xytext=(max_ks_gap_index, 0.05),
                  arrowprops=dict(facecolor='red',
                                  shrink=0.05))  # , arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    axes.plot([max_ks_gap_index, max_ks_gap_index],[bad_ks_result_list[max_ks_gap_index], good_ks_result_list[max_ks_gap_index]], linestyle='--',linewidth=2.5)
    fig.savefig(str(fig_save_path) + str(feature_name) + '_'  + '.png', dpi=180)
    plt.close(fig)
    plt.show()

#get ks_valu from bins:
#input bin_N_good:[good_people,bad_people]    null_N_good:[good_people,bad_people ], index1
#output :
def bin_ks(bin_N_good,null_N_good,index1,plot_null):
    #画包含null值的
    # print null_N_good,bin_N_good
    Base = []
    good_ks_result_list = []
    bad_ks_result_list = []
    ind1 = 0
    ind2 = 0
    good_people_num=0
    bad_people_num=0
    for q in bin_N_good:
        good_people_num += q[0]
        bad_people_num += q[1]

    if plot_null==1:
        # if null_N_good==[]:
        #     bin_label_list=bin_N_good
        # else:
        #     bin_label_list = null_N_good+bin_N_good
        if null_N_good==[]:
            ind1 = 0
            ind2 = 0
        else:
            good_people_num +=null_N_good[0]
            bad_people_num += null_N_good[1]
            ind1 = null_N_good[0] * 1.0 / good_people_num
            ind2 = null_N_good[1] * 1.0 / bad_people_num
            good_ks_result_list.append(ind1)
            bad_ks_result_list.append(ind2)
    else:
        pass

    for j in range(len(bin_N_good)):
        ind1 += bin_N_good[j][0] * 1.0 / good_people_num
        ind2 += bin_N_good[j][1] * 1.0 / bad_people_num
        good_ks_result_list.append(ind1)
        bad_ks_result_list.append(ind2)

    # print null_N_good,bin_N_good
    # print bad_ks_result_list
    Max=[]
    max_ks_gap=0
    max_ks_gap_index=0
    for m in range(len(good_ks_result_list)):
        absmax=abs(good_ks_result_list[m]-bad_ks_result_list[m])
        Max.append(absmax)
        if absmax>max_ks_gap:
            max_ks_gap=absmax
            max_ks_gap_index=m
    return good_ks_result_list,bad_ks_result_list,max_ks_gap_index,max_ks_gap


#find bins and good bad man in every bin
#input data,columns
#output everybin ;   bin_N_good:[good_people,bad_people]    null_N_good:[good_people,bad_people ], index1
def Bin(data_all,col_name):
    data = data_all[[col_name, 'label']]
    data_notnull = data[-data[col_name].isnull()]
    sorted_col = sorted(data_notnull[col_name])
    index = numpy.argsort(data_notnull[col_name])
    label = data_notnull.iloc[index, 1]
    label = list(label)

    # """
    ##########################################  bin_point  #####################################################
    # set the number of bins
    num_bin = 20
    min_num = int(len(data_notnull) * 1.0 / num_bin)
    bin_point = [sorted_col[0]]
    index1 = [0]
    i = 0

    while i < len(data_notnull):
        if (len(data_notnull) - i > min_num):
            i = i + min_num
            tmp = sorted_col[i]
            for j in range(i + 1, len(data_notnull)):
                if (sorted_col[j] == tmp):
                    j = j + 1
                else:
                    tmp = sorted_col[j - 1]
                    i = j - 1
                    index1.append(j - 1)
                    bin_point.append(tmp)
                    break
        else:
            break
    if (len(data_notnull) - 1 - index1[-1] < min_num and index1[-1] != len(data_notnull) - 1):
        bin_point.pop(-1)
        index1.pop(-1)
    if (index1[-1] != len(data_notnull) - 1):
        index1.append(len(data_notnull) - 1)
        bin_point.append(sorted_col[-1])

    group = [sorted_col[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
    # group of label
    group_label = [label[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]

    index1[0] = index1[0] + 1
    bin_point = list(pd.Series(sorted_col)[index1])
    data['odds'] = None
    null_N_good=[]
    if (sum(data[col_name].isnull()) > 0):
        null_N_good = [sum(data[col_name].isnull()) - sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 0)),
                       sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 0))]
    bin_N_good = [[sum(group_label[i]), len(group[i]) - sum(group_label[i])] for i in range(0, len(index1) - 1)]
    return bin_N_good,null_N_good,index1

# 检查参数
def check_arguments(data_path, fig_save_path, feature_file=None):
    """检查data_path是否存在，如果不存在抛出错误; 检查fig_save_path 是否存在，如果不存在创建
    """
    if not os.path.exists(data_path):
        raise NameError("================The input data path dose not exist==============")

    if not os.path.exists(fig_save_path):
        print "The path of saving figs dose not exits. "
        os.makedirs(fig_save_path)
        print "The path of saving figs has been created"

    if fig_save_path[-1] != '/':
        fig_save_path += '/'

    if feature_file:
        if not os.path.exists(feature_file):
            raise NameError("===The input feature list dose not exist==============")
    return data_path, fig_save_path

#load data and columns
def load_data():
    data_name = args.data_name
    data_all = pd.read_csv(data_name)
    return data_all

# 从输入文件中读取所有特征的名称
def gen_feature_names_from_data(data_path, has_label=2):
    with open(data_path,'rb') as f:
        feature_name_list = []
        for line in f.readlines():
            if line == "":
                continue
                # print "line_is_empty"
                # exit()
            tokens = line.strip().split('\t')
            for feature in tokens[has_label].split(','):
                f_tokens = feature.split(':')
                feature_name_list.append(f_tokens[0])
            feature_name_list.pop(0)
            return feature_name_list

# 从一个指定文件中获取特征名称
def gen_feature_name_from_file(file):
    with open(file) as f:
        feature_name_list = []
        for line in f.readlines():
            if line == "":
                continue
            feature_name_list.append(line.strip())
    return feature_name_list

if __name__ == "__main__":
    # load args
    plot_null = args.plot_null
    if plot_null==1:
        fig_save_path = args.null_fig_save_path
    else:
        fig_save_path = args.fig_save_path
    feature_file = args.feature_file
    feature_name = args.feature_name
    has_label=args.has_label
    data_name=args.data_name
    plot_null=args.plot_null
    savename=args.savename
    file0 = open(savename, 'wb+')  # 'wb'
    output = csv.writer(file0, dialect='excel')

    feature_name_list = []
    data_path, fig_save_path = check_arguments(data_name, fig_save_path, feature_file)
    if feature_file is None and feature_name is None:
        print "===feature file and feature name are not assigned, so all features will be draw==="
        feature_name_list = gen_feature_names_from_data(data_path, has_label)
        # feature_name_list = gen_feature_names_from_data(data_path, has_label)
    elif feature_file and feature_name:
        raise NameError("========Cannot assign feature file and feature name both==========")
    elif feature_file:
        feature_name_list = gen_feature_name_from_file(feature_file)
    else:
        feature_name_list.append(feature_name)
    data_all=load_data()
    for col_name in feature_name_list:
        bin_N_good, null_N_good, index1= Bin(data_all,col_name)
        good_ks_result_list, bad_ks_result_list, max_ks_gap_index, max_ks_gap=bin_ks(bin_N_good,null_N_good,index1,plot_null)
        output.writerow([col_name,max_ks_gap])
        plot_features(good_ks_result_list, bad_ks_result_list, col_name, max_ks_gap_index, fig_save_path)

