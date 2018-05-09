import argparse
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt
from compiler.ast import flatten

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='data',help='training data in csv format')
args = parser.parse_args()

if __name__ == "__main__":

    # load args
    data_name = args.data_name
    data_all = pd.read_csv(data_name)
    fea_list = data_all.columns
    total=len(data_all)
    sum_val=[]
    baseline = math.log(sum(data_all['label'] == 1) * 1.0 / (len(data_all) - sum(data_all['label'] == 1)))


    for k in range(1,len(fea_list)):
        col_name=fea_list[k]
        data = data_all[[col_name, 'label']]
        data_notnull = data[-data[col_name].isnull()]
        col = list(data_notnull[col_name])
        sorted_col = sorted(data_notnull[col_name])
        index = numpy.argsort(data_notnull[col_name])
        label = data_notnull.iloc[index, 1]
        label = list(label)
        #baseline = math.log(sum(data['label'] == 1) * 1.0 / (len(data) - sum(data['label'] == 1)))
        #print 'baseline',baseline
        # """
        ##########################################  bin_point  #####################################################
        # set the number of bins
        num_bin = 20
        min_num = int(len(data_notnull) * 1.0 / num_bin)
        # bin_point is bin
        bin_point = [sorted_col[0]]
        # index1 is the location of bin point
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
        # if the last bin is too small, combine it with the previous one
        if (len(data_notnull) - 1 - index1[-1] < min_num and index1[-1] != len(data_notnull) - 1):
            bin_point.pop(-1)
            index1.pop(-1)
        # add the last point to the binpoint
        if (index1[-1] != len(data_notnull) - 1):
            index1.append(len(data_notnull) - 1)
            bin_point.append(sorted_col[-1])
        ################################################## calc odds ##########################################################
         # group of value
        group = [sorted_col[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        # group of label
        group_label = [label[index1[i] + 1: index1[i + 1] + 1] for i in range(0, len(index1) - 1)]
        index1[0] = index1[0] + 1
        bin_point = list(pd.Series(sorted_col)[index1])
        #for i in range(len(group_label)):
            #print group_label[i]
        bin_N_good = [[len(group[i]), sum(group_label[i])] for i in range(0, len(index1) - 1)]
        logodds = [math.log(bin_N_good[i][1] * 1.0 / (bin_N_good[i][0] - bin_N_good[i][1])) for i in
                       range(0, len(index1) - 1)]
        #ratio: #of each bin / total count
        num_ratio=[len(group[i])*1.0/total for i in range(len(group))]
        if (sum(data[col_name].isnull()) > 0):
            null_N_good = [sum(data[col_name].isnull()),
                           sum(map(lambda x, y: x & y, data[col_name].isnull(), data['label'] == 1))]
            null_logodds = math.log(null_N_good[1] * 1.0 / (null_N_good[0] - null_N_good[1]))
            logodds.append(null_logodds)
            num_ratio.append(null_N_good[0]*1.0/total)

        m = len(logodds)
        abs_diff = [abs(logodds[i] - baseline)*num_ratio[i] for i in range(0, m)]
        sum_cur = sum(abs_diff) * 1.0 / m
        sum_val.append(sum_cur)


    #print out result
    sorted_sum_val=sorted(sum_val)
    index3 = numpy.argsort(sum_val)
    fea_list2=fea_list[1:]
    sorted_fea_list = list(pd.Series(fea_list2)[index3])
    output_data=pd.DataFrame(sorted_fea_list,sorted_sum_val)
    f = open('compare.csv', "w")
    output_data.to_csv('compare.csv')
    f.close()

    #make plot
    x=range(len(sum_val))
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, sorted_sum_val, 'b*')
    plt.plot(x, sorted_sum_val, 'r')
    plt.xlabel('feature')
    plt.ylabel('sum_log(odds)_diff')
    plt.xticks(x)
    fig.savefig('compare.png', dpi=180)
