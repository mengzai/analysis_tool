import argparse
import pandas as pd
import numpy
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',type=str,default='xxd_data_180',help='test data in csv format')
parser.add_argument('--col_name',type=str,default='latest_month_income',help='column name')
parser.add_argument('--bin_point',type=list,default=[0.0, 0.0, 1558.9000000000001, 2791.75, 4649.0, 6183.0, 401340866.0],help='bin_point')
parser.add_argument('--logodds',type=list,default=[1.2395485214574768, 1.3361965241371139, 1.4171953682299723, 1.3020846027430475, 1.0923150386390732, 0.7914433596537472],help='logodds')
parser.add_argument('--null_logodds',type=float,default=1.50595556452,help='null_logodds')
args = parser.parse_args()


#add a bin_point, logodds, null_logodds(default is NULL) in the input!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if __name__ == "__main__":

    # load args
    data_name = args.data_name
    col_name = args.col_name
    bin_point=args.bin_point
    logodds=args.logodds
    null_logodds=args.null_logodds

    data = pd.read_csv(data_name)
    data_notnull = data[-data[col_name].isnull()]
    col = list(data_notnull[col_name])

    # put the new bin_point here ######################################################################################
    #bin_point=[0.0, 757.83000000000004, 1600.0, 1900.0, 2162.6999999999998, 2413.0, 2664.6700000000001, 2928.3299999999999, 3206.75, 3529.5, 3941.6700000000001, 4566.6700000000001, 5693.8699999999999, 8098.6700000000001, 12066.67, 17646.0, 25700.0, 38298.0, 61141.5, 26078965584.990002]
    #logodds=[1.6868587262711943, 1.6307833182237756, 1.5872677887420772, 1.4358109118995122, 1.4020342994103514, 1.3446094982479277, 1.3128382123794151, 1.3293220803656354, 1.2968766784513208, 1.3284647059567842, 1.3340823054442075, 1.2872433202804823, 1.0434280195177943, 0.8002143372927188, 0.7724870948891561, 0.7121330674496678, 0.6858476194281334, 0.6964839484139964, 0.7359039219681274]
    #null_logodds=1.50595556452

    for i in range(0, len(bin_point) - 1):
        if (bin_point[i + 1] == bin_point[i]):
            bin_point[i+1] = bin_point[i+1]+0.000001
    for i in range(1, len(bin_point)):
        bin_point[i] = bin_point[i] + 0.000001

    pos = numpy.digitize(col, bin_point)
    data['bin'] = None
    data.loc[-data[col_name].isnull(), 'bin'] = pos
    data.loc[data['bin'] > len(bin_point)-1, 'bin'] = len(bin_point)-1



    data['odds'] = None
    data.loc[-data['bin'].isnull(), 'odds'] = [logodds[i - 1] for i in data.loc[-data['bin'].isnull(), 'bin']]
    if (sum(data[col_name].isnull()) > 0):
        data.loc[data[col_name].isnull(), 'odds'] = null_logodds

    #"""
    # output the odds
    csv_name = col_name + ".csv"
    f = open(csv_name, "w")
    data['odds'].to_csv(csv_name)
    f.close()
    #"""

"""
def main():
    odds_transform('xxd_data_180','month_income')
if __name__ == "__main__":
    main()
"""
