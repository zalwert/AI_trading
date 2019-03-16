
'''

Data downloaded from:
"http: // www.histdata.com / download - free - forex - data /? / excel / 1 - minute -  - quotes"

Column Fields:
DateTime Stamp; OPEN Bid Quote; HIGH Bid Quote; LOW Bid Quote; CLOSE Bid Quote;Volume

License: for education purposes only.

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot
import time
from sklearn.model_selection import train_test_split


def timeit(met):
    """
    decorator to measure the script time

    :param met: function
    :return: time
    """

    def timer(*args, **kw):

        ts = time.time()
        result = met(*args, **kw)
        te = time.time()
        print("\n\n")
        print("Downloading data took : %5.2f secs" % (te-ts))
        return result

    return timer

def FKI_check1(x):
    """
    Currently not i use

    :param x: pandas row data
    :return: positive int
    """

    if x >= 0:
        y = x
    else:
        y = 0
    return y

def FKI_check2(x):
    """
    Currently not i use

    :param x: pandas row data
    :return: negative int
    """
    if x <= 0:
        y = x
    else:
        y = 0
    return y

@timeit
class Get_data:

    def __init__(self, file_location, pips_range, time_range, mins):

        self.columns = ['DateTime_Stamp', 'OPEN_Bid', 'HIGH_Bid', 'LOW_Bid', 'CLOSE_Bid', 'Volume']
        self.data = pd.read_excel(file_location)
        self.data.columns = self.columns
        self.mean = self.data['OPEN_Bid'].mean()
        self.std = self.data['OPEN_Bid'].std()
        self.max = self.data['OPEN_Bid'].max()
        self.min = self.data['OPEN_Bid'].min()
        self.mins = mins

        #missing info in source file
        self.data.drop(['Volume'], axis=1, inplace=True)

        self._add_BollingerBands(mins)
        #self._add_diffs(mins)
        self._add_sok()
        self._add_Y_labels(pips_range, time_range)
        self._see()
        self._plot_data()

    def get_train_test(self, ts, rs):

        self.dataALL = self.data.copy()
        self.data.drop(['DateTime_Stamp', 'OPEN_Bid', 'HIGH_Bid', 'LOW_Bid', 'CLOSE_Bid'], axis=1, inplace=True)

        self.data_torch = self.data.copy()

        for col in self.data.columns:
            if col != 'Y_label':
                mm = self.data[col].mean()
                self.data[col] = self.data[col].apply(lambda x: 1 if x > mm else 0)

        x_train, x_test, y_train, y_test, = train_test_split(self.data.loc[:, self.data.columns != 'Y_label'],
                                                            self.data['Y_label'], test_size=ts, random_state=rs)

        x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(self.dataALL.loc[:, self.data.columns != 'Y_label'],
                                                             self.data['Y_label'], test_size=ts, random_state=rs)

        return x_train, x_test, y_train, y_test, self.data, self.data_torch, x_test_all

    def _add_BollingerBands(self, mins):

        for min_ in mins:

            self._add_MA(mins)
            self._add_STD(mins)
            self._add_diffs(mins)
            self._add_FKI(mins)

            self.data[str(min_) + 'U_Band'] = self.data[str(min_) + 'MIN_MA'] + (self.data[str(min_) + 'MIN_STD'] * 2)
            self.data[str(min_) + 'L_Band'] = self.data[str(min_) + 'MIN_MA'] - (self.data[str(min_) + 'MIN_STD'] * 2)

    def _add_MA(self, mins):

        # loop to add MA
        for min_ in mins:

                self.data[str(min_) + "MIN_MA"] = self.data['OPEN_Bid'].rolling(window=min_).mean()

        return self

    def _add_STD(self, mins):

        # loop to add STD
        for min_ in mins:

                self.data[str(min_) + "MIN_STD"] = self.data['OPEN_Bid'].rolling(window=min_).std()

        return self


    def _add_diffs(self, mins):

        for min_ in mins:
            self.data[str(min_) + "Diff"] = self.data['OPEN_Bid'] - self.data['OPEN_Bid'].shift(min_).fillna(0)
            self.data[str(min_) + "Diff_MA"] = self.data[str(min_) + "Diff"].rolling(window = min_).mean()

            for min2_ in range(min_):
                self.data[str(min_) + "Diff"][min2_] = 0

        return self

    def _add_sok(self):

        self.data['SOK'] = (self.data['CLOSE_Bid'] - self.data['LOW_Bid']) / (self.data['HIGH_Bid'] - self.data['LOW_Bid'])
        self.data['SOK'].fillna(0, inplace = True)

        return self

    def _add_FKI(self, mins):



        # FKI = 100 – [100 / (1 + (Average of Upward Price Change / Average of Downward Price Change))]
        for min_ in mins:

            self.data[str(min_) + "FKI"] = 100 - (100 / ( 1 + self.data[str(min_) + "Diff_MA"].apply(FKI_check1) / self.data[str(min_) + "Diff_MA"].apply(FKI_check2)))
            self.data[str(min_) + "FKI"].fillna(0, inplace = True)



        return self



    def _add_Y_labels(self, pips_range, time_range):

        # default = 25 pips
        self.pips_range = pips_range
        # default = 5 mins
        self.time_range = time_range
        self.ticks = self.data.shape[0] - time_range
        d_temp = []

        for i in range(self.ticks):
            d_temp.append(self.data['OPEN_Bid'][i+self.time_range] - self.data['OPEN_Bid'][i])
        self.meand = np.mean(d_temp)

        # Add mean for missing points in vector
        for x in range(time_range):
            d_temp.append(self.meand)

        self.data['d_temp'] = d_temp
        self.maxd = self.data['d_temp'].max()
        self.mind = self.data['d_temp'].min()

        # new_data = (old_data - x_min) / (x_max - x_min)
        self.data['Y_label'] = self.data['d_temp'].apply(lambda x: ((x - self.mind) / (self.maxd - self.mind)))
        self.data.drop(['d_temp'], axis=1, inplace=True)

    def _plot_data(self):

        for plot in ["MA", "STD"]:
            plt.figure()
            col_temp = [col1 for col1 in self.data.columns if str(plot) in col1]
            for col in col_temp:
                plt.plot(self.data['DateTime_Stamp'], self.data[col])

            plt.title(plot)
            plt.xticks(rotation=90)
            plt.subplots_adjust(bottom=0.2)
            plt.show()

        plt.figure()
        plt.hist(self.data['Y_label'], bins=np.arange(0.00, 1.01, 0.05))
        qqplot(self.data['Y_label'], line='s')

        plt.figure()
        plt.plot(self.data['DateTime_Stamp'], self.data["OPEN_Bid"])
        col_temp = [col1 for col1 in self.data.columns if "Band" in col1]
        for col in col_temp:
            plt.plot(self.data['DateTime_Stamp'], self.data[col])


        return self

    def _see(self):

        print("Columns: ", self.data.columns)
        print("# of columns: ", len(self.data.columns))
        print("Shape: ", self.data.shape)

        print("There were %5d trades above 90 percentiles trained in NN" % (len(self.data[self.data['Y_label'] > 0.9])))
        print("There were %5d trades above 80 percentiles trained in NN" % (len(self.data[self.data['Y_label'] > 0.8])))
        print("There were %5d trades above 70 percentiles trained in NN" % (len(self.data[self.data['Y_label'] > 0.7])))

        print("There were %5d trades below 10 percentiles trained in NN" % (len(self.data[self.data['Y_label'] < 0.1])))
        print("There were %5d trades below 20 percentiles trained in NN" % (len(self.data[self.data['Y_label'] < 0.2])))
        print("There were %5d trades below 30 percentiles trained in NN" % (len(self.data[self.data['Y_label'] < 0.3])))

if __name__ == "__main__":

	pass
	#restricted content
