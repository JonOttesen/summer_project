import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.dirname(__file__))

class files(object):

    def __init__(self):
        self.dirs = ['part_1']

    def opening(self, file_name, earliest_year = 1950):
        os.chdir(self.dirs[0])
        sheet = 0
        df = pd.read_excel(io=file_name, sheet_name=sheet)
        indexes = []
        years = []

        for i in range(len(df.iloc[:])):
            #print(df.iloc[i,1])
            if df.iloc[i,1] >= earliest_year:
                indexes.append(i)

        d_indexes = int(indexes[1]-indexes[0])
        sums = np.zeros_like(indexes)
        counter = 0
        for k in indexes:
            years.append(df.iloc[k,1])
            for j in range(k, k+d_indexes):
                sums[counter] += df.iloc[j,6]

            counter +=1
        os.chdir(path)
        return np.array(years), sums



"""
file_name =  # path to file + file name
sheet =  # sheet name or sheet number or list of sheet numbers and names

df = pd.read_excel(io=file_name, sheet_name=sheet)
print(df.head(5))  # print first 5 rows of the dataframe
"""

data1 = files()
years1 , sum1 = data1.opening('antiepileptika.xls')

data2 = files()
years2 , sum2 = data2.opening('Lamotrigin.xls')


plt.plot(years1, sum2/sum1)
plt.show()
