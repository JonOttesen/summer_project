import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

path = os.path.abspath(os.path.dirname(__file__))

class files(object):

    def __init__(self, folder_name):
        self.directory = folder_name

    def output(self):
        return None

    def year_indexes(self):
        indexes = []
        for i in range(len(self.df.iloc[:])):
            #print(df.iloc[i,1])
            if self.df.iloc[i,1] >= 1900:
                indexes.append(i)
        return indexes

    def opening(self, file_name):
        """
        Returns all the values from the Antall brukere column and all the years
        Array for Antall brukere (x,y) size where x index represent a year and y index represent a age group.
        """
        os.chdir(self.directory)
        self.df = pd.read_excel(io=file_name, sheet_name = 0)
        os.chdir(path)

        indexes = self.year_indexes()

        d_indexes = int(indexes[1]-indexes[0])
        users = np.zeros(shape = (len(indexes), d_indexes))
        years = np.zeros(len(indexes))
        age_groups = []
        places = []
        genders = []

        for g in range(indexes[0], indexes[0] + d_indexes):
            age_groups.append(self.df.iloc[g,2])
            places.append(self.df.iloc[g,4])
            genders.append(self.df.iloc[g,3])

        for k in range(len(indexes)):
            years[k] = (self.df.iloc[indexes[k],1])
            counter = 0

            for j in range(indexes[k], indexes[k]+d_indexes):
                use = self.df.iloc[j,6]
                if type(use) != type(2):
                    use = 0
                users[k,counter] = use
                counter += 1


        return years, users, age_groups, places, genders


    def files_dict(self, file_name):

            os.chdir(self.directory)
            self.df = pd.read_excel(io=file_name, sheet_name = 0)
            os.chdir(path)

            indexes = self.year_indexes()
            users_dict = {}
            a = []

            d_indexes = int(indexes[1]-indexes[0])
            years, users, age_groups, places, genders = self.opening(file_name)
            users1 = np.concatenate(users)
            counter = 0

            for gender in np.unique(genders):

                for year in years:

                    for place in np.unique(places):

                        for age_group in age_groups:
                            a.append({gender: {year: {place: {age_group: users1[counter]}}}})
                            counter += 1
            print(a[0]['Kvinne'])


data1 = files('part_1')
(data1.files_dict('antiepileptika.xls'))


"""
plt.bar(years2, sum2/np.sum(sum2))
plt.bar(years1, sum1/np.sum(sum1))
plt.show()
"""
