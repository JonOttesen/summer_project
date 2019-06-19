import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import xlrd
import time


path = os.path.abspath(os.path.dirname(__file__))

class files(object):

    def __init__(self, folder_name):
        self.directory = folder_name

    def output(self):
        return None

    def year_indexes(self):
        """
        Finds the indexes for all rows which contains a year written in i.e non empty
        """
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

    def ordering(self, array):
        array1, order = np.unique(array, return_index=True)
        counter = 0
        new_array = np.zeros_like(array1)
        for index in np.sort(order):
            new_array[counter] = array[index]
            counter += 1
        return new_array[new_array != 'nan']


    def files_dict(self, file_name):
        """
        Places all parameters in a single dictonary.
        Order in the dictionary: Gender -> Year -> Age group -> Place (fylke/region/hele landet).
        """
        os.chdir(self.directory)

        wb = xlrd.open_workbook(file_name, logfile=open(os.devnull, 'w'))  #39.58 sek on test
        self.df = pd.read_excel(wb, engine='xlrd')
        #self.df = pd.read_excel(io = file_name, sheet = 0)  #38.97 on test

        os.chdir(path)

        indexes = self.year_indexes()
        users_dict = {}
        a = []

        d_indexes = int(indexes[1]-indexes[0])
        years, users, age_groups, places, genders = self.opening(file_name)
        users1 = np.concatenate(users)

        age_groups = self.ordering(age_groups)
        genders = self.ordering(genders)
        places = self.ordering(places)
        years = np.array(years)

        counter = 0
        for gender in genders:  #Looping through the different parameters
            users_dict[gender] = {}

            for year in years:
                users_dict[gender][year] = {}

                for age_group in age_groups:
                    users_dict[gender][year][age_group] = {}

                    for place in places:
                        users_dict[gender][year][age_group][place] = users1[counter]
                        counter += 1
        return users_dict


class visualization(object):

    def __init__(self, folder_name):
        self.folder_name = folder_name

        os.chdir(self.folder_name)
        self.filenames = os.listdir()
        os.chdir(path)

        self.data = []
        self.drugs = []  #Name of the drugs
        for i in self.filenames:
            self.drugs.append(i[:-4])

        for filename in self.filenames:
            self.data.append(files(folder_name).files_dict(filename))

        self.gender_keys = list(self.data[0].keys())
        self.year_keys = list(self.data[0][self.gender_keys[0]].keys())
        self.age_group_keys = list(self.data[0][self.gender_keys[0]][self.year_keys[0]].keys())
        self.places = list(self.data[0][self.gender_keys[0]][self.year_keys[0]][self.age_group_keys[0]].keys())


    def age_parameters(self, age_start, age_end):
        age_indexes = []
        for i in range(0, len(self.age_group_keys)):
            if age_start < 5*(i+1) and age_end > 5*(i):
                age_indexes.append(i)
        #for i in age_indexes:
        #    print(self.age_group_keys[i], age_start, age_end)
        return age_indexes

    def time_evolution(self, f, data, years):
        return None

    def part1(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2019):

        data = np.zeros((len(self.drugs), len(self.year_keys)))
        age_indexes = self.age_parameters(age_start, age_end)
        #print(self.age_group_keys[age_indexes[0]], self.age_group_keys[age_indexes[-1]])

        for i in range(len(self.drugs)):
            for k in range(len(self.year_keys)):
                for j in range(len(age_indexes)):
                    data[i, k] += self.data[i][gender][self.year_keys[k]][self.age_group_keys[age_indexes[j]]][region]


        time = np.linspace(period_start, period_end, period_end-period_start+1)
        for i in range(len(self.drugs)):
            z = np.polyfit(self.year_keys, data[i], 4)
            f = np.poly1d(z)
            plt.plot(self.year_keys, data[i], 'ro')
            plt.plot(time, f(time))
        plt.show()







#files('Antiepileptika').files_dict('Lamotrigin.xls')

test = visualization('Antiepileptika')
test.part1()

os.chdir(path)





































#jao
