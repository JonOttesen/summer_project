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
        """
        Places all parameters in a single dictonary.
        Order in the dictionary: Gender -> Age -> Place (fylke/region/hele landet) -> Aldersgruppe.
        """
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

        for gender in np.unique(genders):  #Looping through the different parameters
            users_dict[gender] = {}

            for year in years:
                users_dict[gender][year] = {}

                for place in np.unique(places):
                    users_dict[gender][year][place] = {}

                    for age_group in age_groups:
                        users_dict[gender][year][place][age_group] = users1[counter]
                        counter += 1
        return users_dict


class visualization(object):

    def __init__(self, folder_name):
        self.folder_name = folder_name
        os.chdir(self.folder_name)
        self.filenames = os.listdir()


visualization('part_1')

os.chdir(path)





































#jao
