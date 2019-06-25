import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import xlrd
import time
import warnings
from scipy.optimize import curve_fit
import statsmodels.api as sm
import sys

warnings.simplefilter('ignore', np.RankWarning)

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
            self.drugs.append(i[:-4])  #Removing the .xls ending

        for filename in self.filenames:
            self.data.append(files(folder_name).files_dict(filename))  #Dictionaries for all the xls files in the folder

        self.gender_keys = list(self.data[0].keys())
        self.year_keys = list(self.data[0][self.gender_keys[0]].keys())
        self.age_group_keys = list(self.data[0][self.gender_keys[0]][self.year_keys[0]].keys())
        self.places = list(self.data[0][self.gender_keys[0]][self.year_keys[0]][self.age_group_keys[0]].keys())


    def age_parameters(self, age_start, age_end):
        age_indexes = []
        for i in range(0, len(self.age_group_keys)):
            if age_start < 5*(i+1) and age_end >= 5*(i):
                age_indexes.append(i)
        #for i in age_indexes:
        #    print(self.age_group_keys[i], age_start, age_end)
        return age_indexes

    def curve_fitting(self, data):
        degree = 1
        years = []
        new_data = []

        years.append(self.year_keys[0]-60)
        years.append(self.year_keys[0]-50)
        for i in self.year_keys:
            years.append(i)
        years.append(self.year_keys[-1]+60)
        years.append(self.year_keys[-1]+70)

        new_data.append(0)
        new_data.append(0)
        for i in data:
            new_data.append(i)
        new_data.append(0)
        new_data.append(0)
        data = np.array(new_data)

        z_lower = np.polyfit(years, data, 2)
        f_best_fit = np.poly1d(z_lower)
        for deg in [2,4,6,8,10]:
            z_higher = np.polyfit(years, data, deg)
            f_higher = np.poly1d(z_higher)
            if np.sum((f_higher(years[2:-2]) - data[2:-2])**2) <= np.sum((f_best_fit(years[2:-2]) - data[2:-2])**2):
                degree = deg
                f_best_fit = f_higher
        return f_best_fit

    def final_function(self, data, f, time):
        if type(time) == type([1]) or type(time) == type(np.array([1])):
            pass
        else:
            f = [f]
            time = [time]

        final = []
        for year in time:
            if year in self.year_keys:
                final.append(data[self.year_keys.index(year)])
            else:
                final.append(f[time == year])
        return np.array(final)

    def part1_plotting(self, data, period_start, period_end, drug_list, age_indexes, gender, region):

        time = np.linspace(period_start, period_end, period_end-period_start+1)
        bars = len(data)
        offset = 0.8/bars
        if len(age_indexes) > 1:
            alder = self.age_group_keys[age_indexes[0]] + ' til ' + self.age_group_keys[age_indexes[-1]]
        else:
            alder = self.age_group_keys[age_indexes[0]]

        if type(data[0]) == type(np.array([1])):
            for i in range(len(data)):
                func = self.curve_fitting(np.log(data[i]))
                plt.bar(time-0.4+i*offset, self.final_function(data[i], np.exp(func(time)), time), width = offset, label = drug_list[i])
                #plt.plot(time, np.exp(func(time)))
                plt.legend()
                plt.title(gender + ' i ' + region + ' alder ' + alder)
            plt.show()
        else:
            func = self.curve_fitting(np.log(data))
            plt.bar(time, self.final_function(data, np.exp(func(time)), time), label = drug_list)
            plt.legend()
            plt.title(gender + ' i ' + region + ' alder ' + alder)
            plt.show()

        return None

    def drug_array(self, age_indexes, region, gender):
        data = np.zeros((len(self.drugs), len(self.year_keys)))

        for i in range(len(self.drugs)):
            for k in range(len(self.year_keys)):
                for j in range(len(age_indexes)):
                    data[i, k] += self.data[i][gender][self.year_keys[k]][self.age_group_keys[age_indexes[j]]][region]

        return data

    def cake_plot(self, gender, region, data_drugs, data_tot_drugs, year, age_indexes, drug_list):

        if len(age_indexes) > 1:
            alder = self.age_group_keys[age_indexes[0]] + ' til ' + self.age_group_keys[age_indexes[-1]]
        else:
            alder = self.age_group_keys[age_indexes[0]]

        func = self.curve_fitting(np.log(data_tot_drugs))
        func_value = self.final_function(data_tot_drugs, np.exp(func(year)), year)
        x = []
        explosion = []

        for i in range(len(data_drugs)):
            func = self.curve_fitting(np.log(data_drugs[i]))
            x.append(self.final_function(data_drugs[i], np.exp(func(year)), year)/func_value)
            explosion.append(0.05)

        explosion.append(0.05)
        x.append(1-np.sum(np.array(x)))
        drug_list.append('Resterende '+ self.folder_name)

        plt.pie(x, explode = explosion, labels = drug_list, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title(gender + ' i ' + region + ' alder ' + alder + ' år ' + str(year))
        plt.show()

        return None


    def part1(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018):

        age_indexes = self.age_parameters(age_start, age_end)

        data = self.drug_array(age_indexes, region, gender)

        med_type_index = self.drugs.index(self.folder_name)
        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        ratio = data_drugs/total_use
        ratio_list = self.drugs[:]
        del ratio_list[med_type_index]

        self.part1_plotting(data, period_start, period_end, self.drugs, age_indexes, gender, region)
        self.part1_plotting(ratio, period_start, period_end, ratio_list, age_indexes, gender, region)
        return None


    def individual(self, drug, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018):

        age_indexes = self.age_parameters(age_start, age_end)
        med_index = self.drugs.index(drug)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        ratio = data/total_use

        self.part1_plotting(data[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region)
        self.part1_plotting(ratio[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region)


    def recommended(self, anbefalt = None, ikke_anbefalt = None, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 1995, period_end = 2030):

        if anbefalt == ikke_anbefalt:
            print('Du må spesisere anbefalt eller ikke anbefalt medisin')
            sys.exit()

        age_indexes = self.age_parameters(age_start, age_end)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        if anbefalt == None:
            data_not_recommended = np.zeros((len(ikke_anbefalt), len(self.year_keys)))
            data_recommended = np.zeros((len(data) - len(ikke_anbefalt) - 1, len(self.year_keys)))
            ikke_anbefalt_indexes = []
            indexes = []
            counter = 0

            for medisin in ikke_anbefalt:
                index = (self.drugs.index(medisin))
                indexes.append(index)
                data_not_recommended[counter] = data[index]
                counter += 1

            counter = 0
            for i in range(len(data)):
                if (i in indexes) or (i == med_type_index):
                    pass
                else:
                    data_recommended[counter] = data[i]
                    counter += 1

        if ikke_anbefalt == None:
            data_recommended = np.zeros((len(anbefalt), len(self.year_keys)))
            data_not_recommended = np.zeros((len(data) - len(anbefalt) - 1, len(self.year_keys)))
            ikke_anbefalt_indexes = []
            indexes = []
            counter = 0

            for medisin in anbefalt:
                index = (self.drugs.index(medisin))
                indexes.append(index)
                data_recommended[counter] = data[index]
                counter += 1

            counter = 0
            for i in range(len(data)):
                if i in indexes or i == med_type_index:
                    pass
                else:
                    data_not_recommended[counter] = data[i]
                    counter += 1
        ratio = np.sum(data_recommended, axis=0)/np.sum(data_not_recommended, axis = 0)
        print(np.sum(data_recommended, axis=0))
        print(np.sum(data_not_recommended, axis = 0))

        self.part1_plotting(ratio, period_start, period_end, 'Ratio', age_indexes, gender, region)


        return None


    def cake(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, year = 2004):

        age_indexes = self.age_parameters(age_start, age_end)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        drug_list = self.drugs.copy()
        del drug_list[med_type_index]

        self.cake_plot(gender, region, data_drugs, total_use, year, age_indexes, drug_list)






if __name__ == "__main__":
    test = visualization('Antiepileptika')
    test.part1()
    test.part1(region='Hele landet', age_start = 15, age_end = 49, period_start = 1980, period_end = 2050)
    test.individual('Valproat', period_start = 2004, period_end = 2018, age_start = 15, age_end = 49)
    test.individual('Valproat', period_start = 2004, period_end = 2018, age_start = 50, age_end = 100)
    test.recommended(ikke_anbefalt = ['Valproat'])

os.chdir(path)

"""
Del 3:
Ratio og Antall brukere
"""





































#jao
