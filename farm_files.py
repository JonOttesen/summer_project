import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import xlrd
import sys
from matplotlib.animation import FuncAnimation
import csv
import time

warnings.filterwarnings("ignore")

path = os.path.abspath(os.path.dirname(__file__))

class files(object):
    """
    A class which reads the excel or csv files from reseptregisteret and writes the file into a dictonary.
    It can also read the excel population file from SSB and place it into a dictonary.
    The functions meant to be called are:
    files_dict for reseptregisteret files
    population_excel for SSB population data (must be excel files)
    To use create a folder containing the data, than create a instance of the class with said name. Than use either "files_dict" or "population_excel" to open the files.
    """

    def __init__(self, folder_name):
        """
        folder_name is the name of the folder the file/files can be found in
        """
        self.directory = folder_name


    def year_indexes(self):
        """
        Finds the indexes for all rows which contains a year written in i.e non empty for the excel files
        """
        indexes = []
        for i in range(len(self.df.iloc[:])):
            if self.df.iloc[i,1] >= 1900:  #Check to see if the number is above 1900 i.e a year over 1900
                indexes.append(i)
        return indexes


    def opening(self, file_name):
        """
        Returns in this order arrays containing:
        All the years for the dataset
        The user data with shape (Total amount of different genders, total amount of different years, indexes pr year pr gender).
        -> The two first indexes selects which gender and years while the last selects the row from the excel file i.e male 2014 and row 14 (index 13) beneath the first entry
        All possible age gropus (strings) as shown in the excel file nicly orderd in the order shown in the excel file.
        Genders (strings) nicly orderd in the order shown in the excel file
        """
        indexes = self.year_indexes()

        d_indexes = int(indexes[1]-indexes[0])  #The distance in indexes between when the year changes i.e the amount of indexes used pr year
        years = np.zeros(len(indexes))
        age_groups = []
        places = []
        genders = []
        for g in range(indexes[0], indexes[0] + d_indexes):  #Creates lists containing all possible age groups, genders and places
            age_groups.append(self.df.iloc[g,2])
            places.append(self.df.iloc[g,4])
            genders.append(self.df.iloc[g,3])

        age_groups = self.ordering(age_groups)  #Remove duplicates and order them in the order of the excel file
        genders = self.ordering(genders)
        places = self.ordering(places)
        users = np.zeros(shape = (len(genders),len(indexes), int(d_indexes/len(genders))))  #Array used to contain the information in the excel file

        #The first index in the excel file is the row while the second is the column
        for k in range(len(indexes)):  #Looping through all the years
            years[k] = float(self.df.iloc[indexes[k],1])
            counter = 0
            index_counter_for_gender = 0
            prev_count = 0

            for j in range(indexes[k], indexes[k]+d_indexes):  #Looping through all the places and age_groups
                use = self.df.iloc[j,6]
                if type(use) != type(2):  #Check for NaN, Inf or "Below 5" types. Setting them to 1 to avoid divide by 0 in the future
                    use = 1

                if counter > len(places) - 1 + prev_count:
                    if index_counter_for_gender >= len(genders)-1:
                        index_counter_for_gender = 0
                        prev_count = counter
                    else:
                        counter = prev_count
                        index_counter_for_gender += 1

                users[index_counter_for_gender, k, counter] = use  #Fills in the value from the excel file in the correct index representing gender, year and age group.
                counter += 1  #Counting down the rows

        return years, users, age_groups, places, genders


    def ordering(self, array):
        """
        Removes duplicates from lists and than sorts them such that the original order is kept
        Takes a numpy array and return the sorted without duplicates.
        """
        array1, order = np.unique(array, return_index=True)
        counter = 0
        new_array = np.zeros_like(array1)
        for index in np.sort(order):
            new_array[counter] = array[index]
            counter += 1
        return new_array[new_array != 'nan']


    def files_dict(self, file_name, gender_column = 4, age_group_column = 3, year_column = 2, region_column = 5, sum_column = 6):
        """
        Places all parameters in a single dictonary for the excel file from reseptregisteret.
        Order in the dictionary: Gender -> Year -> Age group -> Place (fylke/region/hele landet).
        file_name is the name of the file whished read with the filetype ending.
        The colums are which colums in the cvs file version corresponds the the genders, years etc. Give the column number i.e start counting at 1 not the the index number.
        """
        os.chdir(self.directory)

        if 'csv' in file_name:
            #For opening csv pandas is quicker, about 9 times as fast.
            users_dict = {}
            data = pd.read_csv(file_name)
            keys = data.keys()
            gender_column -= 1; age_group_column -= 1; year_column -= 1; region_column -= 1; sum_column -= 1  #Make sure the index represents the correct column

            age_groups = self.ordering(data[keys[age_group_column]])  #Remove duplicates and order them
            genders = self.ordering(data[keys[gender_column]])
            places =self.ordering(data[keys[region_column]])
            years = np.unique(data[keys[year_column]])


            """with open(file_name, newline='', encoding='utf-8') as csvfile:  #A slover version of the above four lines
                reader = csv.DictReader(csvfile)

                genders, years, age_groups, places = [], [], [], []
                gender_column -= 1; age_group_column -= 1; year_column -= 1; region_column -= 1; sum_column -= 1
                for row in reader:
                    gender = row[keys[gender_column]]
                    year = float(row[keys[year_column]])
                    age_group = row[keys[age_group_column]]
                    place = row[keys[region_column]]
                    value = row[keys[sum_column]]
                    if gender not in genders:
                        genders.append(gender)
                    if year not in years:
                        years.append(year)
                    if age_group not in age_groups:
                        age_groups.append(age_group)
                    if place not in places:
                        places.append(place)"""

            for gender in genders:  #Looping through the different parameters and creating the correct dictonary
                users_dict[gender] = {}

                for year in years:
                    users_dict[gender][year] = {}

                    for age_group in age_groups:
                        users_dict[gender][year][age_group] = {}

                        for place in places:
                            users_dict[gender][year][age_group][place] = 1

            gender_key, year_key, age_key, place_key, value_key = keys[np.array([gender_column, year_column, age_group_column, region_column, sum_column])]
            with open(file_name, newline='', encoding='utf-8') as csvfile:  #Filling the dictonary
                reader = csv.DictReader(csvfile)
                for row in reader:

                    gender, year, age_group, place, value = row[gender_key], row[year_key], row[age_key], row[place_key], row[value_key]

                    if value.isdigit():  #Check for missing values
                        users_dict[gender][float(year)][age_group][place] = float(value)

            os.chdir(path)
            return users_dict

        else:
            wb = xlrd.open_workbook(file_name, logfile=open(os.devnull, 'w'))  #39.58 sek on test
            self.df = pd.read_excel(wb, engine='xlrd')
            #self.df = pd.read_excel(io = file_name, sheet = 0)  #38.97 on test


        os.chdir(path)

        indexes = self.year_indexes()
        users_dict = {}
        a = []

        d_indexes = int(indexes[1]-indexes[0])
        years, users, age_groups, places, genders = self.opening(file_name)

        counter2 = 0
        for gender in genders:  #Looping through the different parameters
            users_dict[gender] = {}
            users1 = np.concatenate(users[counter2])
            counter = 0

            for year in years:
                users_dict[gender][year] = {}

                for age_group in age_groups:
                    users_dict[gender][year][age_group] = {}

                    for place in places:
                        users_dict[gender][year][age_group][place] = users1[counter]
                        counter += 1
            counter2 += 1

        os.chdir(path)
        return users_dict


    def population_excel(self, file_name, stop_at_90 = True):
        """
        Places all parameters in a single dictonary for the excel file from SSB.
        Order in the dictionary: Gender -> Year -> Age group -> Place (fylke).
        Takes the filename with the filetype ending and if stop_at_90 = True it will sum up all numbers above 90 and place them in the same category.
        """
        os.chdir(self.directory)

        wb = xlrd.open_workbook(file_name, logfile=open(os.devnull, 'w'))  #39.58 sek on test
        self.df = pd.read_excel(wb, engine='xlrd')
        #self.df = pd.read_excel(io = file_name, sheet = 0)  #38.97 on test

        os.chdir(path)

        region_indexes = []
        places = []
        for i in range(len(self.df.iloc[:])):

            if type(self.df.iloc[i,0]) == type('str'):
                if np.isnan(self.df.iloc[i,3]):
                    last_index = i-1
                    break
                if 'Finnmark' in self.df.iloc[i,0][3:]:  #Removing the extra names of these counties from the SSB excel file
                    places.append('Finnmark')
                elif 'Troms' in (self.df.iloc[i,0][3:]):
                    places.append('Troms')
                else:
                    places.append(self.df.iloc[i,0][3:])
                region_indexes.append(i)

        d_indexes = region_indexes[1] - region_indexes[0]
        years = []
        age_groups = []
        genders = []
        year_indexes = []

        for i in range(3,len(self.df.iloc[region_indexes[0]])):  #3 is the number the year counting starts
            years.append(float(self.df.iloc[2,i]))
            year_indexes.append(i)

        for i in range(region_indexes[0], region_indexes[1]):
            if type(self.df.iloc[i,1]) == type('str'):
                if 'Kvinne' in self.df.iloc[i,1]:  #To make sure both Excel files use the same keys i.e 'Kvinne' for both
                    genders.append(self.df.iloc[i,1][:-1])
                elif 'M' in self.df.iloc[i,1] and 'nn' in self.df.iloc[i,1]:
                    genders.append('Mann')
                else:
                    genders.append(self.df.iloc[i,1])
            if len(genders) < 2:
                if 'år' in self.df.iloc[i, 2]:
                    age_groups.append(self.df.iloc[i, 2][:-3])
                else:
                    age_groups.append(self.df.iloc[i, 2])

        if stop_at_90:
            age_groups2 = []
            for letters in age_groups:  #To match the format of reseptregisteret and end with 90+ not 100+
                nol = len(letters)-1  #Numbers of letters
                if '90' in letters:
                    age_groups2.append('90+')
                    break
                else:
                    age_groups2.append(letters[:int(nol/2)] + ' - ' + letters[int(nol/2+1):])
            delta_age_len = len(age_groups) - len(age_groups2)
            age_groups = age_groups2

        else:
            delta_age_len = 0

        j, k, l, i = 0, 0, 0, 0 #Indexes to interate over
        users_dict = {}

        for gender in genders:  #Looping through the different parameters
            users_dict[gender] = {}
            j = 0
            d = int(d_indexes*i/2)

            for year in years:
                users_dict[gender][year] = {}
                k = 0

                for age_group in age_groups:
                    users_dict[gender][year][age_group] = {}
                    l = 0
                    tronderlag_sum = 0
                    if stop_at_90 and age_group == '90+':  #Stop the summation when not at 90+ or when we don't stop at 90+
                        summation = True
                    else:
                        summation = False

                    for place in places:
                        if 'Trøndelag' in place:  #Sum such that the only key is Trøndelag before 2017 and after. Not divided to north and south.
                            if summation:  #Make sure all people over 90 is summed up not divided into 90-94, 95-99 etc. This is to match the data from reseptregisteret.
                                tronderlag_sum += np.sum(self.df.iloc[region_indexes[l] + k + d: region_indexes[l] + k + delta_age_len + d+ 1, year_indexes[j]])
                            else:
                                tronderlag_sum += self.df.iloc[region_indexes[l] + k + d, year_indexes[j]]
                        else:
                            if summation:
                                users_dict[gender][year][age_group][place] = np.sum(self.df.iloc[region_indexes[l] + k + d: region_indexes[l] + k + delta_age_len + d + 1, year_indexes[j]])
                            else:
                                users_dict[gender][year][age_group][place] = self.df.iloc[region_indexes[l] + k + d, year_indexes[j]]

                        l += 1
                    users_dict[gender][year][age_group]['Trøndelag'] = tronderlag_sum
                    k += 1
                j += 1
            i += 1

        os.chdir(path)
        return users_dict



class visualization(object):
    """
    A class ment to visualize the data from reseptregisteret and SSB in a multitude of graphs and animations.
    """

    def __init__(self, folder_name, stop_at_90 = True):
        self.folder_name = folder_name

        os.chdir(self.folder_name)
        self.filenames = os.listdir()
        os.chdir(path)

        try:
            self.filenames.remove('Befolkning.xlsx')
            self.population = files(folder_name).population_excel('Befolkning.xlsx', stop_at_90)
            self.p_gender_keys = list(self.population.keys())
            self.p_year_keys = list(self.population[self.p_gender_keys[0]].keys())
            self.p_age_group_keys = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]].keys())
            self.p_places = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]][self.p_age_group_keys[0]].keys())
        except:
            print('Det eksisterer ikke en fil med navn \'Befolkning.xlsx\' eller \'Befolkning.csv\' i filmappen ' + folder_name)
            print('Del 3 kan dermed ikke anvendes.')
            pass

        self.data = []
        self.drugs = []  #Name of the drugs
        for i in self.filenames:
            self.drugs.append(i[:-4])  #Removing the .xls or .csv ending

        for filename in self.filenames:
            self.data.append(files(folder_name).files_dict(filename))  #Dictionaries for all the xls files in the folder


        self.gender_keys = list(self.data[0].keys())  #The keys for the different genders
        self.year_keys = list(self.data[0][self.gender_keys[0]].keys())  #The keys for the different years in the data set
        self.age_group_keys = list(self.data[0][self.gender_keys[0]][self.year_keys[0]].keys())  #The keys for the different age groups
        self.places = list(self.data[0][self.gender_keys[0]][self.year_keys[0]][self.age_group_keys[0]].keys())  #The keys for the different regions/locations


    def help(self):
        """
        Prints a table with all the keys for part 1 and part 2 i.e the part without the total population.
        """
        most_elements = max([len(self.places), len(self.age_group_keys), len(self.gender_keys), len(self.drugs)])
        print('-'*109)
        print('|' + 'Tabell av parameterene, alt innenfor \" og med \" er parameteren'.center(107) + '|')
        print('|' + '-'*107 + '|')
        print("|{0:26s}|{1:26s}|{2:26s}|{3:26s}|".format('Kjønn', 'Medisiner/medisintyper', 'Aldersgrupper', 'Regioner/steder'))
        print('|' + '-'*26 + '|' + '-'*26 + '|' + '-'*26 + '|' + '-'*26 + '|')
        for i in range(most_elements):
            try:
                gender = "\"" + self.gender_keys[i] + "\""
            except:
                gender = ''
            try:
                age = "\"" + self.age_group_keys[i] + "\""
            except:
                age = ''
            try:
                place = "\"" + self.places[i] + "\""
            except:
                place = ''
            try:
                drug = "\"" + self.drugs[i] + "\""
            except:
                drug = ''
            print("|{0:26s}|{1:26s}|{2:26s}|{3:26s}|".format(gender, drug, age, place))
        print('-'*109)


    def age_parameters(self, age_start, age_end, age_group_keys = None):
        """
        Nothing interesting really,
        find the indexes corresponding to the correct keys in self.age_group_keys based on the start age and end age.
        A requirement is that self.age_group_keys starts at 0 and ends at 90+ having all the parameters in-between i.e selecting all age gropus from reseptregisteret.
        Another requirement is that the age groups are as followed 0-4, 5-9, 10-14 i.e 5 year groups.
        """
        if age_group_keys == None:
            age_group_keys = self.age_group_keys
        else:
            pass

        age_indexes = []
        if age_start >= 90:
            age_indexes.append(len(age_group_keys)-1)
        else:
            for i in range(0, len(age_group_keys)):
                if age_start < 5*(i+1) and age_end >= 5*(i):
                    age_indexes.append(i)

        return age_indexes


    def curve_fitting(self, data):
        """
        Fits a curve to the given data (1D array): The possible curves are exponential, logarithmic, polynomial, polynomials of negative power and cosine
        ## TODO: Redo this to work with all the given curve types and try machine learning techniques
        Returns the fitted function
        """
        degree = 1
        years = []
        new_data = []

        years.append(self.year_keys[0]-60)
        years.append(self.year_keys[0]-50)
        for i in self.year_keys:
            years.append(i)
        years.append(self.year_keys[-1]+60)
        years.append(self.year_keys[-1]+70)

        new_data.append(0.01)
        new_data.append(0.1)
        for i in data:
            new_data.append(i)
        new_data.append(0.1)
        new_data.append(0.01)
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
        """
        Uses the function found in curve_fitting and the data points to give a final array consisting of both.
        It uses the data points when possible, but if the time array consists of years outside of the years in the data it uses the function.
        data -> the data from reseptregisteret.
        f -> is array of the fitted function already used on the time array.
        time -> the time from period_start to period_end in integer steps.
        returns a array of datapoints used to plot againts the time array.
        """
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


    def part1_plotting(self, data, period_start, period_end, drug_list, age_indexes, gender, region, label = 'Antall utskrivninger', file_name = True):
        """
        A function used for plotting in a histogram like fashion.
        It uses both the data given and if necesarry uses the curve_fitting and final_function functions when there is not enough data.
        data         -> All the data points mathcin the region, desired age and gender. This also includes that of which is not plotted since all data is used to find the best curve fit.
                        Takes a array of size (number of drugs, total number of years).
        period_start -> The earliest year the plot starts at. Takes integer values
        period_end   -> The last year in the plot. Takes integer values
        drug_list    -> A list of all the drugs plotted. Either a list for multiple drugs plotted or a string for a single drug plotted.
        age_indexes  -> A list gotten from the function age_indexes.
        gender       -> The gender the data is for, this is a string.
        region       -> The region the data is for, this is a string.
        label        -> Label for for the y-axis
        """

        time = np.linspace(period_start, period_end, period_end-period_start+1)
        bars = len(data)
        offset = 0.8/bars

        if len(age_indexes) > 1:
            alder = self.age_group_keys[age_indexes[0]] + ' til ' + self.age_group_keys[age_indexes[-1]]
        else:
            alder = self.age_group_keys[age_indexes[0]]

        plt.figure(figsize = [12, 4.8])
        if type(data[0]) == type(np.array([1])):
            for i in range(len(data)):
                func = self.curve_fitting(np.log(data[i]))
                plt.bar(time-0.4+i*offset, self.final_function(data[i], np.exp(func(time)), time), width = offset, label = drug_list[i])
                #plt.plot(time, np.exp(func(time)))
                plt.legend()
                plt.title(gender + ' i ' + region + ' alder ' + alder)
        else:
            func = self.curve_fitting(np.log(data))
            plt.bar(time, self.final_function(data, np.exp(func(time)), time), label = drug_list)
            plt.legend()
            plt.title(gender + ' i ' + region + ' alder ' + alder)

        plt.xlabel('År')
        plt.ylabel(label)

        if file_name:
            pass
        else:
            plt.savefig(file_name+'.png')
        plt.show()

        return None


    def drug_array(self, age_indexes, region, gender):
        """
        Returns an array containing the date from reseptregisteret for the desired ages, regions and genders. The data for every year is returned.
        Array has the size (number of drugs, total number of years in the reseptregisteret data).
        The returned array sums upp all contributions from the start age to the end age specified when getting the age_indexes list.
        """
        data = np.zeros((len(self.drugs), len(self.year_keys)), dtype = np.float)

        for i in range(len(self.drugs)):
            for k in range(len(self.year_keys)):
                for j in range(len(age_indexes)):
                    data[i, k] += self.data[i][gender][self.year_keys[k]][self.age_group_keys[age_indexes[j]]][region]

        return data


    def population_array(self, age_indexes, region, gender, year_keys = True):
        """
        Returns an array containing the date from SSB for the desired ages, regions and genders. The data for every year is returned.
        Array has the size (total number of years in the specified year_keys parameter (often reseptregisteret)).
        The returned array sums upp all contributions from the start age to the end age specified when getting the age_indexes list.
        """
        if year_keys:
            year_keys = self.year_keys
        else:
            pass

        data = np.zeros((len(year_keys)))

        for k in range(len(year_keys)):
            for j in range(len(age_indexes)):
                data[k] += self.population[gender][year_keys[k]][self.p_age_group_keys[age_indexes[j]]][region]
        return data


    def cake_plot(self, gender, region, data_drugs, data_tot_drugs, year, age_indexes, drug_list):
        """
        A simpe cake plot for the specified gender, region, year and age group. These are all strings
        data_drugs     -> Is the data for all the different drugs plotted in the cake diagram size (number of drugs, total number of years)
        data_tot_drugs -> Is the data for the total number of users of all the drugs given pluss others size (total number of years)
        year           -> The integer year the data is gotten from. This can be anything since the data is curvefitted using curve_fitting and finally final_function.
        age_indexes    -> A list gotten from the function age_indexes.
        drug_list      -> A list of the drugs plotted, the index for the drug must be the same as in data_drugs.
        """

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

        plt.figure(figsize = [12, 4.8])
        plt.pie(x, explode = explosion, labels = drug_list, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title(gender + ' i ' + region + ' alder ' + alder + ' år ' + str(year))
        plt.show()

        return None


    def probability(self, probs):

        if np.sum(probs)>1:
            probs /= 100
            print('Do not use probability in %, this is now fixed but for furture reference')
        else:
            pass

        tot_prob = probs[0]
        for i in range(1, len(probs)):
            tot_prob += probs[i] + probs[i]*probs[i-1]

        return tot_prob


    def generelt_medisinforbruk(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = False):
        """
        Gives a bar plot of the medical use in Norway based on all the data in the folder and the specified parameters. The plot can either be of ratios or total use.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        gender       -> The gender (string) must be the same as the string given in the help function.
        region       -> The region (string) must be the same as the string given in the help function.
        age_start    -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end      -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end   -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                        period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        ratio        -> Either True or something else and chooses whether to plot the ratio or total use.
        """

        age_indexes = self.age_parameters(age_start, age_end)

        data = self.drug_array(age_indexes, region, gender)

        med_type_index = self.drugs.index(self.folder_name)
        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        ratio_data = data_drugs/total_use
        ratio_list = self.drugs[:]
        del ratio_list[med_type_index]
        if ratio:
            self.part1_plotting(ratio_data, period_start, period_end, ratio_list, age_indexes, gender, region, label = 'Ratio')
        else:
            self.part1_plotting(data, period_start, period_end, self.drugs, age_indexes, gender, region)


    def individuelt_medisinforbruk(self, drug, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = False):
        """
        Gives a bar plot of the medical use for a specific medicine in Norway based on the data in the folder and for the specified parameters. The plot can either be of ratios or total use.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        drug         -> The medicine the plot is for (string). The string must be exactly the same as the name of the datafile in the folder without the .xls or .csv ending.
        gender       -> The gender (string) must be the same as the string given in the help function.
        region       -> The region (string) must be the same as the string given in the help function.
        age_start    -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end      -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end   -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                        period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        ratio        -> Either True or something else and chooses whether to plot the ratio or total use.
        """

        age_indexes = self.age_parameters(age_start, age_end)
        med_index = self.drugs.index(drug)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        ratio_data = data/total_use

        if ratio:
            self.part1_plotting(ratio_data[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region, label = 'Ratio')
        else:
            self.part1_plotting(data[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region)


    def forhold_medisin(self, anbefalt = None, ikke_anbefalt = None, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 1995, period_end = 2030):
        """
        Gives a bar plot of the ratio (recommended medicine)/(not recommended medicine) for the specified medicine based on the data in the folder and for the specified parameters.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        anbefalt      -> The recommended medicine or medicines, if specified the ikke_anbefalt parameter MUST NOT be specified
                         The parameter takes a string or list of strings of the recommended medicines.
        ikke_anbefalt -> The not recommended medicine or medicines, if specified the anbefalt parameter MUST NOT be specified
                         The parameter takes a string or list of strings of the not recommended medicines.
        gender        -> The gender (string) must be the same as the string given in the help function.
        region        -> The region (string) must be the same as the string given in the help function.
        age_start     -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end       -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start  -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end    -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                         period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        """

        if anbefalt == ikke_anbefalt:
            print('Du må spesisere anbefalt eller ikke anbefalt medisin')
            sys.exit()

        age_indexes = self.age_parameters(age_start, age_end)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        if anbefalt == None:
            if type(ikke_anbefalt) == type([]):
                pass
            else:
                ikke_anbefalt = [ikke_anbefalt]

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
            if type(anbefalt) == type([]):
                pass
            else:
                anbefalt = [anbefalt]

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

        self.part1_plotting(ratio, period_start, period_end, 'Ratio', age_indexes, gender, region, label = 'Ratio')


        return None


    def kake_medisinforbruk(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, year = 2004):
        """
        Gives a cake plot of the ratios medicine/(medicine group) for all the medicines in the data folder.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        gender        -> The gender (string) must be the same as the string given in the help function.
        region        -> The region (string) must be the same as the string given in the help function.
        age_start     -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end       -> The oldest age the data is chosen from (int number from age_start -> infinity)
        year          -> The year the cake plot is for which is an integer from 0 -> infinity but it's recommended to stay within +- 10 years of the actual data.
        """

        age_indexes = self.age_parameters(age_start, age_end)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        drug_list = self.drugs.copy()
        del drug_list[med_type_index]

        self.cake_plot(gender, region, data_drugs, total_use, year, age_indexes, drug_list)


    def medisinforbruk_tidsutviling(self, drug, gender = 'Kvinne', region = 'Hele landet', ratio = False):
        """
        An animation of the time evolution of either the ratio or total number of users of a specific medicine. The ratio is the medicine/(medicine type)
        drug    -> The name of the medicine as given in the data folder but whithout the .csv or .xls ending.
        gender  -> The gender (string) must be the same as the string given in the help function.
        region  -> The region (string) must be the same as the string given in the help function.
        ratio   -> Either True for ratio or something else for no ratio. It plots the ratio of medicine/(medicine type).
        """

        age_indexes = self.age_parameters(0, 100)
        med_index = self.drugs.index(drug)
        med_type_index = self.drugs.index(self.folder_name)
        med_dict = self.data[med_index]
        data = np.zeros((len(self.age_group_keys) ,len(self.year_keys)))
        x_axis = np.linspace(1, len(self.age_group_keys), len(self.age_group_keys))

        index = 0
        for k in self.age_group_keys:
            for i in range(len(self.year_keys)):
                year = self.year_keys[i]
                data[index][i] = med_dict[gender][year][k][region]
            index += 1

        if ratio:
            med_dict2 = self.data[med_type_index]
            data_tot = np.zeros((len(self.age_group_keys) ,len(self.year_keys)))
            index = 0

            for k in self.age_group_keys:
                for i in range(len(self.year_keys)):
                    year = self.year_keys[i]
                    data_tot[index][i] = med_dict2[gender][year][k][region]
                index += 1

            f = interpolate.interp1d(self.year_keys, data/data_tot, fill_value='extrapolate')
        else:
            f = interpolate.interp1d(self.year_keys, data, fill_value='extrapolate')

        x = np.linspace(0, len(self.age_group_keys)-1, len(self.age_group_keys))

        fig, ax = plt.subplots(figsize = [12, 4.8])

        xdata, ydata = [], []
        ln, = plt.plot([], [], 'ro')

        def init():
            ax.set_xlim(-1, len(self.age_group_keys))
            plt.xticks(np.arange(len(self.age_group_keys)), self.age_group_keys, fontsize = 7)
            plt.xlabel('Alder')
            if ratio:
                ax.set_ylim(0, 1.5*np.max(data/data_tot))
                plt.ylabel('Ratio')
            else:
                ax.set_ylim(0, 1.5*np.max(data))
                plt.ylabel('Antall utskrivninger')
            return ln,

        def update(frame):
            ln.set_data(x, f(frame))
            ax.set_title('Medisin for %s i %s: %s i år %.2f' %(gender, region, drug, frame))
            return ln,

        try:
            __IPYTHON__
        except NameError:
            ani = FuncAnimation(fig, update, frames=np.linspace(self.year_keys[0], self.year_keys[-1], 250), init_func=init, blit=False, interval = 1, repeat = True)
            plt.show()
        else:
            ani = FuncAnimation(fig, update, frames=np.linspace(self.year_keys[0], self.year_keys[-1], 250), init_func=init, blit=True, interval = 40, repeat = True)
            return ani.to_html5_video()
        # conda install -c conda-forge ffmpeg ##Into the terminal made it work for me

    def medisiner_og_befolkning(self, prevalens, sykdom = 'Epilepsi', gender = 'Mann', region = 'Hele landet', age_start = 0, age_end= 100, period_start = 2004, period_end = 2030):
        """
        Returns a bar plot of the number of users for the different medicines and medicine type given in the data folder with the number of people having a specified disease or diseases.
        The data for the diseases is based of the probability of having said disease and the number of inhabitants gotten from SSB.
        prevalens     -> The probability for a disease or diseases, can be either a number or a list of numbers all in probability % not fractions.
                         For multiple prevalens it uses regular probability calcualtions see the probability function.
        sykdom        -> Might drop ## TODO: see what to do with this one.
        gender        -> The gender (string) must be the same as the string given in the help function.
        region        -> The region (string) must be the same as the string given in the help function.
        age_start     -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end       -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start  -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end    -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                         period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        """

        if type(prevalens) == type([]):
            prevalens = self.probability(np.array(prevalens)/100)
        else:
            prevalens /= 100
        #Source SSB https://www.ssb.no/statbank/table/07459/

        age_indexes = self.age_parameters(age_start, age_end)
        data = self.drug_array(age_indexes, region, gender)

        if region == 'Hele landet':
            p_data = self.population_array(age_indexes, self.p_places[0], gender)
            for k in range(1,len(self.p_places)):
                p_data += self.population_array(age_indexes, self.p_places[k], gender)
        else:
            p_data = self.population_array(age_indexes, region, gender)

        plotting_data =  np.zeros((len(self.drugs) + 1, len(self.year_keys)))
        plotting_data[:-1] = data
        plotting_data[-1] = p_data*prevalens
        drugs_name = self.drugs[:]
        drugs_name.append('Prevalens: %.2f%%' %(prevalens*100))

        self.part1_plotting(plotting_data, period_start, period_end, drugs_name, age_indexes, gender, region, label = 'Antall personer')


    def medisiner_og_befolkning(self, prevalens, drug = 'Valproat', gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = True):
        """
        Returns a bar plot of either the ratio using a medicine/(number of people having the disease) or the number of people having the disease and the number of people using the medicine.
        prevalens     -> The probability for a disease or diseases, can be either a number or a list of numbers all in probability % not fractions.
                         For multiple prevalens it uses regular probability calcualtions see the probability function.
        drug          -> The medicine the plot is for (string) not using the .csv or .xls endings.
        gender        -> The gender (string) must be the same as the string given in the help function.
        region        -> The region (string) must be the same as the string given in the help function.
        age_start     -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end       -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start  -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end    -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                         period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        """


        if type(prevalens) == type([]):
            prevalens = self.probability(np.array(prevalens)/100)
        else:
            prevalens /= 100
        #Source SSB https://www.ssb.no/statbank/table/07459/

        age_indexes = self.age_parameters(age_start, age_end)
        data = self.drug_array(age_indexes, region, gender)

        if region == 'Hele landet':
            p_data = self.population_array(age_indexes, self.p_places[0], gender)
            for k in range(1,len(self.p_places)):
                p_data += self.population_array(age_indexes, self.p_places[k], gender)
        else:
            p_data = self.population_array(age_indexes, region, gender)

        med_index = self.drugs.index(drug)
        med_type_index = self.drugs.index(self.folder_name)

        data_drugs = np.delete(data, med_type_index, 0)

        if ratio:
            self.part1_plotting(data[med_index]/(prevalens*p_data), period_start, period_end, 'Ratio '+self.drugs[med_index]+' over antall med sykdom X', age_indexes, gender, region, label = 'Ratio')
        else:
            self.part1_plotting([data[med_index], p_data], period_start, period_end, [self.drugs[med_index], 'Befolkning med X'] , age_indexes, gender, region)




if __name__ == "__main__":
    #opening_test = files('Antiepileptika')
    #test2 = opening_test.files_dict('Antiepileptika.xls')

    #opening_test2 = files('Antiepileptika2')
    #test = opening_test2.files_dict('Antiepileptika2.csv')
    #print(test2 == test)


    #test = visualization('Antiepileptika')
    time1 = time.time()
    test2 = visualization("Antiepileptika")


os.chdir(path)

"""
Del 3:
Ratio og Antall brukere
"""





































#jao
