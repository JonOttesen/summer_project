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
    The methods ment to be called are after initialization:
    - tabell
    - generelt_medisinforbruk
    - individuelt_medisinforbruk
    - forhold_medisin
    - kake_medisinforbruk
    - medisinforbruk_tidsutviling
    - medisiner_og_befolkning
    - medisiner_og_befolkning2
    """

    def __init__(self, folder_name, plot_type = 'column', stop_at_90 = True):
        self.folder_name = folder_name

        if plot_type == 'dot':  #Decides which type of plot is plotted
            self.plot_type = 1
        elif plot_type == 'line':
            self.plot_type = 2
        else:
            self.plot_type = 3


        os.chdir(self.folder_name)
        self.filenames = os.listdir()
        os.chdir(path)

        try:  #Checks for either the csv or xlsx population file from either reseptregisteret or SSB.
            self.filenames.remove('Befolkning.xlsx')
            self.population = files(folder_name).population_excel('Befolkning.xlsx', stop_at_90)
            self.p_gender_keys = list(self.population.keys())
            self.p_year_keys = list(self.population[self.p_gender_keys[0]].keys())
            self.p_age_group_keys = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]].keys())
            self.p_places = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]][self.p_age_group_keys[0]].keys())
        except:
            pass
        try:
            self.filenames.remove('Befolkning.csv')
            self.population = files(folder_name).files_dict('Befolkning.csv')
            self.p_gender_keys = list(self.population.keys())
            self.p_year_keys = list(self.population[self.p_gender_keys[0]].keys())
            self.p_age_group_keys = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]].keys())
            self.p_places = list(self.population[self.p_gender_keys[0]][self.p_year_keys[0]][self.p_age_group_keys[0]].keys())
        except:
            print('Det eksisterer ikke en fil med navn \'Befolkning.xlsx\' eller \'Befolkning.csv\' i filmappen ' + folder_name)
            print('Funksjonene: medisiner_og_befolkning og medisiner_og_befolkning2 kan dermed ikke anvendes')
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

        check = False
        for name in self.filenames:
            if folder_name in name:
                check = True
        if check:
            self.med_type_index = self.drugs.index(self.folder_name)
        else:
            self.med_type_index = 'Not given'
            print('Det eksisterer ingen ' + folder_name + '.csv/.xls fil i mappen.' + folder_name)
            print('Flere funksjonaliteter vil dermed ikke fungere, disse er:')
            print('Ratio delen av: generelt_medisinforbruk, individuelt_medisinforbruk og medisinforbruk_tidsutviling.')
            print('Metoden kake_medisinforbruk vil nå kun vise fordelingen av medisinene som om summen av bruken var hele forbruket.')


    def tabell(self):
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

        if len(str(self.year_keys)[1:-1]) <= 107:
            string = '|' + str(self.year_keys)[1:-1] + ' '*(int(107-len(str(self.year_keys)[1:-1]))) + '|'
            print('|' + '-'*107 + '|')
            print('|' + 'Årstall med datapunkter:'.center(107) + '|')
            print('|' + '-'*107 + '|')
            print(string)
            print('-'*109)
        else:
            print('-'*(109))
            print('Årstall med datapunkter:')
            print(str(self.year_keys)[1:-1])


    def probability(self, probs):

        if np.sum(probs)>1:
            probs /= 100
            print('Do not use probability in %, this is now fixed but for furture reference')
        else:
            pass

        tot_prob = probs[0]  #Summing up independent probabilities.
        for i in range(1, len(probs)):
            tot_prob += probs[i] + probs[i]*probs[i-1]

        return tot_prob


    def age_parameters(self, age_start, age_end):
        """
        Nothing interesting really,
        find the indexes corresponding to the correct keys in self.age_group_keys based on the start age and end age.
        A requirement is that self.age_group_keys starts at 0 and ends at 90+ having all the parameters in-between i.e selecting all age gropus from reseptregisteret.
        Another requirement is that the age groups are as followed 0-4, 5-9, 10-14 i.e 5 year groups.
        """

        age_indexes = []
        if age_start >= 90:
            age_indexes.append(len(self.age_group_keys)-1)
        else:
            for i in range(0, len(self.age_group_keys)):
                if age_start < 5*(i+1) and age_end >= 5*(i):
                    age_indexes.append(i)

        return age_indexes


    def curve_fitting(self, data, time):
        """
        Fits a curve to the given data (1D array): The possible curves are exponential, logarithmic, polynomial, polynomials of negative power and cosine
        ## TODO: Redo this to work with all the given curve types and try machine learning techniques
        Returns the fitted function
        """
        years = np.array(self.year_keys)

        counter = 0
        for g in time:
            if g in self.year_keys:
                counter +=1
        if counter == len(time):
            return None

        linear_func = np.poly1d(np.polyfit(years, data, 1))

        if 0 in data:  #To avodi possible log(0)
            linear_exp1D = 0
        else:
            linear_exp1D = np.poly1d(np.polyfit(years, np.log(data), 1))  #Remember to np.exp np.exp(linear_exp(data))

        def func(x, a, b, c):
            return a*np.exp(-((x-b)/c)**2)

        a = np.linspace(1*np.max(data), 2*np.max(data), 21)
        b = np.linspace(years[0] - 12, years[-1] + 12, 24 + (years[-1] - years[0]) + 1)
        c = np.linspace(0.1, 100, 100)

        X = np.array(np.meshgrid(a,b,c)).T.reshape(-1,3)

        test = func(years, X[0,0], X[0,1], X[0,2])
        least_squares = np.sum((test-data)**2)
        a, b, c = X[0,0], X[0,1], X[0,2]

        for i, j, k in X:  #Finds the best bell curve fit
            test = func(years, i, j, k)
            ls = np.sum((test-data)**2)
            if ls < least_squares:
                least_squares = ls
                a, b, c = i, j, k

        combs_1 = np.linspace(1, 10, 21)  #Starts at 1 to avoid division by zero
        combs_2 = np.linspace(0, 5, 12)
        if linear_exp1D == 0:
            lincombs = np.array(np.meshgrid(combs_1,combs_2)).T.reshape(-1,2)
            least_squares = 1e20
            lin_weight = 4

            for i, k in lincombs:
                f_new = 1/(lin_weight*i + k)*(lin_weight*i*linear_func(years) + k*func(years, a, b, c))  #Expands the best as a basis
                ls = np.sum((f_new - data)**2)
                if least_squares > ls:
                    least_squares = ls
                    f = 1/(lin_weight*i + k)*(lin_weight*i*linear_func(time) + k*func(time, a, b, c))  #Expands the best as a basis
                    weights = [i,j]
        else:
            lincombs = np.array(np.meshgrid(combs_1,combs_2,combs_2)).T.reshape(-1,3)
            least_squares = 1e20
            lin_weight = 4

            for i, j, k in lincombs:
                f_new = 1/(lin_weight*i + j + k)*(lin_weight*i*linear_func(years) + j*np.exp(linear_exp1D(years)) + k*func(years, a, b, c))  #Expands the best as a basis
                ls = np.sum((f_new - data)**2)
                if least_squares > ls:
                    least_squares = ls
                    f = 1/(lin_weight*i + j + k)*(lin_weight*i*linear_func(time) + j*np.exp(linear_exp1D(time)) + k*func(time, a, b, c))  #Expands the best as a basis
                    weights = [i,j,k]

        return f


    def final_function(self, data, f, time):
        """
        Uses the function found in curve_fitting and the data points to give a final array consisting of both.
        It uses the data points when possible, but if the time array consists of years outside of the years in the data it uses the function.
        data -> the data from reseptregisteret.
        f -> is array of the fitted function already used on the time array.
        time -> the time from period_start to period_end in integer steps.
        returns a array of datapoints used to plot againts the time array.
        """
        if type(time) == type([1]) or type(time) == type(np.array([1])):  #Makes sure the type of time and f is correct.
            pass
        else:
            f = [f]
            time = [time]

        final = []
        for year in time:
            if year in self.year_keys:
                final.append(data[self.year_keys.index(year)])  #Checks whether there exist data for that year, if not use the curve fitted data.
            else:
                final.append(f[time == year])
        return np.array(final)


    def part1_plotting(self, data, period_start, period_end, drug_list, age_indexes, gender, region, label = 'Forbruk', save_fig = False):
        """
        A function used for plotting in a histogram like fashion.
        It uses both the data given and if necesarry uses the curve_fitting and final_function functions when there is not enough data.
        data         -> All the data points mathcin the region, desired age and gender. This also includes that of which is not plotted since all data is used to find the best curve fit.
                        Takes a array of size (number of drugs, total number of years). Has to be either a 1D or 2D array.
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
        markers = ['.', '^', '1', '8', 's', 'p', '*', 'x', 'D', 'h']
        if type(data[0]) == type(np.array([1])):
            for i in range(len(data)):
                func = self.curve_fitting(data[i], time)

                if self.plot_type == 3:  #Plots either dot, line or column based on self.plot_type
                    plt.bar(time-0.4+i*offset, self.final_function(data[i], func, time), width = offset, label = drug_list[i])
                elif self.plot_type == 2:
                    plt.plot(time, self.final_function(data[i], func, time), label = drug_list[i])
                else:
                    plt.plot(time, self.final_function(data[i], func, time), 'ro', marker = markers[i], label = drug_list[i], color = np.random.rand(3,))

                plt.legend()
                plt.title(gender + ' i ' + region + ' alder ' + alder)
        else:
            func = self.curve_fitting(data, time = time)

            if self.plot_type == 3:  #Plots either dot, line or column based on self.plot_type
                plt.bar(time, self.final_function(data, func, time), label = drug_list)
            elif self.plot_type == 2:
                plt.plot(time, self.final_function(data, func, time), label = drug_list)
            else:
                plt.plot(time, self.final_function(data, func, time), 'ro', label = drug_list)
            plt.legend()
            plt.title(gender + ' i ' + region + ' alder ' + alder)

        plt.xlabel('År')
        plt.ylabel(label)

        if save_fig == False:
            pass
        else:
            if not os.path.exists('Figures'):  #Save the plotes in a sub folder with name Figures
                os.makedirs('Figures')
            try:
                plt.savefig('Figures\\' + str(save_fig))  #Actually saving the plots.
            except:
                plt.savefig('Figures\\' + str(save_fig) + '.png')
            os.chdir(path)
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


    def cake_plot(self, gender, region, data_drugs, data_tot_drugs, year, age_indexes, drug_list, save_fig = False):
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

        time = np.array([year])
        func = self.curve_fitting(data_tot_drugs, time)
        func_value = self.final_function(data_tot_drugs, func, time)
        x = []
        explosion = []

        for i in range(len(data_drugs)):
            func = self.curve_fitting(data_drugs[i], time)
            x.append(self.final_function(data_drugs[i], func, time)/func_value)
            explosion.append(0.05)

        if self.med_type_index == 'Not given':  #Checks if there is a file with the same name as the folder to use as 100% of the cake diagram
            pass
        else:
            explosion.append(0.05)
            x.append(1-np.sum(np.array(x)))
            drug_list.append('Resterende '+ self.folder_name)

        if x[-1] < 0:  #If the sum of the data is greater than the data from the file with the same name as the folder remove it.
            del x[-1]
            del explosion[-1]
            del drug_list[-1]

        plt.figure(figsize = [12, 4.8])
        plt.pie(x, explode = explosion, labels = drug_list, autopct='%1.1f%%', shadow=True)
        plt.title(gender + ' i ' + region + ' alder ' + alder + ' år ' + str(year))

        if save_fig == False:
            pass
        else:
            try:
                plt.savefig(save_fig)
            except:
                plt.savefig(save_fig + '.png')

        plt.show()

        return None


    def medisinforbruk(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = False, save_fig = False, label = False):
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
        save_fig     -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """

        age_indexes = self.age_parameters(age_start, age_end)  #The correct indexes in self.age_indexes based on the age arguments

        data = self.drug_array(age_indexes, region, gender)

        if ratio:
            total_use = np.copy(data[self.med_type_index])
            data_drugs = np.delete(data, self.med_type_index, 0)
            ratio_data = data_drugs/total_use
            ratio_list = self.drugs[:]
            del ratio_list[self.med_type_index]

            if label == False:
                self.part1_plotting(ratio_data, period_start, period_end, ratio_list, age_indexes, gender, region, label = 'Ratio ' + self.folder_name + '/(Legemiddel X)', save_fig = save_fig)
            else:
                self.part1_plotting(ratio_data, period_start, period_end, ratio_list, age_indexes, gender, region, label = label, save_fig = save_fig)
        else:
            if label == False:
                self.part1_plotting(data, period_start, period_end, self.drugs, age_indexes, gender, region, save_fig = save_fig)
            else:
                self.part1_plotting(data, period_start, period_end, self.drugs, age_indexes, gender, region, save_fig = save_fig, label = label)


    def medisinforbruk_i(self, drug, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = False, save_fig = False, label = False):
        """
        Gives a bar plot of the medical use for a specific medicine in Norway based on the data in the folder and for the specified parameters. The plot can either be of ratios or total use.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        drug         -> The medicine the plot is for (string) or list. The string must be exactly the same as the name of the datafile in the folder without the .xls or .csv ending.
        gender       -> The gender (string) must be the same as the string given in the help function.
        region       -> The region (string) must be the same as the string given in the help function.
        age_start    -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end      -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end   -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                        period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        ratio        -> Either True or something else and chooses whether to plot the ratio or total use.
        save_fig     -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """

        age_indexes = self.age_parameters(age_start, age_end)  #The correct indexes in self.age_indexes based on the age arguments
        data = self.drug_array(age_indexes, region, gender)
        if type(drug) == type(['yolo']):
            pass
        else:
            drug = [drug]

        med_index = []
        for i in range(len(drug)):
            med_index.append(self.drugs.index(drug[i]))

        drugs = []
        for i in med_index:
            drugs.append(self.drugs[i])

        if ratio:
            total_use = np.copy(data[self.med_type_index])
            data_drugs = np.delete(data, self.med_type_index, 0)
            ratio_data = data/total_use
            if label == False:
                self.part1_plotting(ratio_data[med_index], period_start, period_end, drugs, age_indexes, gender, region, label = 'Ratio ' + self.folder_name + '/(Legemiddel X)', save_fig = save_fig)
            else:
                self.part1_plotting(ratio_data[med_index], period_start, period_end, drugs, age_indexes, gender, region, label = label, save_fig = save_fig)
        else:
            if label == False:
                self.part1_plotting(data[med_index], period_start, period_end, drugs, age_indexes, gender, region, save_fig = save_fig)
            else:
                self.part1_plotting(data[med_index], period_start, period_end, drugs, age_indexes, gender, region, save_fig = save_fig, label = label)


    def forhold_medisin(self, teller = None, nevner = None, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, save_fig = False, label = False):
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
        save_fig     -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """

        if teller == None or nevner == None:
            print('You have to specify both the \'teller\' and \'nevner\' variables')
            sys.exit()

        age_indexes = self.age_parameters(age_start, age_end) #The correct indexes in self.age_indexes based on the age arguments
        data = self.drug_array(age_indexes, region, gender)

        if type(teller) == type(['yolo']):  #Checks whether the teller or nevner is of the type list, if not make them so
            pass
        else:
            teller = [teller]

        if type(nevner) == type(['yolo']):
            pass
        else:
            nevner = [nevner]

        teller_indexes = []
        nevner_indexes = []

        for i in range(len(self.drugs)):
            if self.drugs[i] in teller:
                teller_indexes.append(i)
            if self.drugs[i] in nevner:
                nevner_indexes.append(i)
            else:
                pass

        legend_string = '('
        for i in teller:
            legend_string += i + ' + '

        legend_string = legend_string[:-3] + ')/('

        for i in nevner:
            legend_string += i + ' + '

        legend_string = legend_string[:-3] + ')'

        if label == False:
            self.part1_plotting(np.sum(data[teller_indexes], axis = 0)/np.sum(data[nevner_indexes], axis = 0), period_start, period_end, legend_string, age_indexes, gender, region, label = 'Ratio', save_fig = save_fig)
        else:
            self.part1_plotting(np.sum(data[teller_indexes], axis = 0)/np.sum(data[nevner_indexes], axis = 0), period_start, period_end, legend_string, age_indexes, gender, region, label = label, save_fig = save_fig)

        return None


    def kake_medisinforbruk(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, year = 2004, save_fig = False):
        """
        Gives a cake plot of the ratios medicine/(medicine group) for all the medicines in the data folder.
        The plotted data is for the chosen gender in a specific region for a specific age group.
        gender        -> The gender (string) must be the same as the string given in the help function.
        region        -> The region (string) must be the same as the string given in the help function.
        age_start     -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end       -> The oldest age the data is chosen from (int number from age_start -> infinity)
        year          -> The year the cake plot is for which is an integer from 0 -> infinity but it's recommended to stay within +- 10 years of the actual data.
        save_fig     -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """

        age_indexes = self.age_parameters(age_start, age_end) #The correct indexes in self.age_indexes based on the age arguments

        data = self.drug_array(age_indexes, region, gender)

        if self.med_type_index == 'Not given':
            total_use = np.copy(np.sum(data, axis = 0))
            data_drugs = np.copy(data)
            drug_list = self.drugs.copy()
        else:
            total_use = np.copy(data[self.med_type_index])
            data_drugs = np.delete(data, self.med_type_index, 0)
            drug_list = self.drugs.copy()
            del drug_list[self.med_type_index]

        self.cake_plot(gender, region, data_drugs, total_use, year, age_indexes, drug_list, save_fig = save_fig)


    def medisinforbruk_tidsutviling(self, drug, gender = 'Kvinne', region = 'Hele landet', ratio = False, year = False, save_fig = False):
        """
        An animation of the time evolution of either the ratio or total number of users of a specific medicine. The ratio is the medicine/(medicine type)
        drug    -> The name of the medicine as given in the data folder but whithout the .csv or .xls ending.
        gender  -> The gender (string) must be the same as the string given in the help function.
        region  -> The region (string) must be the same as the string given in the help function.
        ratio   -> Either True for ratio or something else for no ratio. It plots the ratio of medicine/(medicine type).
        year    -> A specific year to see the plot instead of a animation
        """

        if type(drug) == type([]):
            print('The \'drug\' argument has to be a string')
            sys.exit()

        age_indexes = self.age_parameters(0, 100)  #Including all indexes for all ages
        med_index = self.drugs.index(drug)  #The index of the specified drug
        med_dict = self.data[med_index]
        data = np.zeros((len(self.age_group_keys) ,len(self.year_keys)))
        x_axis = np.linspace(1, len(self.age_group_keys), len(self.age_group_keys))

        index = 0
        for k in self.age_group_keys:
            for i in range(len(self.year_keys)):
                year_index = self.year_keys[i]
                data[index][i] = med_dict[gender][year_index][k][region]
            index += 1

        if ratio:
            med_dict2 = self.data[self.med_type_index]
            data_tot = np.zeros((len(self.age_group_keys) ,len(self.year_keys)))
            index = 0

            for k in self.age_group_keys:
                for i in range(len(self.year_keys)):
                    year_index = self.year_keys[i]
                    data_tot[index][i] = med_dict2[gender][year_index][k][region]
                index += 1

            f = interpolate.interp1d(self.year_keys, data/data_tot, fill_value='extrapolate')  #Interpolating the data to make a smooth animation
        else:
            f = interpolate.interp1d(self.year_keys, data, fill_value='extrapolate')  #Interpolating the data to make a smooth animation

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

        if year != False:
            if str(year).isdigit() == False:
                print('The \'year\' parameter must be a number not', year)
            else:
                init()
                update(year)
                if save_fig == False:
                    pass
                else:
                    if type(save_fig) == type('str'):
                        try:
                            plt.savefig(save_fig)
                        except:
                            plt.savefig(save_fig + '.png')
                    else:
                        print('save_fig must be a string.')
                        pass
                plt.show()
            return None

        try:
            __IPYTHON__
        except NameError:
            ani = FuncAnimation(fig, update, frames=np.linspace(self.year_keys[0], self.year_keys[-1], 250), init_func=init, blit=False, interval = 1, repeat = True)
            plt.show()
        else:
            ani = FuncAnimation(fig, update, frames=np.linspace(self.year_keys[0], self.year_keys[-1], 250), init_func=init, blit=True, interval = 40, repeat = True)
            return ani.to_html5_video()
        # conda install -c conda-forge ffmpeg ##Into the terminal made it work for me

    def medisiner_og_befolkning(self, prevalens, sykdom = None, gender = 'Mann', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, save_fig = False, label = False):
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
        save_fig      -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """

        if type(prevalens) == type([]):
            prevalens = self.probability(np.array(prevalens)/100)
        else:
            prevalens /= 100
        #Source SSB https://www.ssb.no/statbank/table/07459/

        age_indexes = self.age_parameters(age_start, age_end)  #The correct indexes in self.age_indexes based on the age arguments
        data = self.drug_array(age_indexes, region, gender)

        if region == 'Hele landet':  #The data from SSB do not have the variable Hele landet, this it's necessarry to sum up everything.
            if 'Hele landet' in self.p_places:
                p_data = self.population_array(age_indexes, region, gender)  #If the data is from reseptregisteret do nothing.
            else:  #If the data is from SSB and not having the variable Hele landet
                p_data = self.population_array(age_indexes, self.p_places[0], gender)
                for k in range(1,len(self.p_places)):
                    p_data += self.population_array(age_indexes, self.p_places[k], gender)
        else:
            p_data = self.population_array(age_indexes, region, gender)

        plotting_data =  np.zeros((len(self.drugs) + 1, len(self.year_keys)))
        plotting_data[:-1] = data
        plotting_data[-1] = p_data*prevalens
        drugs_name = self.drugs[:]
        try:
            drugs_name.append(sykdom + ' Prevalens: %.2f%%' %(prevalens*100))
        except:
            drugs_name.append('Prevalens: %.2f%%' %(prevalens*100))

        if label == False:
            self.part1_plotting(plotting_data, period_start, period_end, drugs_name, age_indexes, gender, region, label = 'Antall personer', save_fig = save_fig)
        else:
            self.part1_plotting(plotting_data, period_start, period_end, drugs_name, age_indexes, gender, region, label = label, save_fig = save_fig)


    def medisiner_og_befolkning2(self, prevalens, drug = 'Valproat', gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018, ratio = True, save_fig = False, label = False):
        """
        Returns a bar plot of either the ratio using a medicine/(number of people having the disease) or the number of people having the disease and the number of people using the medicine.
        prevalens    -> The probability for a disease or diseases, can be either a number or a list of numbers all in probability % not fractions.
                        For multiple prevalens it uses regular probability calcualtions see the probability function.
        drug         -> The medicine the plot is for (string) not using the .csv or .xls endings.
        gender       -> The gender (string) must be the same as the string given in the help function.
        region       -> The region (string) must be the same as the string given in the help function.
        age_start    -> The yougest age the data is chosen from (int number from 0 -> age_end)
        age_end      -> The oldest age the data is chosen from (int number from age_start -> infinity)
        period_start -> The earliest year in the plot (int number from 0 -> infinity) should be chosen somewhat close to the actual datapoints
        period_end   -> The last year in the plot (int number from period_start -> infinity) should be chosen somewhat close to the actual datapoints
                        period_start and period_end recommended to stay within +- 10 years of the earliest and latest data point from reseptregisteret.
        save_fig     -> Option to save the figure. If False the figure WON'T be saved. To save enter a string with .png or .jpg endings
        """


        if type(prevalens) == type([]):
            prevalens = self.probability(np.array(prevalens)/100)
        else:
            prevalens /= 100

        if type(drug) == type([]):
            pass
        else:
            drug = [drug]
        #Source SSB https://www.ssb.no/statbank/table/07459/

        age_indexes = self.age_parameters(age_start, age_end)  #The correct indexes in self.age_indexes based on the age arguments
        data = self.drug_array(age_indexes, region, gender)

        if region == 'Hele landet':
            if 'Hele landet' in self.p_places:
                p_data = self.population_array(age_indexes, region, gender)
            else:  #If the data is from SSB and not having the variable Hele landet
                p_data = self.population_array(age_indexes, self.p_places[0], gender)
                for k in range(1,len(self.p_places)):
                    p_data += self.population_array(age_indexes, self.p_places[k], gender)
        else:
            p_data = self.population_array(age_indexes, region, gender)

        med_index = []
        drugs = []
        for i in drug:
            try:
                med_index.append(self.drugs.index(i))
                drugs.append(i)
            except:
                print('Legemiddelet ' + i + ' finnes ikke blant datasettene, sjekk mulige skrivefeil.')

        if ratio:
            if label == False:
                self.part1_plotting(data[med_index]/(prevalens*p_data), period_start, period_end, drugs, age_indexes, gender, region, label = 'Ratio: Brukere av legemiddel X / Antall med sykdom Y', save_fig = save_fig)
            else:
                self.part1_plotting(data[med_index]/(prevalens*p_data), period_start, period_end, drugs, age_indexes, gender, region, label = label, save_fig = save_fig)
        else:
            drugs.append('Prevalens: %.2f%%' %(prevalens*100))
            data2 = np.zeros((len(med_index) + 1, len(data[0])))
            for i in range(len(med_index)):
                data2[i] = data[med_index[i]]
            data2[-1] = p_data*prevalens
            if label == False:
                self.part1_plotting(data2, period_start, period_end, drugs, age_indexes, gender, region, save_fig = save_fig)
            else:
                self.part1_plotting(data2, period_start, period_end, drugs, age_indexes, gender, region, save_fig = save_fig, label = label)


    def fodsler(self, births_sykdom = 98, drug = 'Valproat', prevalens = 0.7, save_fig = False, label = False, ratio = False):
        births = np.array([2, 12, 36, 106, 239, 403, 719, 1007, 1451, 1791, 2452, 2981, 3692, 4169, 4426, 4585, 4503, 4090, 3804, 3210, 2656, 2299, 1890, 1445, 1105, 748, 510, 306, 211, 107, 73, 40, 18, 10, 6])  #The birthnumbers from SSB for 2018 age 15-49, change these to update.
        # https://www.ssb.no/statbank/table/06990/
        age_indexes = self.age_parameters(15, 49)  #The correct indexes in self.age_indexes for fertile women.
        sums_birth = np.zeros(len(age_indexes))
        p_data = np.zeros(len(age_indexes))

        if type(prevalens) == type([]):
            prevalens = self.probability(np.array(prevalens)/100)
        else:
            prevalens /= 100

        med_users = np.zeros((len(self.year_keys), len(age_indexes)))

        for i in range(len(sums_birth)):
            sums_birth[i] += np.sum(births[:i*5 + 5]) - np.sum(births[:i*5])  #Summing up to match the age groups in self.age_keys
            p_data[i] = self.population_array([age_indexes[i]], 'Hele landet', 'Kvinne')[self.year_keys.index(2018)]
            med_users[:, i] = self.drug_array([age_indexes[i]], 'Hele landet', 'Kvinne')[self.drugs.index(drug)]

        age_indexes2 = self.age_parameters(0, 100)
        prob_birth = np.copy(sums_birth/p_data)  #The probability of giving birth for a specific age group 15-19, 20-24 etc.
        drugs_tot = self.drug_array(age_indexes2, 'Hele landet', 'Kvinne')[self.drugs.index(drug)]

        ratio2 = births_sykdom/(np.sum(births)*prevalens)  #A ratio number to correct for the fact that less women using Antiepileptika is having children.
        births = np.sum(prob_birth*med_users, axis = 1)*ratio2

        if ratio == True:
            births_val_ratio = births/drugs_tot
            if label == False:
                self.part1_plotting(births_val_ratio, 2004, 2018, drug, age_indexes, 'Kvinne', 'Hele landet', save_fig = save_fig, label = 'Forholdet: fødlser blant ' + drug + ' brukere/antall ' + drug + ' brukere')
            else:
                self.part1_plotting(births_val_ratio, 2004, 2018, drug, age_indexes, 'Kvinne', 'Hele landet', save_fig = save_fig, label = label)
        else:
            if label == False:
                self.part1_plotting(births, 2004, 2018, drug, age_indexes, 'Kvinne', 'Hele landet', save_fig = save_fig, label = 'Antall fødsler blant ' + drug + ' brukere')
            else:
                self.part1_plotting(births, 2004, 2018, drug, age_indexes, 'Kvinne', 'Hele landet', save_fig = save_fig, label = label)



if __name__ == "__main__":
    test = visualization('Antiepileptika', 'dot')
    #print(path)
    #test2 = visualization('R')
    #test3 = visualization('R1')
    test.medisiner_og_befolkning2(prevalens = 2.5, ratio = True, drug = ['Valproat', 'Lamotrigin'], period_end = 2020)
    #test.fodsler(drug = 'Antiepileptika', ratio = True, save_fig = 'woopwoop.jpg')





    pass


os.chdir(path)





































#jao
