import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import xlrd
import time
import warnings
import statsmodels.api as sm
import sys
from matplotlib.animation import FuncAnimation


warnings.simplefilter('ignore', np.RankWarning)

path = os.path.abspath(os.path.dirname(__file__))

class files(object):

    def __init__(self, folder_name):
        self.directory = folder_name

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
        years = np.zeros(len(indexes))
        age_groups = []
        places = []
        genders = []
        for g in range(indexes[0], indexes[0] + d_indexes):
            age_groups.append(self.df.iloc[g,2])
            places.append(self.df.iloc[g,4])
            genders.append(self.df.iloc[g,3])

        age_groups = self.ordering(age_groups)
        genders = self.ordering(genders)
        places = self.ordering(places)
        users = np.zeros(shape = (len(genders),len(indexes), int(d_indexes/len(genders))))

        #The first index in the excel file is the row while the second is the column
        for k in range(len(indexes)):
            years[k] = (self.df.iloc[indexes[k],1])
            counter = 0
            index_counter_for_gender = 0
            prev_count = 0

            for j in range(indexes[k], indexes[k]+d_indexes):
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

                users[index_counter_for_gender, k, counter] = use
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

        return users_dict

    def population_excel(self, file_name, stop_at_90 = True):
        """
        Reads the population data from SSB
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
                if 'Finnmark' in self.df.iloc[i,0][3:]:
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
                        if 'Trøndelag' in place:  #Sum such that the only key is Trøndelag
                            if summation:
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

        return users_dict





class visualization(object):

    def __init__(self, folder_name):
        self.folder_name = folder_name

        os.chdir(self.folder_name)
        self.filenames = os.listdir()
        os.chdir(path)
        try:
            self.filenames.remove('Befolkning.xlsx')
        except:
            pass

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
        if age_start >= 90:
            age_indexes.append(len(self.age_group_keys)-1)
        else:
            for i in range(0, len(self.age_group_keys)):
                if age_start < 5*(i+1) and age_end >= 5*(i):
                    age_indexes.append(i)

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


    def part1_plotting(self, data, period_start, period_end, drug_list, age_indexes, gender, region, label = 'Antall utskrivninger'):

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
            plt.xlabel('År')
            plt.ylabel(label)
            plt.show()
        else:
            func = self.curve_fitting(np.log(data))
            plt.bar(time, self.final_function(data, np.exp(func(time)), time), label = drug_list)
            plt.legend()
            plt.title(gender + ' i ' + region + ' alder ' + alder)
            plt.xlabel('År')
            plt.ylabel(label)
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
        self.part1_plotting(ratio, period_start, period_end, ratio_list, age_indexes, gender, region, label = 'Ratio')


    def individual(self, drug, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018):

        age_indexes = self.age_parameters(age_start, age_end)
        med_index = self.drugs.index(drug)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        ratio = data/total_use

        self.part1_plotting(data[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region)
        self.part1_plotting(ratio[med_index], period_start, period_end, self.drugs[med_index], age_indexes, gender, region, label = 'Ratio')


    def recommended(self, anbefalt = None, ikke_anbefalt = None, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 1995, period_end = 2030):

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


    def cake(self, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, year = 2004):

        age_indexes = self.age_parameters(age_start, age_end)
        med_type_index = self.drugs.index(self.folder_name)

        data = self.drug_array(age_indexes, region, gender)

        total_use = np.copy(data[med_type_index])
        data_drugs = np.delete(data, med_type_index, 0)
        drug_list = self.drugs.copy()
        del drug_list[med_type_index]

        self.cake_plot(gender, region, data_drugs, total_use, year, age_indexes, drug_list)


    def individual_time(self, drug, gender = 'Kvinne', region = 'Hele landet', ratio = False):

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
            ani = FuncAnimation(fig, update, frames=np.linspace(self.year_keys[0], self.year_keys[-1], 250), init_func=init, blit=False, interval = 40, repeat = True)
            plt.close()
            return ani.to_html5_video()
        # conda install -c conda-forge ffmpeg ##Into the terminal made it work for me


    def part3(self, prevalens, sykdom = 'Epilepsi', gruppe_antall = 56552, antall_tilfeller = 98, gender = 'Kvinne', region = 'Hele landet', age_start = 15, age_end= 49, period_start = 2004, period_end = 2018):
        chance_pr_person = antall_tilfeller/gruppe_antall

        #Source SSB https://www.ssb.no/statbank/table/07459/
        males_in_norway = np.array([151852, 164083, 164061, 165165, 176336, 189610, 186264, 181466, 179750, 194074, 187947, 166140, 152590, 136125, 125400, 76728, 47100, 27209, 13173])
        females_in_norway = np.array([143011, 155970, 155981, 155334, 164895, 181578, 178708, 170931, 170461, 184354, 178097, 159560, 151153, 136879, 130391, 87319, 62110, 44612, 31795])
        print(self.places)





if __name__ == "__main__":
    opening_test = files('Antiepileptika')
    opening_test.population_excel('Befolkning.xlsx')

    #test = visualization('Antiepileptika')
    #test.part1(gender = 'Mann')
    #test.part1(region='Hele landet', age_start = 15, age_end = 49, period_start = 1980, period_end = 2050)
    #test.individual('Valproat', period_start = 2004, period_end = 2018, age_start = 20, age_end = 35)
    #test.individual('Valproat', period_start = 2004, period_end = 2018, age_start = 15, age_end = 49)
    #test.individual('Valproat', period_start = 2004, period_end = 2018, age_start = 15, age_end = 49, gender = 'Mann')
    #test.recommended(ikke_anbefalt = ['Valproat'])
    #test.individual_time('Valproat', gender = 'Mann')
    #test.part3(0.7)


os.chdir(path)

"""
Del 3:
Ratio og Antall brukere
"""





































#jao
