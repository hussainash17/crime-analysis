import numpy as np
import pandas as pd
# from numpy.random import randn
import random
from itertools import combinations
from pandas import DataFrame
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

sns.set_theme(style="white")


def generateHexbin(firstColumn, second_Column):
    sns.jointplot(x=firstColumn, y=second_Column, kind="reg", color="#4CB391")
    plt.savefig('regression.png', dpi=300, bbox_inches='tight')
    sns.jointplot(x=firstColumn, y=second_Column, kind="kde", color="#4CB391")
    plt.savefig('kdejointplot.png', dpi=300, bbox_inches='tight')
    plt.show()


def kdePlot(df):
    numberOfColumns = df.shape[1]
    for i in range(numberOfColumns):
        if is_numeric_dtype(df.iloc[:, i]):
            ax = sns.kdeplot(df.iloc[:, i], shade=True, color='b')
            # plt.title('KDE Plot')
            # plt.ylabel('Density')
            # plt.savefig('kde.png', dpi=300, bbox_inches='tight')


def generatehistogram(dataset, name):
    # Generate a mask for the upper triangle
    result = dataset.corr()
    mask = np.triu(np.ones_like(result, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(result, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

# checks if a field value is missing


def is_nan(num):
    return num != num

# returns a ratio of the missing value diveded by the size of a dataset


def count_missing_values(dataframe):

    row = len(dataframe)
    col = len(dataframe.columns)
    total = row * col
    count = 0
    for a in range(row):
        for b in range(col):
            if not is_nan(dataframe.iat[a, b]):
                count += 1
    r = count / total
    return r


def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame


def correlation(final_dataframe, intersection):
    result = final_dataframe.corr(method='pearson')
    numberOfColumns = result.shape[1]
    result = result.iloc[0: numberOfColumns]

    # numberOfColumns = result.shape[1]
    # finalReuslt = result.iloc[:, 0:numberOfColumns].values

    headers = list(result)

    id = []

    for i in range((len(intersection) - 1), -1, -1):
        id.append(headers.index(intersection[i]))

    for i in range(numberOfColumns):
        c = 0
        for j in range(len(intersection)):
            if (abs(result.iat[i, id[j]]) < .1):
                c = c + 1
        if c < len(intersection):
            final_dataframe.pop(headers[i])
    # print(df)
    return final_dataframe

def correlation1(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    print(dataset)

# def solve(df, t1, t2):
#     titles = list(df)
#     # if 'longitude_x' and 'longitude_y' in titles:
#     #     del df['longitude_y']
#     #     df.rename(columns={ 'longitude_x' : 'longitude'}, inplace=True)
#     # if 'latitude_x' and 'latitude_y' in titles:
#     #     del df['latitude_y']
#     #     df.rename(columns={ 'latitude_x' : 'latitude'}, inplace=True)
#     for i in range(0, len(t1)):
#         temp_string1 = t1[i] + '_x'
#         temp_string2 = t1[i] + '_y'
#         if temp_string1 and temp_string2 in titles:
#             del df[temp_string2]
#             df.rename(columns={ temp_string1 : t1[i]}, inplace=True)
#             titles = list(df)
#     for i in range(0, len(t2)):
#         temp_string1 = t2[i] + '_x'
#         temp_string2 = t2[i] + '_y'
#         if temp_string1 and temp_string2 in titles:
#             del df[temp_string2]
#             df.rename(columns={ temp_string1 : t2[i]}, inplace=True)
#             titles = list(df)
            
dataframe = pd.read_csv(r'/home/kanababa/Desktop/DS and ML/basics/test.csv')
# print(dataframe)
correlation1(dataframe, 0.9)
fl = open('output.txt', 'a')
fl.write('##################################' + ' \n' + ' \n' + ' \n')
fl.write('both initial dataset dimension: 5000 x 10' + ' \n')

columns = dataframe.columns

fixedColumn = [
    'eventid',
    'latitude',
    'longitude'
]

fixedColumn1 = [
    'eventid',
    'latitude',
    'longitude'
]


c = 0

for i in range(0, 10000):
    temp = random.choice(columns)
    if temp not in fixedColumn:
        fixedColumn.append(temp)
        c = c + 1
    if c == 7:
        break

c = 0

for i in range(0, 10000):
    temp = random.choice(columns)
    if temp not in fixedColumn1:
        fixedColumn1.append(temp)
        c = c + 1
    if c == 7:
        break


df = dataframe.iloc[0: 500]
df1 = dataframe.iloc[0: 500]


dataset1 = select_columns(df, fixedColumn)
# generateGraph(dataset1)
dataset2 = select_columns(df1, fixedColumn1)
# generateGraph(dataset2)
dataset1.to_csv(r'firstdataset.csv')
dataset2.to_csv(r'seconddataset.csv')

# is extracts the headers of the dataset
list1 = list(dataset1)
list2 = list(dataset2)

# formatting the the headers to the lowercase
for i in range(len(list1)):
    list1[i] = list1[i].lower()
for i in range(len(list2)):
    list2[i] = list2[i].lower()
# print(list1)
# print(list2)

# .shape[1] excludes the line numbers and headers from the dataset
# .iloc[range] converts the csv values within the range to list of lists
numberOfColumns1 = dataset1.shape[1]
data1 = dataset1.iloc[:, 0:numberOfColumns1].values
numberOfColumns2 = dataset2.shape[1]
data2 = dataset2.iloc[:, 0:numberOfColumns2].values
# print(data1)
# print(data2)

# converting all the list values to dataframe structure
# you can check the format of dataframe using print
df1 = pd.DataFrame([data1[0]], columns=list1)
for i in range(1, len(data1)):
    df = pd.DataFrame([data1[i]], columns=list1)
    df1 = df1.append(df, ignore_index=True)

df2 = pd.DataFrame([data2[0]], columns=list2)
for i in range(1, len(data2)):
    df = pd.DataFrame([data2[i]], columns=list2)
    df2 = df2.append(df, ignore_index=True)

# colab gives an error for this line. colab cant convert the set into a list
# intersection holds the common attributes betweens the two datasets
intersection = list(set(list1).intersection(list2))
# print(intersection)

# initializing with a value, which will be updated later within the loops
# merged_dataframe = pd.merge(df1, df2)              # optimal join result
merged_dataframe = []
# print(correlation(merged_dataframe, intersection))
# print('zxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
# final_dataframe = merged_dataframe[:]
final_dataframe = []
# ratio of missing value to the optimal result
missing_value_ratio = 0
# attributes on which we got the optimal result
# parameters = intersection[:]
parameters = []
# final_correlated_dataframe = final_dataframe[:]
final_correlated_dataframe = []

# print('correlated change')
# print(final_correlated_dataframe.shape)

temp = []

for i in range(1, len(intersection) + 1):
    temp += list(combinations(intersection, i))

numberOfIterations = 10

finalMergedDataframe = []

# for x in range(1, len(intersection) + 1):
#     print(" .....::::::: taking " + str(x) +
#           " attributes :::::::::::..........")
#     fl.write('taking ' + str(x) + ' attributes in common.......' + ' \n')
for j in range(0, min(len(intersection) + 1, numberOfIterations)):
    i = random.choice(temp)
    temp.remove(i)
    fl.write('common attributes taken' + str(i) + ' \n')
    temp_dataframe: DataFrame = pd.merge(
        df1, df2, how='inner', left_on=list(i), right_on=list(i))
    solve(temp_dataframe, list(dataset1), list(dataset2))
    print(temp_dataframe.shape)
    if len(temp_dataframe.index) == len(df1.index):
        fl.write('merged dataset dimension: ' +
                 str(temp_dataframe.shape) + ' \n')
        tempCopy = copy.deepcopy(temp_dataframe)
        correlated_dataframe = correlation(tempCopy, list(i))
        fl.write('dimension after running correlation: ' +
                 str(correlated_dataframe.shape) + ' \n')
        if len(correlated_dataframe.columns) > 0:
            ratio = count_missing_values(correlated_dataframe)
            fl.write('missing value ratio: ' + str(1.0 - ratio) + ' \n')
        else:
            ratio = -1
            print('all columns dropped')
        if ratio > missing_value_ratio:
            finalMergedDataframe = copy.deepcopy(temp_dataframe)
            print("..................")
            print(finalMergedDataframe.shape)
            missing_value_ratio = copy.deepcopy(ratio)
            # final_dataframe = temp_dataframe[:]
            final_correlated_dataframe = copy.deepcopy(correlated_dataframe)
            print(final_correlated_dataframe.shape)
            print("..................")
            parameters = copy.deepcopy(list(i))

print("finalMergedDataframe size")
print(finalMergedDataframe.shape)
print('final_correlated_dataframe size')
print(final_correlated_dataframe.shape)
final_correlated_dataframe.to_csv(r'finalResult.csv')
print(finalMergedDataframe)
print("..............")
print(final_correlated_dataframe)
# kdePlot(finalMergedDataframe)
generatehistogram(finalMergedDataframe, 'mergedData.png')
# fArray = final_correlated_dataframe.iloc[:, 1].values
# sArray = final_correlated_dataframe.iloc[:, 0].values
# generateHexbin(fArray, sArray)\
kdePlot(final_correlated_dataframe)
generatehistogram(final_correlated_dataframe, 'finalData.png')
# sns.pairplot(final_correlated_dataframe, kind="hist")
# plt.show()
final_correlated_dataframe.to_csv(r'out.csv')
print(parameters)
print(missing_value_ratio)
fl.write('best output among those runs(according to missing value ratio):' + ' \n')
fl.write('parameters: ' + str(parameters) + ' \n')
fl.write('missing value ratio: ' + str(1.0 - missing_value_ratio) +
         ' \n' + ' \n' + ' \n' + ' \n')


fl.close()


# missing value percentage
# number of parameters
# parameter
