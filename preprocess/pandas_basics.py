import numpy as np
import pandas as pd
# from numpy.random import randn
import random
from itertools import combinations
from pandas import DataFrame

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
    result = final_dataframe.corr(method ='pearson')
    numberOfColumns = result.shape[1]
    result = result.iloc[0: numberOfColumns]

    # numberOfColumns = result.shape[1]
    # finalReuslt = result.iloc[:, 0:numberOfColumns].values

    headers = list(result)
    
    id = []
    
    for i in range((len(intersection) - 1), -1, -1):
        id.append(headers.index(intersection[i]))

    # print(id)
    # print('..................')
    # print(result.iat[0, 0])
    # print('............')
    for i in range(numberOfColumns):
        c = 0
        for j in range(len(intersection)):
            if ( abs(result.iat[i, id[j]]) < 0.1 ):
                c = c + 1
        if c < len(intersection):
             final_dataframe.pop(headers[i])
    # print(df)
    return final_dataframe


dataframe = pd.read_csv('/home/kanababa/Desktop/DS and ML/basics/test.csv')

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

for i in range(0,10000):
    temp = random.choice(columns)
    if temp not in fixedColumn:
        fixedColumn.append(temp)
        c = c + 1
    if c == 7:
        break

c = 0

for i in range(0,10000):
    temp = random.choice(columns)
    if temp not in fixedColumn1:
        fixedColumn1.append(temp)
        c = c + 1
    if c == 7:
        break



df = dataframe.iloc[0: 500]
df1 = dataframe.iloc[501: 1000]

dataset1 = select_columns(df, fixedColumn)
dataset2 = select_columns(df1, fixedColumn1)

print(dataset1, dataset2)

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
merged_dataframe = pd.merge(df1, df2)              # optimal join result
# print(correlation(merged_dataframe, intersection))
# print('zxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
final_dataframe = merged_dataframe[:]
missing_value_ratio = 0                           # ratio of missing value to the optimal result
parameters = intersection[:]                        # attributes on which we got the optimal result
final_correlated_dataframe = final_dataframe[:]

print('correlated change')
print(final_correlated_dataframe.shape)

for x in range(1, len(intersection) + 1):
    print(" .....::::::: taking " + str(x) + " attributes :::::::::::..........")
    for i in combinations(intersection, x):
        temp_dataframe: DataFrame = pd.merge(df1, df2, how='left', left_on=list(i), right_on=list(i))
        if len(temp_dataframe.index) == len(df1.index): 
            print('temp dataframe')
            print(temp_dataframe.shape)
            correlated_dataframe = correlation(temp_dataframe, list(i))
            print('correlated_dataframe dataframe')
            print(correlated_dataframe.shape)
            ratio = count_missing_values(correlated_dataframe)
            print(ratio)
            if ratio > missing_value_ratio:
                missing_value_ratio = ratio
                final_dataframe = temp_dataframe[:]
                final_correlated_dataframe = correlated_dataframe[:]
                parameters = list(i)

print("..............")
print(final_correlated_dataframe)
final_correlated_dataframe.to_csv(r'out.csv')
print(parameters)
print(missing_value_ratio)


# missing value percentage
# number of parameters
# parameter






# ***********************************

# # series can hold any data types
# # series are so fast for retrieving something
# #  like hash tables
# # creates a series based on my_data
# series = pd.Series(my_data)

# # creates a series of key value by labels and my_data
# series = pd.Series(my_data, labels)

# # seed is used for make same random number always
# np.random.seed(101)
# # create a dataframe of 5x4
# # basically its a series
# dff = pd.DataFrame(randn(6, 3), ['A', 'B', 'C', 'D', 'E', 'F'], ['W', 'X', 'Y'])
# print(dff)

# # selecting columns
# col = df['W']

# # add a new column
# df['new'] = df['W'] + df['X']

# # delete a column
# # if inplace is true then it will deleted from mail df
# # if false then main df still same
# df.drop('new', axis=1, inplace=True)

# # delete a row
# # axis comes from the shape of the matrix
# # row is the first index of the shape and column is the
# # second index of the shape like (5,4) where 5 is in index 0
# df.drop('E', axis=0, inplace=False)

# # selecting rows
# row = df.loc['A']
# row = df.iloc[2]

# # select a specific row value
# row = df.loc['B', 'Z']

# # selecting both row and column
# row = df.loc[['A', 'B'], ['X', 'Z']]

# # conditional selection
# dff = df[df > 0]

# # conditional selection of column
# # it removes row C because its W value is less then 0
# dfff = df[df['W'] > 0]

# # conditional selection of row
# # it removes all the rows except C because all the  values are less then 0
# dff = df[df['Z'] < 0]

# # two or more conditions
# # we can't use and because this and can't handle a series of boolean
# # and handles only single boolean value
# # for or we use |
# dff = df[(df['W'] > 0) & (df['Y'] > 1)]

# # reset index which reset the index column and make a new index column
# dfff = df.reset_index()

# # add new column in existing data frame
# newind = 'AS HJ KL KJ LO'.split()
# df['States'] = newind

# # make this new column to the index of DF
# df = df.set_index('States')
# print(df)
