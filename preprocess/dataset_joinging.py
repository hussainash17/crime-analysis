from itertools import combinations
import pandas as pd
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


# give the directory of the two datasets
dataset1 = pd.read_csv(r'C:\Users\This PC\PycharmProjects\Crime\testdata1.csv')
dataset2 = pd.read_csv(r'C:\Users\This PC\PycharmProjects\Crime\testdata2.csv')

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
final_dataframe = pd.merge(df1, df2)              # optimal join result
missing_value_ratio = 0                           # ratio of missing value to the optimal result
parameters = intersection                         # attributes on which we got the optimal result

for x in range(1, len(intersection) + 1):
    print(" .....::::::: taking " + str(x) + " attributes :::::::::::..........")
    for i in combinations(intersection, x):
        temp_dataframe: DataFrame = pd.merge(df1, df2, how='left', left_on=list(i), right_on=list(i))
        ratio = count_missing_values(temp_dataframe)
        print(ratio)
        if ratio > missing_value_ratio:
            missing_value_ratio = ratio
            final_dataframe = temp_dataframe
            parameters = list(i)

print("..............")
print(final_dataframe)
print(parameters)
print(missing_value_ratio)
