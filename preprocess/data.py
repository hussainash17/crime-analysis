import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./readydata.csv')
# data.index = pd.DatetimeIndex(data.iyear)
iyeardata = data['iyear'].value_counts().iloc[:15].index

plt.figure(figsize= (15,10))
sns.countplot(y = 'iyear', data = data, order = iyeardata)

# plt.plot(data.resample('Y').size())
# plt.title('Test')
# plt.xlabel('X')
# plt.ylabel('Y')

plt.show()