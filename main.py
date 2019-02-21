import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.read_csv('../data/wine.data')
df = pd.read_csv('../data/dataR2.csv')
print(df.head())


# sns.distplot(df.values[1], kde=False)
# sns.distplot(df["Classification"], kde=False, bins=40)

# sns.swarmplot(x="Classification", y="BMI", data=df)
# sns.violinplot(x = "BMI", y="Age",hue = 'Classification', data = df)
# sns.regplot(x="BMI", y="Age",data=df)
plt.show()
