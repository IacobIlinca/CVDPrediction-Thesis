import warnings

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

f = open("../results/HeartFailureDT/DataAnalysisHeartFailureDTResults.txt", "w")

df = pd.read_csv('../dataSets/heartFailureDataSet.csv')

f.write("Data Analysis part for Heart Failure Dataset")

f.write("\n")
f.write("First 5 rows of the table")
f.write("\n")
f.write(df.head(5).to_string())
f.write("\n")
f.write("Shape of the dataSet")
f.write(df.shape.__str__())
f.write("\n")
# f.write("Information about the dataset")
# f.write(df.info())
f.write("\n")

f.write("Basic statistical details like percentile, mean, std, etc. of a data frame ")
f.write("\n")
f.write(df.describe().to_string())
f.write("\n")

f.write("The count of null values for each column")
f.write("\n")
f.write(df.isna().sum().to_string())
f.write("\n")

f.write("How many people suffered from heart diseases (0-no, 1-yes)")
f.write("\n")
f.write(df['HeartDisease'].value_counts().to_string())
f.write("\n")
f.write("Number of duplicated rows is: ")
f.write(str(df.duplicated().sum()))

f.write("\n")
f.write("Datatypes for each column:")
f.write("\n")
f.write(df.dtypes.to_string())

plt.title('Heart Failure Over Sex')
sns.countplot(data=df, x='Sex', hue='HeartDisease', palette=['pink', 'purple'])
plt.savefig('../results/HeartFailureDT/heart_failure_over_sex_plot.png')  # Save the plot to a file

labels = ['1', '0']
count = df['HeartDisease'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(count, labels=labels, autopct='%.0f', explode=(0, .1), colors=['purple', 'pink'])
plt.legend(['heart disease', 'Normal'], loc=1)
plt.title('Heart Disease')

# Save the plot to a file
plt.savefig('../results/HeartFailureDT/heart_disease_pie_chart.png')

df.hist(figsize=(12, 10), color='pink')
plt.savefig('../results/HeartFailureDT/histogram_plot.png')

crosstab_df = pd.crosstab(df.Age, df.HeartDisease)
crosstab_df.plot(kind="bar", figsize=(20, 6), color=['pink', 'purple'])

# Add title, xlabel, ylabel
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Save the plot to a file
plt.savefig('../results/HeartFailureDT/heart_disease_frequency_for_ages_plot.png')

plt.close()
f.close()

# transform non-numerical labels
labelEncoder = LabelEncoder()
obj = df.select_dtypes(include='object')
not_obj = df.select_dtypes(exclude='object')
for i in range(0, obj.shape[1]):
    obj.iloc[:, i] = labelEncoder.fit_transform(obj.iloc[:, i])

df_new = pd.concat([obj, not_obj], axis=1)

print(df_new.head(5))

corr = df_new.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr.rank(axis='columns'), annot=True, fmt='.1f', linewidth=.5, cmap="RdPu")
plt.savefig('../results/HeartFailureDT/correlation_heatmap.png')

figsize = (6*6, 20)
fig = plt.figure(figsize=figsize)

# Define colors for heart disease and normal instances
colors = ['#FF69B4', '#800080']

for idx, col in enumerate(df_new[:-1], start=1):
    ax = plt.subplot(3, 3, idx % 9 + 1)
    sns.kdeplot(data=df_new, hue='HeartDisease', fill=True, x=col, legend=False, palette=colors)

    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', weight='bold', fontsize=22)

fig.suptitle(f'Features vs Target\n\n\n', ha='center', fontweight='bold', fontsize=25)
fig.legend(['Heart Disease', 'Normal'], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=21, ncol=2)

plt.tight_layout()

# Save the figure to a file
fig.savefig('../results/HeartFailureDT/KernelDensityEstimate.png')
