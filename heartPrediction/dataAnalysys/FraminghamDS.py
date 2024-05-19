import warnings

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = '../dataSets/framinghamDataSet.dta'

# Read the .dta file into a DataFrame
df = pd.read_stata(file_path)

f = open("../results/FraminghamDT/DataAnalysisFraminghamDTResults.txt", "w")

f.write("Data Analysis part for Framingham Dataset")

f.write("\n")
f.write("First 5 rows of the table")
f.write("\n")
f.write(df.head(5).to_string())
f.write("\n")
f.write("Shape of the dataSet")
f.write(df.shape.__str__())
f.write("\n")

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
f.write("0: " + str((df['cexam'] == 0).sum()))
f.write("\n")
f.write("1: " + str((df['cexam'] != 0).sum()))
f.write("\n")
f.write("Number of duplicated rows is: ")
f.write(str(df.duplicated().sum()))

f.write("\n")
f.write("Datatypes for each column:")
f.write("\n")
f.write(df.dtypes.to_string())

df_copy = df.copy()
# Mapping cexam values to meaningful categories in the copy
df_copy['cexam'] = df_copy['cexam'].replace({0: 'No Heart Disease', 1: 'Heart Disease', 2: 'Heart Disease', 3: 'Heart Disease', 4: 'Heart Disease',
                                             5: 'Heart Disease', 6: 'Heart Disease', 7: 'Heart Disease', 8: 'Heart Disease', 9: 'Heart Disease',
                                             10: 'Heart Disease', 11: 'Heart Disease', 12: 'Heart Disease', 13: 'Heart Disease', 14: 'Heart Disease',
                                             15: 'Heart Disease', 16: 'Heart Disease'})

plt.title('Heart Disease Over Sex')
sns.countplot(data=df_copy, x='sex', hue='cexam', palette=['green', 'red'])
plt.savefig('../results/FraminghamDT/heart_failure_over_sex_plot.png')

count = df_copy['cexam'].value_counts()

# Plotting pie chart
labels = count.index.tolist()
plt.figure(figsize=(5, 5))
plt.pie(count, labels=labels, autopct='%.0f%%', explode=(0, 0.1), colors=['red', 'green'])
plt.legend([ 'No Heart Disease', 'Heart Disease'], loc=1)
plt.title('Heart Disease Distribution')
plt.savefig('../results/FraminghamDT/heart_disease_pie_chart.png')

df.hist(figsize=(12, 10), color='red')
plt.savefig('../results/FraminghamDT/histogram_plot.png')

crosstab_df = pd.crosstab(df_copy.age, df_copy.cexam)
crosstab_df.plot(kind="bar", figsize=(20, 6), color=['red', 'green'])

# Add title, xlabel, ylabel
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('../results/FraminghamDT/heart_disease_frequency_for_ages_plot.png')

plt.close()
f.close()

# transform non-numerical labels
labelEncoder = LabelEncoder()
obj = df_copy.select_dtypes(include='object')
not_obj = df_copy.select_dtypes(exclude='object')
for i in range(0, obj.shape[1]):
    obj.iloc[:, i] = labelEncoder.fit_transform(obj.iloc[:, i])

df_new = pd.concat([obj, not_obj], axis=1)

print(df_new.head(5))

corr = df_new.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr.rank(axis='columns'), annot=True, fmt='.1f', linewidth=.5, cmap="Reds")
plt.savefig('../results/FraminghamDT/correlation_heatmap.png')

numeric_cols = df_new.select_dtypes(include='number').columns
cols_with_no_nulls = df_new.columns[df_new.isnull().mean() == 0]
selected_cols = list(set(numeric_cols).intersection(cols_with_no_nulls))

# figsize = (6*6, 20)
# fig = plt.figure(figsize=figsize)
#
# # Define colors for heart disease and normal instances
# colors = ['#FF0000', '#008000']  # Red and Green
#
# for idx, col in enumerate(selected_cols[:-1], start=1):
#     ax = plt.subplot(3, 3, idx % 9 + 1)
#     sns.kdeplot(data=df_new, hue='cexam', fill=True, x=col, legend=False, palette=colors)
#
#     ax.set_ylabel('')
#     ax.spines['top'].set_visible(False)
#     ax.set_xlabel('')
#     ax.spines['right'].set_visible(False)
#     ax.set_title(f'{col}', loc='right', weight='bold', fontsize=22)
#
# fig.suptitle(f'Features vs Target\n\n\n', ha='center', fontweight='bold', fontsize=25)
# fig.legend(['Heart Disease', 'Normal'], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=21, ncol=2)
#
# plt.tight_layout()
#
# fig.savefig('results/FraminghamDT/KernelDensityEstimate.png')

