import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '../results/classifier_metrics3.csv'
data = pd.read_csv(file_path)

# Rename columns and adjust data as needed
data.rename(columns={'Classififer': 'Classifier', 'DataSet': 'Dataset'}, inplace=True)
data['Adjusted Classifier'] = data['Classifier'].apply(lambda x: 'Extra Tree' if x == 'Extra Tree' else 'Extra Tree with Optuna' if x == 'Extra Tree with Optuna' else 'Others')

# Filter data for specific classifiers if necessary
data = data[data['Classifier'].str.contains('Extra Tree')]

# Create the plot
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(data=data, x='Dataset', y='Accuracy', hue='Adjusted Classifier', palette=['orange', 'purple'], dodge=True)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
bar_plot.legend(title='Classifier Type')
plt.grid(True, axis='y')

# Adding the text annotations for accuracy on each bar, omitting any zero values
for p in bar_plot.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero values
        bar_plot.annotate(format(height, '.3f'),
                          (p.get_x() + p.get_width() / 2., height),
                          ha = 'center', va = 'center',
                          xytext = (0, 9),
                          textcoords = 'offset points')

# Save the plot
output_file_path = 'optunaImrpovements.png'
plt.savefig(output_file_path)
plt.show()
