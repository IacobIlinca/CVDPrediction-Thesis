import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('results/classifier_metrics.csv')

# Filter the DataFrame to include only rows where 'DataSet' is 'F'
filtered_df = df[df['DataSet'] == 'HD']

# Remove rows where the 'Classififer' column is 'Nearest Centroid'
filtered_df = filtered_df[filtered_df['Classififer'] != 'Nearest Centroid']

# Rename the 'Classififer' column to 'Classifier'
filtered_df.rename(columns={'Classififer': 'Classifier'}, inplace=True)

# Format numerical columns to keep only four digits after the decimal
for col in filtered_df.select_dtypes(include=['float64']).columns:
    filtered_df[col] = filtered_df[col].apply(lambda x: f'{x:.4f}')

# Plotting the Accuracy for each Classifier with lines connecting the dots
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_df)))
accuracies = filtered_df['Accuracy'].astype(float).tolist()  # Convert to list for easier handling
x_ticks = list(range(len(filtered_df)))
scatter = plt.scatter(x_ticks, accuracies, c=colors, s=100, edgecolor='k', label='Accuracy')
plt.plot(x_ticks, accuracies, 'k-', alpha=0.5)  # Connect points with a black line

# Adding data labels for each point using a loop with correct indexing
for i, accuracy in enumerate(accuracies):
    plt.annotate(f'{accuracy}', (i, accuracy), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(x_ticks, filtered_df['Classifier'].tolist(), rotation=45, ha="right")  # Convert to list for safety
plt.title('Accuracy per Classifier')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.colorbar(scatter, label='Position Index Color Scale')

# Save and show the plot
plt.tight_layout()  # Adjust layout to make room for label rotation
plt.savefig('results/finalResults/accuracy_plotHD.png')
plt.show()


# Display and save the table using matplotlib
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=filtered_df.values, colLabels=filtered_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.savefig('results/finalresults/resultsHD.png')  # Specify the path and filename for saving the image
plt.show()
