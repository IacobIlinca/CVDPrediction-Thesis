import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

def compile_metrics_table(file_path='results/classifier_metrics2.csv'):
    df_metrics = pd.read_csv(file_path)
    # Rename column from 'Classififer' to 'Classifier'
    df_metrics.rename(columns={'Classififer': 'Classifier'}, inplace=True)
    # Format numerical columns to keep only four digits after the decimal point
    for col in df_metrics.select_dtypes(include=['float64', 'float32']):
        df_metrics[col] = df_metrics[col].apply(lambda x: f'{x:.4f}')
    return df_metrics

# Compile the big table
big_table = compile_metrics_table()

def plot_and_save_table_as_png(df, filename='big_table.png'):
    # Set the figure size and color
    fig, ax = plt.subplots(figsize=(10, df.shape[0] * 0.625))  # Adjust size as needed
    ax.axis('off')
    # Define column widths to reduce whitespace, here assuming uniform width across columns
    col_widths = [0.1] * len(df.columns)  # You may need to adjust this based on the content
    # Increase column widths if text is cramped
    col_widths = [0.2 if col == 'Classifier' else 0.1 for col in df.columns]
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=col_widths)

    # Style the table with font size and auto scale
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(2, 2)  # You might need to adjust the scaling factor

    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Plot and save the big table as a PNG
plot_and_save_table_as_png(big_table, 'results/compiled_metrics_table2.png')
