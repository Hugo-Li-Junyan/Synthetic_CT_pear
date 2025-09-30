import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_vae_log(csv_path, columns):
    """
    Reads a CSV file, selects specified columns, and plots them as line plots on the same figure.

    Parameters:
        csv_path (str): Path to the CSV file.
        columns (list of str): List of exactly four column names to plot.
    """

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Check if all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")

    # Set scientific style
    sns.set(style="whitegrid", context="talk", palette="muted")

    plt.figure(figsize=(12, 6))
    # Plot each column
    labels = ['Train Loss', 'Validation Loss']
    for i, col in enumerate(columns):
        sns.lineplot(data=df, x=df.index, y=col, label=labels[i])

    #plt.ylim(-0.002, 0.01)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_vae_log(r'J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250926-120749\diffuser_log.csv', ['Train Loss', 'Validation Loss'])