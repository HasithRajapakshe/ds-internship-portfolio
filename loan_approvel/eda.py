"""EDA: exploratory data analysis with heatmap and pairplot."""

import seaborn as sns
import matplotlib.pyplot as plt


def run_eda(df):
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
