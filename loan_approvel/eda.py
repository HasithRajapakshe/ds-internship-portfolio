"""EDA"""
import matplotlib.pyplot as plt
import seaborn as sns


def run(df):
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',  # print the number
                vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Heatmap")

    plt.show()
