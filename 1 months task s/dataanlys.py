import pandas as pd

df_smartphone = pd.read_csv('Smartphone_Usage_Productivity_Dataset_50000.csv')
df_student = pd.read_csv('Student_data.csv')
df_performance = pd.read_csv('StudentPerformanceFactors.csv')

datasets = {
    "Smartphone Usage": df_smartphone,
    "Student Data": df_student,
    "Student Performance Factors": df_performance,
}
for name, df in datasets.items():
    print("shape", df.shape)
    print("\ndata types", df.dtypes)
    print("\describe", df.describe())
