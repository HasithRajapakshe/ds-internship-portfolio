class DataCleaner:
    def __init__(self, data):
        self.data = data

    def detect_missing(self):
        print("Missing Values in Each Column:")
        print(self.data.isnull().sum(), "\n")

    def fill_missing(self):
        self.data['School ID'] = self.data['School ID'].fillna(
            self.data['School ID'].median())
        self.data['Address'] = self.data['Address'].fillna('Unknown')
        self.data['City'] = self.data['City'].fillna('Unknown')
        self.data['Marks'] = self.data['Marks'].fillna(
            self.data['Marks'].median())
        self.data['Rank'] = self.data['Rank'].fillna(
            self.data['Rank'].median())
        self.data['Grade'] = self.data['Grade'].fillna('Unknown')
        print(self.data, "\n")

    def remove_outliers(self):
        q1 = self.data['Marks'].quantile(0.25)
        q3 = self.data['Marks'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.data = self.data[(self.data['Marks'] >= lower_bound) & (
            self.data['Marks'] <= upper_bound)]
        print("After removing outliers:")
        print(self.data, "\n")

    def encode_categoricals(self):
        self.data['Grade'] = self.data['Grade'].map(
            {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'Unknown': 4})
        print("After encoding categoricals:")
