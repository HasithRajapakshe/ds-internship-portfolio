class DataCleaner:
    def __init__(self, data):
        self.data = data

    def detect_missing(self):
        print("Missing Values in Each Column:")
        print(self.data.isnull().sum(), "\n")

    def fill_missing(self):
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col].fillna('Unknown', inplace=True)
            else:
                self.data[col].fillna(
                    self.data[col].median(), inplace=True)
        print("After filling missing values:")
        print(self.data, "\n")

    def remove_outliers(self):
        for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.data = self.data[(self.data[col] >= lower_bound) & (
                self.data[col] <= upper_bound)]

        print("After removing outliers:")
        print(self.data, "\n")

    def encode_categoricals(self):
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category').cat.codes
        print("After encoding categorical variables:")
