import pandas as pd

class OutliersHandler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect_outliers(self, column, method="iqr"):
        """Detecta outliers en una columna usando el método IQR."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

    def handle_outliers(self, strategy='cap'):
        """Trata los outliers en todas las columnas numéricas según la estrategia."""
        for column in self.df.select_dtypes(include='number').columns:
            outliers = self.detect_outliers(column)
            if strategy == 'cap':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df[column] = self.df[column].clip(lower_bound, upper_bound)
            elif strategy == 'remove':
                self.df = self.df.drop(outliers.index)
        return self.df
