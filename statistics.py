import pandas as pd

class Statistics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_basic_statistics(self):
        """Devuelve estadísticas básicas (media, mediana, etc.) de todas las columnas numéricas."""
        return self.df.describe()

    def get_correlation_matrix(self, method='pearson'):
        """Devuelve la matriz de correlación de las columnas numéricas."""
        return self.df.corr(method=method)
