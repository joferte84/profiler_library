import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizations:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_histogram(self, column):
        """Genera un histograma de una columna."""
        plt.figure()
        self.df[column].hist()
        plt.title(f'Histograma de {column}')
        plt.show()

    def generate_boxplot(self, column):
        """Genera un boxplot de una columna."""
        plt.figure()
        sns.boxplot(x=self.df[column])
        plt.title(f'Boxplot de {column}')
        plt.show()

    def generate_all_plots(self):
        """Genera todos los gráficos relevantes (histogramas y boxplots) para todas las columnas numéricas."""
        plots = []
        for column in self.df.select_dtypes(include='number').columns:
            self.generate_histogram(column)
            self.generate_boxplot(column)
            plots.append(f'Generado gráfico de {column}')
        return plots
