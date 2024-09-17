import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List

class DataProfiler:
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger]=None):
        """
        Inicializa una instancia de DataProfiler.

        Args:
            df (pd.DataFrame): DataFrame a analizar.
            logger (Optional[logging.Logger]): Logger personalizado.
        """
        self.df = df.copy()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler('data_profiler.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        self.changes_log = []

    def log_action(self, message):
        """Registra acciones en el historial y en el log."""
        self.changes_log.append(message)
        self.logger.info(message)

    def get_numeric_summary(self):
        """Devuelve un resumen estadístico de las variables numéricas."""
        if not self.numeric_cols:
            self.log_action("No hay variables numéricas para resumir.")
            return pd.DataFrame()
        summary = self.df[self.numeric_cols].describe().transpose()
        self.log_action("Resumen estadístico de variables numéricas generado.")
        return summary

    def get_categorical_summary(self):
        """Devuelve un resumen de las variables categóricas."""
        summaries = {}
        if not self.categorical_cols:
            self.log_action("No hay variables categóricas para resumir.")
            return summaries
        for col in self.categorical_cols:
            counts = self.df[col].value_counts(dropna=False)
            summaries[col] = counts
            self.log_action(f"Resumen de la variable categórica '{col}' generado.")
        return summaries

    def detect_missing_values(self):
        """Devuelve un resumen de los valores faltantes por variable."""
        missing_counts = self.df.isna().sum()
        missing_percentage = self.df.isna().mean() * 100
        summary = pd.DataFrame({'missing_count': missing_counts, 'missing_percentage': missing_percentage})
        self.log_action("Resumen de valores faltantes generado.")
        return summary

    def detect_outliers(self, method='iqr', factor=1.5):
        """
        Detecta outliers en variables numéricas y devuelve un DataFrame con indicadores.

        Args:
            method (str): Método de detección ('iqr').
            factor (float): Factor para el método IQR.
        """
        outlier_flags = pd.DataFrame(index=self.df.index)
        if not self.numeric_cols:
            self.log_action("No hay variables numéricas para detectar outliers.")
            return outlier_flags
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                outlier_condition = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_flags[col] = outlier_condition
                self.log_action(f"Outliers detectados en columna '{col}' utilizando IQR.")
            else:
                self.log_action(f"Método '{method}' no implementado para detección de outliers.")
        return outlier_flags

    def get_correlations(self):
        """Devuelve la matriz de correlación de las variables numéricas."""
        if len(self.numeric_cols) < 2:
            self.log_action("No hay suficientes variables numéricas para calcular correlaciones.")
            return pd.DataFrame()
        corr_matrix = self.df[self.numeric_cols].corr()
        self.log_action("Matriz de correlación generada.")
        return corr_matrix

    def plot_histograms(self, columns: Optional[List[str]] = None, bins=10, show_plot=False, save_plot=False):
        """
        Genera histogramas para las variables numéricas.

        Args:
            columns (List[str], opcional): Lista de columnas a visualizar.
            bins (int): Número de bins para el histograma.
            show_plot (bool): Si True, muestra el gráfico.
            save_plot (bool): Si True, guarda el gráfico en un archivo.
        """
        columns = columns or self.numeric_cols
        for col in columns:
            if col not in self.df.columns:
                self.log_action(f"Columna '{col}' no encontrada en el DataFrame.")
                continue
            plt.figure(figsize=(8, 6))
            self.df[col].dropna().hist(bins=bins)
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.title(f'Histograma de {col}')
            plt.tight_layout()
            if save_plot:
                filename = f"{col}_histogram.png"
                plt.savefig(filename)
                self.log_action(f"Histograma de '{col}' guardado como '{filename}'.")
            if show_plot:
                plt.show()
            plt.close()

    def plot_boxplots(self, columns: Optional[List[str]] = None, show_plot=False, save_plot=False):
        """
        Genera boxplots para las variables numéricas.

        Args:
            columns (List[str], opcional): Lista de columnas a visualizar.
            show_plot (bool): Si True, muestra el gráfico.
            save_plot (bool): Si True, guarda el gráfico en un archivo.
        """
        columns = columns or self.numeric_cols
        for col in columns:
            if col not in self.df.columns:
                self.log_action(f"Columna '{col}' no encontrada en el DataFrame.")
                continue
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=self.df[col].dropna())
            plt.title(f'Boxplot de {col}')
            plt.tight_layout()
            if save_plot:
                filename = f"{col}_boxplot.png"
                plt.savefig(filename)
                self.log_action(f"Boxplot de '{col}' guardado como '{filename}'.")
            if show_plot:
                plt.show()
            plt.close()

    def plot_bar_charts(self, columns: Optional[List[str]] = None, show_plot=False, save_plot=False):
        """
        Genera gráficos de barras para las variables categóricas.

        Args:
            columns (List[str], opcional): Lista de columnas a visualizar.
            show_plot (bool): Si True, muestra el gráfico.
            save_plot (bool): Si True, guarda el gráfico en un archivo.
        """
        columns = columns or self.categorical_cols
        for col in columns:
            if col not in self.df.columns:
                self.log_action(f"Columna '{col}' no encontrada en el DataFrame.")
                continue
            plt.figure(figsize=(10, 6))
            counts = self.df[col].value_counts(dropna=False).head(20)
            counts.plot(kind='bar')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.title(f'Gráfico de barras de {col}')
            plt.tight_layout()
            if save_plot:
                filename = f"{col}_bar_chart.png"
                plt.savefig(filename)
                self.log_action(f"Gráfico de barras de '{col}' guardado como '{filename}'.")
            if show_plot:
                plt.show()
            plt.close()

    def plot_scatter_matrix(self, columns: Optional[List[str]] = None, show_plot=False, save_plot=False):
        """
        Genera una matriz de dispersión de las variables numéricas.

        Args:
            columns (List[str], opcional): Lista de columnas a incluir.
            show_plot (bool): Si True, muestra el gráfico.
            save_plot (bool): Si True, guarda el gráfico en un archivo.
        """
        columns = columns or self.numeric_cols
        if len(columns) < 2:
            self.log_action("No hay suficientes variables numéricas para generar una matriz de dispersión.")
            return
        sns.pairplot(self.df[columns].dropna())
        plt.tight_layout()
        if save_plot:
            filename = "scatter_matrix.png"
            plt.savefig(filename)
            self.log_action(f"Matriz de dispersión guardada como '{filename}'.")
        if show_plot:
            plt.show()
        plt.close()

    def generate_report(self):
        """
        Genera un reporte con todos los análisis.

        Returns:
            str: Reporte generado.
        """
        report = "=== Resumen Estadístico de Variables Numéricas ===\n"
        numeric_summary = self.get_numeric_summary()
        if not numeric_summary.empty:
            report += str(numeric_summary)
        else:
            report += "No hay variables numéricas para resumir."
        report += "\n\n=== Resumen de Variables Categóricas ===\n"
        categorical_summary = self.get_categorical_summary()
        if categorical_summary:
            for col, counts in categorical_summary.items():
                report += f"\n-- {col} --\n{counts}\n"
        else:
            report += "No hay variables categóricas para resumir."
        report += "\n\n=== Valores Faltantes ===\n"
        missing_values = self.detect_missing_values()
        report += str(missing_values)
        report += "\n\n=== Matriz de Correlación ===\n"
        correlations = self.get_correlations()
        if not correlations.empty:
            report += str(correlations)
        else:
            report += "No hay suficientes variables numéricas para calcular correlaciones."
        self.log_action("Reporte generado.")
        return report
