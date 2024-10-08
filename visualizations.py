# visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, style: str = 'darkgrid', context: str = 'notebook', font_scale: float = 1.0):
        self.style = style
        self.context = context
        self.font_scale = font_scale
        sns.set_style(self.style)
        sns.set_context(self.context, font_scale=self.font_scale)
    
    def set_global_style(self, style: str = 'darkgrid', context: str = 'notebook', font_scale: float = 1.0):
        """
        Establece el estilo global para las gráficas.

        Parámetros:
            style (str): Estilo de Seaborn (por defecto 'darkgrid').
            context (str): Contexto de Seaborn ('paper', 'notebook', 'talk', 'poster').
            font_scale (float): Escala de fuente (por defecto 1.0).
        """
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)

    def plot_histogram(self, data: pd.DataFrame, column: str, bins: int = 10, kde: bool = False,
                       title: str = None, xlabel: str = None, ylabel: str = 'Frecuencia',
                       figsize: tuple = (10, 6)):
        """
        Genera un histograma de la columna especificada.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            column (str): Nombre de la columna a graficar.
            bins (int): Número de bins (por defecto 10).
            kde (bool): Si se incluye una curva KDE (por defecto False).
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            figsize (tuple): Tamaño de la figura (por defecto (10, 6)).
        """
        if column not in data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
        plt.figure(figsize=figsize)
        sns.histplot(data[column].dropna(), bins=bins, kde=kde)
        plt.title(title if title else f'Histograma de {column}')
        plt.xlabel(xlabel if xlabel else column)
        plt.ylabel(ylabel)
        plt.show()

    def plot_boxplot(self, data: pd.DataFrame, column: str, title: str = None,
                     ylabel: str = None, figsize: tuple = (8, 6)):
        """
        Genera un boxplot de la columna especificada.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            column (str): Nombre de la columna a graficar.
            title (str): Título del gráfico.
            ylabel (str): Etiqueta del eje y.
            figsize (tuple): Tamaño de la figura.
        """
        if column not in data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
        plt.figure(figsize=figsize)
        sns.boxplot(y=data[column].dropna())
        plt.title(title if title else f'Boxplot de {column}')
        plt.ylabel(ylabel if ylabel else column)
        plt.show()

    def plot_scatter(self, data: pd.DataFrame, x: str, y: str, hue: str = None,
                     title: str = None, xlabel: str = None, ylabel: str = None,
                     figsize: tuple = (10, 6)):
        """
        Genera un gráfico de dispersión entre dos variables.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            x (str): Nombre de la columna en el eje x.
            y (str): Nombre de la columna en el eje y.
            hue (str): Columna para distinguir por colores.
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            figsize (tuple): Tamaño de la figura.
        """
        for col in [x, y, hue]:
            if col and col not in data.columns:
                raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
        data_clean = data[[col for col in [x, y, hue] if col]].dropna()
        plt.figure(figsize=figsize)
        sns.scatterplot(data=data_clean, x=x, y=y, hue=hue)
        plt.title(title if title else f'{y} vs {x}')
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        plt.show()

    def plot_line(self, data: pd.DataFrame, x: str, y: str, title: str = None,
                  xlabel: str = None, ylabel: str = None, figsize: tuple = (12, 6)):
        """
        Genera un gráfico de líneas.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            x (str): Nombre de la columna en el eje x.
            y (str): Nombre de la columna en el eje y.
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            figsize (tuple): Tamaño de la figura.
        """
        for col in [x, y]:
            if col not in data.columns:
                raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
        data_clean = data[[x, y]].dropna()
        plt.figure(figsize=figsize)
        sns.lineplot(data=data_clean, x=x, y=y)
        plt.title(title if title else f'Gráfico de Líneas de {y} vs {x}')
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        plt.show()

    def plot_correlation_heatmap(self, data: pd.DataFrame, method: str = 'pearson',
                                 title: str = 'Matriz de Correlación', figsize: tuple = (12, 10)):
        """
        Genera un heatmap de la matriz de correlación.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            method (str): Método de correlación ('pearson', 'spearman', 'kendall').
            title (str): Título del gráfico.
            figsize (tuple): Tamaño de la figura.
        """
        corr_matrix = data.corr(method=method)
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(title)
        plt.show()

    def plot_pairplot(self, data: pd.DataFrame, vars: list = None, hue: str = None):
        """
        Genera un pairplot para visualizar relaciones entre variables.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            vars (list): Lista de variables a incluir.
            hue (str): Columna para distinguir por colores.
        """
        if hue and hue not in data.columns:
            raise ValueError(f"La columna '{hue}' no existe en el DataFrame.")
        if vars:
            for var in vars:
                if var not in data.columns:
                    raise ValueError(f"La columna '{var}' no existe en el DataFrame.")
            data = data[vars + ([hue] if hue else [])].dropna()
        else:
            data = data.dropna()
        sns.pairplot(data, hue=hue)
        plt.show()
    
    def plot_time_series(self, data: pd.DataFrame, date_col: str, value_col: str,
                         title: str = None, xlabel: str = None, ylabel: str = None,
                         figsize: tuple = (12, 6)):
        """
        Genera un gráfico de serie temporal.

        Parámetros:
            data (pd.DataFrame): DataFrame que contiene los datos.
            date_col (str): Nombre de la columna de fechas.
            value_col (str): Nombre de la columna de valores.
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            figsize (tuple): Tamaño de la figura.
        """
        if date_col not in data.columns or value_col not in data.columns:
            raise ValueError("Las columnas especificadas no existen en el DataFrame.")
        data_clean = data[[date_col, value_col]].dropna()
        # Convertir la columna de fechas a tipo datetime si no lo es
        if not np.issubdtype(data_clean[date_col].dtype, np.datetime64):
            data_clean[date_col] = pd.to_datetime(data_clean[date_col])
        data_clean.sort_values(by=date_col, inplace=True)
        plt.figure(figsize=figsize)
        sns.lineplot(data=data_clean, x=date_col, y=value_col)
        plt.title(title if title else f'Serie Temporal de {value_col}')
        plt.xlabel(xlabel if xlabel else date_col)
        plt.ylabel(ylabel if ylabel else value_col)
        plt.show()
