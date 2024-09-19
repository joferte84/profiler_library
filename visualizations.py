# visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_global_style(style: str = 'darkgrid', context: str = 'notebook', font_scale: float = 1.0):
    """
    Establece el estilo global para las gráficas.

    Parámetros:
        style (str): Estilo de Seaborn (por defecto 'darkgrid').
        context (str): Contexto de Seaborn ('paper', 'notebook', 'talk', 'poster').
        font_scale (float): Escala de fuente (por defecto 1.0).
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)

def plot_histogram(data: pd.DataFrame, column: str, bins: int = 10, kde: bool = False,
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
    plt.figure(figsize=figsize)
    sns.histplot(data[column].dropna(), bins=bins, kde=kde)
    plt.title(title if title else f'Histograma de {column}')
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.show()

def plot_boxplot(data: pd.DataFrame, column: str, title: str = None,
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
    plt.figure(figsize=figsize)
    sns.boxplot(y=data[column].dropna())
    plt.title(title if title else f'Boxplot de {column}')
    plt.ylabel(ylabel if ylabel else column)
    plt.show()

def plot_scatter(data: pd.DataFrame, x: str, y: str, hue: str = None,
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
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.title(title if title else f'{y} vs {x}')
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.show()

def plot_line(data: pd.DataFrame, x: str, y: str, title: str = None,
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
    plt.figure(figsize=figsize)
    sns.lineplot(data=data, x=x, y=y)
    plt.title(title if title else f'Gráfico de Líneas de {y} vs {x}')
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.show()

def plot_correlation_heatmap(data: pd.DataFrame, title: str = 'Matriz de Correlación',
                             figsize: tuple = (12, 10)):
    """
    Genera un heatmap de la matriz de correlación.

    Parámetros:
        data (pd.DataFrame): DataFrame que contiene los datos.
        title (str): Título del gráfico.
        figsize (tuple): Tamaño de la figura.
    """
    corr_matrix = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()
