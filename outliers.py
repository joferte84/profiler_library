import logging
from typing import Optional, Dict, Any, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierHandler:
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger]=None):
        """
        Inicializa una instancia de OutlierHandler.

        Args:
            df (pd.DataFrame): DataFrame a procesar.
            logger (Optional[logging.Logger]): Logger personalizado.
        """
        self.original_df = df  # DataFrame original
        self.df = df.copy()    # Copia para trabajar
        self.changes_log = []  # Historial de cambios
        self.outlier_flags = pd.DataFrame(index=self.df.index)  # DataFrame para marcar outliers

        # Configuración del logger específico de la clase
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler_exists = any([isinstance(h, logging.FileHandler) for h in self.logger.handlers])
        if not handler_exists:
            handler = logging.FileHandler('outlier_handler.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Diccionario para almacenar rangos válidos definidos por el usuario
        self.valid_ranges = {}
        
    # Métodos auxiliares
    def log_action(self, message):
        """Registra acciones en el historial y en el log."""
        self.changes_log.append(message)
        self.logger.info(message)
        
    def generate_outlier_report(self):
        """Genera un reporte de los outliers detectados y las acciones tomadas."""
        report = "\n".join(self.changes_log)
        return report

    def revert_changes(self):
        """Revierte los cambios y restaura el DataFrame original."""
        self.df = self.original_df.copy()
        self.outlier_flags = pd.DataFrame(index=self.df.index)
        self.changes_log.append("Se han revertido los cambios y se ha restaurado el DataFrame original.")
        self.logger.info("Se han revertido los cambios y se ha restaurado el DataFrame original.")
        
    def set_valid_ranges(self, ranges: Dict[str, tuple]):
        """
        Establece rangos válidos para variables específicas.

        Args:
            ranges (Dict[str, tuple]): Diccionario con tuplas (min, max) por variable.
        """
        self.valid_ranges = ranges
        self.log_action(f"Rangos válidos establecidos: {self.valid_ranges}")
        
    # Métodos de detección de outliers
    def detect_outliers(self, method='iqr', columns: Optional[List[str]] = None, **kwargs):
        """
        Detecta outliers utilizando el método especificado.

        Args:
            method (str): Método de detección ('iqr', 'z_score', 'custom').
            columns (List[str], opcional): Lista de columnas a analizar. Si es None, se analizan todas las columnas numéricas.
            **kwargs: Parámetros adicionales para el método de detección.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in self.df.columns:
                self.log_action(f"Columna '{column}' no encontrada en el DataFrame.")
                continue
            
            col_data = self.df[column]
            if pd.api.types.is_numeric_dtype(col_data):
                if method == 'iqr':
                    self._detect_outliers_iqr(column, **kwargs)
                elif method == 'z_score':
                    self._detect_outliers_z_score(column, **kwargs)
                elif method == 'custom':
                    func = kwargs.get('func')
                    if func and callable(func):
                        self._detect_outliers_custom(column, func)
                    else:
                        self.log_action(f"Función personalizada no proporcionada o no es callable para la columna '{column}'.")
                else:
                    self.log_action(f"Método de detección '{method}' no reconocido para la columna '{column}'.")
            else:
                self.log_action(f"Columna '{column}' no es numérica y será omitida en la detección de outliers.")
        
    def _detect_outliers_iqr(self, column, factor=1.5):
        """Detecta outliers utilizando el método IQR."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Aplicar rangos válidos si están definidos
        min_valid, max_valid = self.valid_ranges.get(column, (None, None))
        if min_valid is not None:
            lower_bound = max(lower_bound, min_valid)
        if max_valid is not None:
            upper_bound = min(upper_bound, max_valid)
        
        outlier_condition = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        self.outlier_flags[column] = outlier_condition
        self.log_action(f"Outliers detectados en columna '{column}' utilizando IQR con factor {factor}.")
        
    def _detect_outliers_z_score(self, column, threshold=3):
        """Detecta outliers utilizando el método de Z-score."""
        mean = self.df[column].mean()
        std = self.df[column].std()
        z_scores = (self.df[column] - mean) / std
        
        outlier_condition = z_scores.abs() > threshold
        
        # Aplicar rangos válidos si están definidos
        min_valid, max_valid = self.valid_ranges.get(column, (None, None))
        if min_valid is not None:
            outlier_condition &= self.df[column] >= min_valid
        if max_valid is not None:
            outlier_condition &= self.df[column] <= max_valid
        
        self.outlier_flags[column] = outlier_condition
        self.log_action(f"Outliers detectados en columna '{column}' utilizando Z-score con umbral {threshold}.")
        
    def _detect_outliers_custom(self, column, func: Callable):
        """Detecta outliers utilizando una función personalizada."""
        outlier_condition = func(self.df[column])
        if not isinstance(outlier_condition, pd.Series):
            self.log_action(f"La función personalizada debe devolver un pd.Series booleano para la columna '{column}'.")
            return
        self.outlier_flags[column] = outlier_condition
        self.log_action(f"Outliers detectados en columna '{column}' utilizando función personalizada.")
        
    # Métodos de tratamiento de outliers
    def handle_outliers(self, actions: Optional[Dict[str, str]] = None, default_action='keep', cap_values: Optional[Dict[str, tuple]] = None):
        """
        Aplica acciones de tratamiento a los outliers detectados.

        Args:
            actions (Dict[str, str], opcional): Diccionario con acciones específicas por variable.
            default_action (str): Acción predeterminada para variables no especificadas.
            cap_values (Dict[str, tuple], opcional): Valores mínimos y máximos para la acción 'cap'.
        """
        actions = actions or {}
        cap_values = cap_values or {}
        
        for column in self.outlier_flags.columns:
            outlier_indices = self.outlier_flags[self.outlier_flags[column]].index
            action = actions.get(column, default_action)
            
            if action == 'remove':
                self.df = self.df.drop(index=outlier_indices)
                self.log_action(f"Outliers eliminados en columna '{column}'.")
            elif action == 'cap':
                lower_cap, upper_cap = cap_values.get(column, (None, None))
                if lower_cap is not None:
                    self.df.loc[outlier_indices & (self.df[column] < lower_cap), column] = lower_cap
                if upper_cap is not None:
                    self.df.loc[outlier_indices & (self.df[column] > upper_cap), column] = upper_cap
                self.log_action(f"Outliers capados en columna '{column}' con valores ({lower_cap}, {upper_cap}).")
            elif action == 'impute':
                median_value = self.df[column].median()
                self.df.loc[outlier_indices, column] = median_value
                self.log_action(f"Outliers imputados en columna '{column}' con la mediana '{median_value}'.")
            elif action == 'transform':
                self.df[column] = np.log1p(self.df[column])
                self.log_action(f"Transformación logarítmica aplicada en columna '{column}'.")
            elif action == 'mark':
                self.df[f'{column}_is_outlier'] = self.outlier_flags[column]
                self.log_action(f"Outliers marcados en columna '{column}' con etiqueta '{column}_is_outlier'.")
            elif action == 'keep':
                self.log_action(f"Outliers mantenidos sin cambios en columna '{column}'.")
            else:
                self.log_action(f"Acción '{action}' no reconocida para columna '{column}'.")
        
        # Actualizar outlier_flags eliminando filas que ya no existen
        self.outlier_flags = self.outlier_flags.loc[self.df.index]
    
    # Métodos de visualización
    def visualize_outliers(self, columns: Optional[List[str]] = None, before=True, show_plot=False, save_plot=False, plot_filename='outliers_plot.png'):
        """
        Genera gráficos para visualizar outliers.

        Args:
            columns (List[str], opcional): Lista de columnas a visualizar. Si es None, se visualizan todas las columnas numéricas.
            before (bool): Si True, muestra los datos antes del tratamiento; si False, después.
            show_plot (bool): Si True, muestra el gráfico.
            save_plot (bool): Si True, guarda el gráfico en un archivo.
            plot_filename (str): Nombre del archivo para guardar el gráfico.
        """
        df_to_use = self.original_df if before else self.df
        title_suffix = ' (Antes del tratamiento)' if before else ' (Después del tratamiento)'
        
        if columns is None:
            columns = df_to_use.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df_to_use.columns:
                self.log_action(f"Columna '{column}' no encontrada en el DataFrame.")
                continue
            
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df_to_use[column])
            plt.title(f'Distribución de {column}{title_suffix}')
            plt.tight_layout()
            
            if save_plot:
                filename = f"{column}_{'before' if before else 'after'}_{plot_filename}"
                plt.savefig(filename)
                self.log_action(f"Gráfico de outliers guardado como '{filename}'.")
            
            if show_plot:
                plt.show()
            
            plt.close()
