import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NanHandler:
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger]=None):
        self.original_df = df  # Guardamos el DataFrame original
        self.df = df.copy()    # Trabajamos sobre una copia
        self.changes_log = []  # Historial de cambios

        # Configuración del logger específico de la clase
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler_exists = any([isinstance(h, logging.FileHandler) for h in self.logger.handlers])
        if not handler_exists:
            handler = logging.FileHandler('nan_handler.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    # Métodos auxiliares
    def log_action(self, message):
        """Registra acciones en el historial y en el log."""
        self.changes_log.append(message)
        self.logger.info(message)
    
    def generate_changes_report(self):
        """Genera un reporte de los cambios realizados."""
        report = "\n".join(self.changes_log)
        return report
    
    def revert_changes(self):
        """Revierte los cambios y restaura el DataFrame original."""
        self.df = self.original_df.copy()
        self.changes_log.append("Se han revertido los cambios y se ha restaurado el DataFrame original.")
        self.logger.info("Se han revertido los cambios y se ha restaurado el DataFrame original.")
    
    # Métodos de preprocesamiento
    def handle_infinite_values(self):
        """Reemplaza valores infinitos por NaN."""
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.log_action("Valores infinitos reemplazados por NaN.")
    
    def detect_nans(self):
        """Detecta y devuelve el porcentaje de NaNs por columna."""
        nan_percentage = self.df.isna().mean() * 100
        return nan_percentage
    
    def get_nan_summary(self):
        """Devuelve un resumen de NaNs por columna."""
        nan_counts = self.df.isna().sum()
        nan_percentage = self.df.isna().mean() * 100
        summary = pd.DataFrame({'num_nans': nan_counts, 'perc_nans': nan_percentage})
        return summary
    
    # Métodos de visualización
    def visualize_nans_bars(self, before=True, show_plot=False, save_plot=False, plot_filename='nan_bars.png'):
        """
        Genera un gráfico de barras que muestra el porcentaje de NaNs por columna.
        """
        if before:
            df_to_use = self.original_df
            title = 'Porcentaje de NaNs por columna (Antes del tratamiento)'
        else:
            df_to_use = self.df
            title = 'Porcentaje de NaNs por columna (Después del tratamiento)'

        nan_percentage = df_to_use.isna().mean() * 100
        nan_percentage = nan_percentage[nan_percentage > 0]  # Solo columnas con NaNs

        if nan_percentage.empty:
            print("No hay NaNs en el DataFrame seleccionado.")
            return

        nan_percentage.sort_values(ascending=False, inplace=True)

        plt.figure(figsize=(10, 6))
        nan_percentage.plot(kind='bar')
        plt.ylabel('Porcentaje de NaNs')
        plt.title(title)
        plt.tight_layout()

        if save_plot:
            plt.savefig(plot_filename)
            self.log_action(f"Gráfico de NaNs guardado como '{plot_filename}'.")

        if show_plot:
            plt.show()

        plt.close()
    
    # Métodos principales
    def advanced_imputation(self, method='knn', columns=None, **kwargs):
        """Realiza imputación avanzada usando el método especificado."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if columns is not None:
            columns = [col for col in columns if col in numeric_cols]
        else:
            columns = numeric_cols

        if columns.any():
            if method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(**kwargs)
                imputed_values = imputer.fit_transform(self.df[columns])
                imputed_df = pd.DataFrame(imputed_values, columns=columns, index=self.df.index)
                self.df[columns] = imputed_df
                self.log_action(f"Imputación KNN realizada en columnas: {columns}.")
            elif method == 'mice':
                from sklearn.experimental import enable_iterative_imputer  # Necesario para activar IterativeImputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(**kwargs)
                imputed_values = imputer.fit_transform(self.df[columns])
                imputed_df = pd.DataFrame(imputed_values, columns=columns, index=self.df.index)
                self.df[columns] = imputed_df
                self.log_action(f"Imputación MICE realizada en columnas: {columns}.")
            else:
                raise ValueError(f"Método de imputación '{method}' no reconocido.")
        else:
            self.log_action("No hay columnas numéricas para imputar con el método avanzado.")

    def handle_nans(self, drop_threshold=10, multi_column_drop_threshold=50, decisions=None, default_action='omit', default_custom_values=None):
        """
        Maneja los NaNs en el DataFrame.
        """
        nan_percentage = self.detect_nans()
        decisions = decisions or {}
        default_custom_values = default_custom_values or {}

        columns_needing_decisions = []

        for column in self.df.columns:
            perc_nan = nan_percentage[column]
            self.log_action(f"Columna '{column}': {perc_nan:.2f}% de NaNs")

            if perc_nan == 0:
                continue  # No se necesita acción

            action = decisions.get(column, default_action)

            if perc_nan >= drop_threshold and action != 'keep':
                if action == 'drop_column':
                    self.df.drop(columns=[column], inplace=True)
                    self.log_action(f"Columna '{column}': Eliminada por tener más del {drop_threshold}% de NaNs.")
                    continue  # Pasar a la siguiente columna
                elif action == 'keep':
                    self.log_action(f"Columna '{column}': Manteniendo la columna a pesar de tener {perc_nan:.2f}% de NaNs.")
                else:
                    self.log_action(f"Columna '{column}': Acción '{action}' no válida para columnas con más del {drop_threshold}% de NaNs.")
                    columns_needing_decisions.append(column)
                    continue

            col_dtype = self.df[column].dtype

            if pd.api.types.is_numeric_dtype(col_dtype):
                # Columna numérica
                if action == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                    self.log_action(f"Columna '{column}': Imputada con la media.")
                elif action == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                    self.log_action(f"Columna '{column}': Imputada con la mediana.")
                elif action == 'custom':
                    custom_value = decisions.get(f"{column}_value", default_custom_values.get(column))
                    if custom_value is None:
                        columns_needing_decisions.append(column)
                        self.log_action(f"Columna '{column}': Valor personalizado faltante para imputación.")
                    else:
                        if np.issubdtype(type(custom_value), np.number):
                            self.df[column].fillna(custom_value, inplace=True)
                            self.log_action(f"Columna '{column}': Imputada con valor personalizado {custom_value}.")
                        else:
                            self.log_action(f"Columna '{column}': Valor personalizado '{custom_value}' no es numérico.")
                            columns_needing_decisions.append(column)
                elif action == 'drop_rows':
                    self.df.dropna(subset=[column], inplace=True)
                    self.log_action(f"Columna '{column}': Filas con NaNs eliminadas.")
                elif action == 'omit':
                    self.log_action(f"Columna '{column}': Tratamiento omitido, la columna se mantiene sin cambios.")
                else:
                    self.log_action(f"Columna '{column}': Acción desconocida '{action}'.")
                    columns_needing_decisions.append(column)
            elif pd.api.types.is_categorical_dtype(col_dtype) or col_dtype == object:
                # Columna categórica
                if action == 'mode':
                    mode_values = self.df[column].mode()
                    if len(mode_values) > 0:
                        selected_mode = mode_values[0]  # Usar la primera moda
                        self.df[column].fillna(selected_mode, inplace=True)
                        self.log_action(f"Columna '{column}': Imputada con la moda '{selected_mode}'.")
                    else:
                        default_value = default_custom_values.get(column)
                        if default_value is not None:
                            self.df[column].fillna(default_value, inplace=True)
                            self.log_action(f"Columna '{column}': Imputada con valor predeterminado '{default_value}'.")
                        else:
                            self.log_action(f"Columna '{column}': No se pudo calcular la moda y no hay valor predeterminado.")
                            columns_needing_decisions.append(column)
                elif action == 'custom':
                    custom_value = decisions.get(f"{column}_value", default_custom_values.get(column))
                    if custom_value is None:
                        columns_needing_decisions.append(column)
                        self.log_action(f"Columna '{column}': Valor personalizado faltante para imputación.")
                    else:
                        self.df[column].fillna(custom_value, inplace=True)
                        self.log_action(f"Columna '{column}': Imputada con valor personalizado '{custom_value}'.")
                elif action == 'drop_rows':
                    self.df.dropna(subset=[column], inplace=True)
                    self.log_action(f"Columna '{column}': Filas con NaNs eliminadas.")
                elif action == 'omit':
                    self.log_action(f"Columna '{column}': Tratamiento omitido, la columna se mantiene sin cambios.")
                else:
                    self.log_action(f"Columna '{column}': Acción desconocida '{action}'.")
                    columns_needing_decisions.append(column)
            elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                # Columna datetime
                if action == 'median':
                    try:
                        median_date = pd.to_datetime(self.df[column]).dropna().astype(int).median()
                        median_date = pd.to_datetime(median_date)
                        self.df[column].fillna(median_date, inplace=True)
                        self.log_action(f"Columna '{column}': Imputada con la mediana de fecha '{median_date}'.")
                    except Exception as e:
                        self.log_action(f"Columna '{column}': Error al calcular la mediana de fecha: {e}")
                        columns_needing_decisions.append(column)
                elif action == 'custom':
                    custom_value = decisions.get(f"{column}_value", default_custom_values.get(column))
                    if custom_value is None:
                        columns_needing_decisions.append(column)
                        self.log_action(f"Columna '{column}': Valor personalizado faltante para imputación.")
                    else:
                        try:
                            custom_value = pd.to_datetime(custom_value)
                            self.df[column].fillna(custom_value, inplace=True)
                            self.log_action(f"Columna '{column}': Imputada con valor personalizado '{custom_value}'.")
                        except (ValueError, TypeError):
                            self.log_action(f"Columna '{column}': Valor personalizado '{custom_value}' no es una fecha válida.")
                            columns_needing_decisions.append(column)
                elif action == 'drop_rows':
                    self.df.dropna(subset=[column], inplace=True)
                    self.log_action(f"Columna '{column}': Filas con NaNs eliminadas.")
                elif action == 'omit':
                    self.log_action(f"Columna '{column}': Tratamiento omitido, la columna se mantiene sin cambios.")
                else:
                    self.log_action(f"Columna '{column}': Acción desconocida '{action}' para columnas datetime.")
                    columns_needing_decisions.append(column)
            else:
                # Tipo de dato no manejado
                action = decisions.get(column, default_action)
                if action == 'drop_column':
                    self.df.drop(columns=[column], inplace=True)
                    self.log_action(f"Columna '{column}': Eliminada por tener tipo de dato no soportado.")
                elif action == 'omit':
                    self.log_action(f"Columna '{column}': Tratamiento omitido para tipo de dato '{col_dtype}'.")
                else:
                    self.log_action(f"Columna '{column}': Tipo de dato '{col_dtype}' no soportado y acción '{action}' desconocida.")
                    columns_needing_decisions.append(column)

        # Eliminar filas si tienen NaNs en más de X% de columnas
        drop_rows_action = decisions.get('drop_rows_with_nans', default_action)
        if drop_rows_action == 'yes':
            nan_per_row = self.df.isna().mean(axis=1) * 100
            self.df = self.df[nan_per_row < multi_column_drop_threshold]
            self.log_action(f"Filas con más del {multi_column_drop_threshold}% de NaNs eliminadas.")
        elif drop_rows_action == 'no':
            self.log_action("Filas mantenidas sin cambios.")
        elif drop_rows_action == 'omit':
            self.log_action("No se ha aplicado ninguna acción sobre las filas con NaNs.")
        else:
            self.log_action(f"Acción desconocida para 'drop_rows_with_nans': '{drop_rows_action}'")
            columns_needing_decisions.append('drop_rows_with_nans')

        if columns_needing_decisions:
            missing_decisions = ", ".join(columns_needing_decisions)
            error_message = f"Decisiones faltantes o inválidas para las siguientes columnas o acciones: {missing_decisions}"
            self.log_action(error_message)
            print(error_message)
            # Devolver también la lista de columnas no procesadas
            return self.df, columns_needing_decisions
        else:
            return self.df, []

