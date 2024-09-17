import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List
import base64
from io import BytesIO
import os

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

    # Métodos para generar visualizaciones y convertirlas en imágenes embebidas en base64
    def _generate_histogram(self, col):
        plt.figure(figsize=(6, 4))
        self.df[col].dropna().hist(bins=30)
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.title(f'Histograma de {col}')
        plt.tight_layout()
        return self._fig_to_base64()

    def _generate_boxplot(self, col):
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=self.df[col].dropna())
        plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        return self._fig_to_base64()

    def _generate_bar_chart(self, col):
        plt.figure(figsize=(6, 4))
        counts = self.df[col].value_counts(dropna=False).head(20)
        counts.plot(kind='bar')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.title(f'Gráfico de barras de {col}')
        plt.tight_layout()
        return self._fig_to_base64()

    def _generate_missing_data_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False)
        plt.title('Mapa de calor de valores faltantes')
        plt.xlabel('Columnas')
        plt.ylabel('Filas')
        plt.tight_layout()
        return self._fig_to_base64()

    def _generate_correlation_heatmap(self):
        plt.figure(figsize=(8, 6))
        corr = self.df[self.numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        return self._fig_to_base64()

    def _fig_to_base64(self):
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    def generate_report(self, report_filename='data_profile_report.html'):
        """
        Genera un reporte detallado en formato HTML.

        Args:
            report_filename (str): Nombre del archivo HTML de salida.

        Returns:
            None
        """
        # Información general
        num_rows, num_cols = self.df.shape
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024 ** 2  # En MB
        info = {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'memory_usage': f"{memory_usage:.2f} MB",
            'numeric_cols': len(self.numeric_cols),
            'categorical_cols': len(self.categorical_cols),
            'datetime_cols': len(self.datetime_cols),
        }

        # Distribución de tipos de datos
        dtypes_counts = self.df.dtypes.value_counts().to_dict()

        # Estadísticas descriptivas
        numeric_summary = self.df[self.numeric_cols].describe().transpose()
        categorical_summary = {}
        for col in self.categorical_cols:
            counts = self.df[col].value_counts(dropna=False)
            categorical_summary[col] = counts

        # Visualizaciones
        numeric_histograms = {}
        numeric_boxplots = {}
        for col in self.numeric_cols:
            numeric_histograms[col] = self._generate_histogram(col)
            numeric_boxplots[col] = self._generate_boxplot(col)

        categorical_bar_charts = {}
        for col in self.categorical_cols:
            categorical_bar_charts[col] = self._generate_bar_chart(col)

        # Valores faltantes
        missing_values = self.df.isna().sum()
        missing_percentage = self.df.isna().mean() * 100
        missing_summary = pd.DataFrame({
            'Variable': self.df.columns,
            'Valores Faltantes': missing_values,
            'Porcentaje Faltante': missing_percentage
        })

        missing_heatmap = self._generate_missing_data_heatmap()

        # Correlaciones
        if len(self.numeric_cols) >= 2:
            correlation_matrix = self.df[self.numeric_cols].corr()
            correlation_heatmap = self._generate_correlation_heatmap()
        else:
            correlation_matrix = pd.DataFrame()
            correlation_heatmap = None

        # Construir el HTML
        html_content = self._build_html_report(
            info,
            dtypes_counts,
            numeric_summary,
            categorical_summary,
            numeric_histograms,
            numeric_boxplots,
            categorical_bar_charts,
            missing_summary,
            missing_heatmap,
            correlation_matrix,
            correlation_heatmap
        )

        # Guardar el reporte
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.log_action(f"Reporte generado y guardado como '{report_filename}'.")

    def _build_html_report(self, info, dtypes_counts, numeric_summary, categorical_summary,
                           numeric_histograms, numeric_boxplots, categorical_bar_charts,
                           missing_summary, missing_heatmap, correlation_matrix, correlation_heatmap):
        """
        Construye el contenido HTML del reporte.

        Args:
            (Todos los parámetros son los datos generados para el reporte)

        Returns:
            str: Contenido HTML del reporte.
        """
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Perfilamiento de Datos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
                .section {{ margin-bottom: 40px; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Perfilamiento de Datos</h1>

            <div class="section">
                <h2>Información General</h2>
                <ul>
                    <li><strong>Número de filas:</strong> {info['num_rows']}</li>
                    <li><strong>Número de columnas:</strong> {info['num_cols']}</li>
                    <li><strong>Uso de memoria:</strong> {info['memory_usage']}</li>
                    <li><strong>Variables numéricas:</strong> {info['numeric_cols']}</li>
                    <li><strong>Variables categóricas:</strong> {info['categorical_cols']}</li>
                    <li><strong>Variables de fecha/hora:</strong> {info['datetime_cols']}</li>
                </ul>
            </div>

            <div class="section">
                <h2>Distribución de Tipos de Datos</h2>
                <table>
                    <tr>
                        <th>Tipo de dato</th>
                        <th>Cantidad</th>
                    </tr>
        """

        for dtype, count in dtypes_counts.items():
            html += f"""
                    <tr>
                        <td>{dtype}</td>
                        <td>{count}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        """

        # Variables numéricas
        html += """
            <div class="section">
                <h2>Variables Numéricas</h2>
        """
        if not numeric_summary.empty:
            html += numeric_summary.reset_index().to_html(index=False)
            for col in self.numeric_cols:
                html += f"""
                <h3>{col}</h3>
                <img src="data:image/png;base64,{numeric_histograms[col]}" alt="Histograma de {col}">
                <img src="data:image/png;base64,{numeric_boxplots[col]}" alt="Boxplot de {col}">
                """
        else:
            html += "<p>No hay variables numéricas para resumir.</p>"
        html += "</div>"

        # Variables categóricas
        html += """
            <div class="section">
                <h2>Variables Categóricas</h2>
        """
        if categorical_summary:
            for col, counts in categorical_summary.items():
                html += f"""
                <h3>{col}</h3>
                {counts.to_frame().reset_index().to_html(index=False)}
                <img src="data:image/png;base64,{categorical_bar_charts[col]}" alt="Gráfico de barras de {col}">
                """
        else:
            html += "<p>No hay variables categóricas para resumir.</p>"
        html += "</div>"

        # Valores faltantes
        html += """
            <div class="section">
                <h2>Valores Faltantes</h2>
        """
        html += missing_summary.to_html(index=False)
        html += f"""
            <img src="data:image/png;base64,{missing_heatmap}" alt="Mapa de calor de valores faltantes">
        </div>
        """

        # Correlaciones
        if not correlation_matrix.empty:
            html += """
            <div class="section">
                <h2>Correlaciones</h2>
            """
            html += correlation_matrix.reset_index().to_html(index=False)
            html += f"""
                <img src="data:image/png;base64,{correlation_heatmap}" alt="Heatmap de correlaciones">
            </div>
            """
        else:
            html += """
            <div class="section">
                <h2>Correlaciones</h2>
                <p>No hay suficientes variables numéricas para calcular correlaciones.</p>
            </div>
            """

        html += """
        </body>
        </html>
        """
        return html
