import pandas as pd
from .nans import NanHandler
from .outliers import OutliersHandler
from .statistics import Statistics
from .visualizations import Visualizations
from .report import ReportGenerator

class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}

    def generate_report(self, visualize=True, handle_nans=True, handle_outliers=True):
        """Genera un informe completo con opciones de configuración."""
        if handle_nans:
            self.df = NanHandler(self.df).handle_nans()
        
        if handle_outliers:
            outliers_report = OutliersHandler(self.df).handle_outliers()
            self.report['outliers'] = outliers_report
        
        self.report['statistics'] = Statistics(self.df).get_basic_statistics()
        
        if visualize:
            self.report['visualizations'] = Visualizations(self.df).generate_all_plots()
        
        return self.report

    def export_report(self, format='html'):
        """Exporta el informe en el formato deseado."""
        return ReportGenerator(self.report).export(format)
