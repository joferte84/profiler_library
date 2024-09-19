# __init__.py

# Importación de clases y funciones desde los módulos del paquete

from .nans import NanHandler
from .outliers import OutlierHandler
from .profiler import DataProfiler
from .report import ReportGenerator
from .statistics_utils import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_variance,
    calculate_std_dev,
    calculate_iqr,
    calculate_percentiles,
    perform_t_test_one_sample,
    perform_t_test_independent,
    perform_chi_square_test,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    SimpleLinearRegression,
    normal_distribution_pdf,
    binomial_distribution_pmf,
    calculate_confidence_interval_mean,
)
from .utils import (
    is_valid_email,
    is_valid_date,
    is_valid_json,
    read_file,
    write_file,
    compress_file,
    decompress_file,
    convert_datetime_format,
    time_difference,
    timer,
    simple_cache,
    remove_whitespace,
    normalize_text,
    calculate_percentage,
)
from .visualizations import Visualizer

# Definición de __all__ para controlar lo que se importa con 'from package import *'

__all__ = [
    'NanHandler',
    'OutlierHandler',
    'DataProfiler',
    'ReportGenerator',
    'calculate_mean',
    'calculate_median',
    'calculate_mode',
    'calculate_variance',
    'calculate_std_dev',
    'calculate_iqr',
    'calculate_percentiles',
    'perform_t_test_one_sample',
    'perform_t_test_independent',
    'perform_chi_square_test',
    'calculate_pearson_correlation',
    'calculate_spearman_correlation',
    'SimpleLinearRegression',
    'normal_distribution_pdf',
    'binomial_distribution_pmf',
    'calculate_confidence_interval_mean',
    'is_valid_email',
    'is_valid_date',
    'is_valid_json',
    'read_file',
    'write_file',
    'compress_file',
    'decompress_file',
    'convert_datetime_format',
    'time_difference',
    'timer',
    'simple_cache',
    'remove_whitespace',
    'normalize_text',
    'calculate_percentage',
    'Visualizer',
]
