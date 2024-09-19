import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Union, List

# Medidas Descriptivas

def _validate_data(data: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
    """
    Valida y prepara los datos de entrada unidimensionales eliminando valores NaN.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        np.ndarray: Array NumPy unidimensional sin valores NaN.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs, o si no son unidimensionales.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Los datos deben ser un array unidimensional.")
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("Los datos no deben estar vacíos o contener solo NaNs.")
    return data

def _validate_paired_data(x: Union[pd.Series, np.ndarray, List[float]],
                          y: Union[pd.Series, np.ndarray, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valida y prepara dos conjuntos de datos emparejados, eliminando pares con valores NaN.

    Parámetros:
        x (Union[pd.Series, np.ndarray, List[float]]): Primer conjunto de datos.
        y (Union[pd.Series, np.ndarray, List[float]]): Segundo conjunto de datos.

    Retorna:
        Tuple[np.ndarray, np.ndarray]: Arrays NumPy unidimensionales sin valores NaN, de la misma longitud.

    Lanza:
        ValueError: Si los datos no son unidimensionales, no tienen la misma longitud,
                    o si después de eliminar NaNs hay menos de dos observaciones.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Las variables x e y deben ser arrays unidimensionales.")
    if len(x) != len(y):
        raise ValueError("Las variables x e y deben tener el mismo número de observaciones.")
    # Crear una máscara para pares donde ambos valores no son NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        raise ValueError("Después de eliminar NaNs, hay menos de dos observaciones.")
    return x_clean, y_clean

def calculate_mean(data: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calcula la media aritmética de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        float: Media aritmética de los datos.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    return np.mean(data)

def calculate_median(data: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calcula la mediana de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        float: Mediana de los datos.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    return np.median(data)

def calculate_mode(data: Union[pd.Series, np.ndarray, List[float]]) -> Union[float, List[float]]:
    """
    Calcula la moda de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        Union[float, List[float]]: Valor modal o lista de valores modales si hay múltiples modas.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    modes = stats.mode(data, keepdims=True)
    return modes.mode[0] if len(modes.mode) == 1 else modes.mode.flatten().tolist()

def calculate_variance(data: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calcula la varianza muestral de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        float: Varianza muestral de los datos.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    return np.var(data, ddof=1)

def calculate_std_dev(data: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calcula la desviación estándar muestral de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        float: Desviación estándar muestral de los datos.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    return np.std(data, ddof=1)

def calculate_iqr(data: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calcula el rango intercuartílico (IQR) de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.

    Retorna:
        float: Rango intercuartílico (IQR) de los datos.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    Q1 = np.percentile(data, 25, method='midpoint')
    Q3 = np.percentile(data, 75, method='midpoint')
    return Q3 - Q1

def calculate_percentiles(data: Union[pd.Series, np.ndarray, List[float]], percentiles: List[float]) -> np.ndarray:
    """
    Calcula los percentiles especificados de los datos.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.
        percentiles (List[float]): Lista de percentiles a calcular (valores entre 0 y 100).

    Retorna:
        np.ndarray: Valores de los percentiles calculados.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    return np.percentile(data, percentiles, method='linear')

# Pruebas de Hipótesis

def perform_t_test_one_sample(data: Union[pd.Series, np.ndarray, List[float]], popmean: float) -> Tuple[float, float]:
    """
    Realiza una prueba t de una muestra para comparar la media muestral con una media poblacional hipotética.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.
        popmean (float): Media poblacional hipotética.

    Retorna:
        Tuple[float, float]: Estadístico t y valor p de la prueba.

    Lanza:
        ValueError: Si la muestra tiene menos de dos observaciones válidas.
    """
    data = _validate_data(data)
    if len(data) < 2:
        raise ValueError("La muestra debe tener al menos dos observaciones.")
    t_stat, p_value = stats.ttest_1samp(data, popmean, nan_policy='omit')
    return t_stat, p_value

def perform_t_test_independent(sample1: Union[pd.Series, np.ndarray, List[float]],
                               sample2: Union[pd.Series, np.ndarray, List[float]],
                               equal_var: bool = True) -> Tuple[float, float]:
    """
    Realiza una prueba t para muestras independientes para comparar las medias de dos grupos.

    Parámetros:
        sample1 (Union[pd.Series, np.ndarray, List[float]]): Primera muestra de datos numéricos.
        sample2 (Union[pd.Series, np.ndarray, List[float]]): Segunda muestra de datos numéricos.
        equal_var (bool): Si se asume varianza igual entre los grupos (por defecto True).

    Retorna:
        Tuple[float, float]: Estadístico t y valor p de la prueba.

    Lanza:
        ValueError: Si alguna de las muestras tiene menos de dos observaciones válidas.
    """
    sample1 = _validate_data(sample1)
    sample2 = _validate_data(sample2)
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Cada muestra debe tener al menos dos observaciones.")
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return t_stat, p_value

def perform_chi_square_test(observed: Union[pd.Series, np.ndarray, List[float]],
                            expected: Optional[Union[pd.Series, np.ndarray, List[float]]] = None) -> Tuple[float, float, int, np.ndarray]:
    """
    Realiza una prueba de chi-cuadrado para evaluar si las frecuencias observadas difieren significativamente de las esperadas.

    Parámetros:
        observed (Union[pd.Series, np.ndarray, List[float]]): Frecuencias observadas.
        expected (Optional[Union[pd.Series, np.ndarray, List[float]]]): Frecuencias esperadas.
            Si no se proporcionan, se asume una distribución uniforme.

    Retorna:
        Tuple[float, float, int, np.ndarray]:
            - chi2_stat (float): Estadístico chi-cuadrado calculado.
            - p_value (float): Valor p de la prueba.
            - dof (int): Grados de libertad.
            - expected (np.ndarray): Frecuencias esperadas utilizadas en la prueba.

    Lanza:
        ValueError: Si 'observed' y 'expected' tienen formas diferentes,
                    o si contienen valores NaN.
    """
    observed = np.asarray(observed)
    if expected is not None:
        expected = np.asarray(expected)
        if observed.shape != expected.shape:
            raise ValueError("Las matrices 'observed' y 'expected' deben tener la misma forma.")
    # Verificar que no hay valores NaN
    if np.isnan(observed).any() or (expected is not None and np.isnan(expected).any()):
        raise ValueError("Los datos no deben contener NaNs.")
    chi2_stat, p_value, dof, ex = stats.chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat, p_value, dof, ex

# Análisis de Correlación

def calculate_pearson_correlation(x: Union[pd.Series, np.ndarray, List[float]],
                                  y: Union[pd.Series, np.ndarray, List[float]]) -> Tuple[float, float]:
    """
    Calcula el coeficiente de correlación de Pearson entre dos variables.

    Parámetros:
        x (Union[pd.Series, np.ndarray, List[float]]): Primera variable.
        y (Union[pd.Series, np.ndarray, List[float]]): Segunda variable.

    Retorna:
        Tuple[float, float]: Coeficiente de correlación de Pearson y valor p.

    Lanza:
        ValueError: Si las variables no tienen el mismo número de observaciones válidas,
                    o si después de eliminar NaNs hay menos de dos observaciones.
    """
    x_clean, y_clean = _validate_paired_data(x, y)
    corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
    return corr_coef, p_value

def calculate_spearman_correlation(x: Union[pd.Series, np.ndarray, List[float]],
                                   y: Union[pd.Series, np.ndarray, List[float]]) -> Tuple[float, float]:
    """
    Calcula el coeficiente de correlación de Spearman entre dos variables.

    Parámetros:
        x (Union[pd.Series, np.ndarray, List[float]]): Primera variable.
        y (Union[pd.Series, np.ndarray, List[float]]): Segunda variable.

    Retorna:
        Tuple[float, float]: Coeficiente de correlación de Spearman y valor p.

    Lanza:
        ValueError: Si las variables no tienen el mismo número de observaciones válidas,
                    o si después de eliminar NaNs hay menos de dos observaciones.
    """
    x_clean, y_clean = _validate_paired_data(x, y)
    corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
    return corr_coef, p_value

# Regresión Lineal Simple

class SimpleLinearRegression:
    """
    Clase para realizar regresión lineal simple entre una variable independiente y una dependiente.
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_value = None
        self.p_value = None
        self.std_err = None

    def fit(self, x: Union[pd.Series, np.ndarray, List[float]], y: Union[pd.Series, np.ndarray, List[float]]):
        """
        Ajusta el modelo de regresión lineal simple a los datos proporcionados.

        Parámetros:
            x (Union[pd.Series, np.ndarray, List[float]]): Variable independiente.
            y (Union[pd.Series, np.ndarray, List[float]]): Variable dependiente.

        Lanza:
            ValueError: Si las variables no tienen el mismo número de observaciones válidas,
                        o si después de eliminar NaNs hay menos de dos observaciones.
        """
        x_clean, y_clean = _validate_paired_data(x, y)
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(x_clean, y_clean)

    def predict(self, x: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """
        Predice los valores de y para los valores de x proporcionados utilizando el modelo ajustado.

        Parámetros:
            x (Union[pd.Series, np.ndarray, List[float]]): Valores de la variable independiente.

        Retorna:
            np.ndarray: Valores predichos de la variable dependiente.

        Lanza:
            ValueError: Si el modelo no ha sido ajustado aún.
        """
        x = np.asarray(x)
        return self.intercept + self.slope * x

    def get_params(self) -> dict:
        """
        Devuelve los parámetros del modelo ajustado.

        Retorna:
            dict: Diccionario con los parámetros 'slope', 'intercept', 'r_value', 'p_value' y 'std_err'.

        Lanza:
            ValueError: Si el modelo no ha sido ajustado aún.
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("El modelo no ha sido ajustado. No hay parámetros para devolver.")
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'r_value': self.r_value,
            'p_value': self.p_value,
            'std_err': self.std_err
        }

    def score(self) -> float:
        """
        Calcula el coeficiente de determinación R² del modelo ajustado.

        Retorna:
            float: Coeficiente de determinación R².

        Lanza:
            ValueError: Si el modelo no ha sido ajustado aún.
        """
        return self.r_value ** 2

# Distribuciones de Probabilidad

def normal_distribution_pdf(x: Union[float, np.ndarray], mean: float = 0, std: float = 1) -> Union[float, np.ndarray]:
    """
    Calcula la función de densidad de probabilidad (PDF) de una distribución normal para los valores dados.

    Parámetros:
        x (Union[float, np.ndarray]): Valor o valores en los que evaluar la PDF.
        mean (float): Media de la distribución normal (por defecto 0).
        std (float): Desviación estándar de la distribución normal (por defecto 1).

    Retorna:
        Union[float, np.ndarray]: Valor o valores de la PDF evaluados en x.
    """
    return stats.norm.pdf(x, loc=mean, scale=std)

def binomial_distribution_pmf(k: int, n: int, p: float) -> float:
    """
    Calcula la función de masa de probabilidad (PMF) de una distribución binomial.

    Parámetros:
        k (int): Número de éxitos observados.
        n (int): Número de ensayos.
        p (float): Probabilidad de éxito en un ensayo individual.

    Retorna:
        float: Valor de la PMF para los parámetros dados.
    """
    return stats.binom.pmf(k, n, p)

# Intervalos de Confianza

def calculate_confidence_interval_mean(data: Union[pd.Series, np.ndarray, List[float]], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza para la media de los datos con un nivel de confianza dado.

    Parámetros:
        data (Union[pd.Series, np.ndarray, List[float]]): Serie de datos numéricos.
        confidence (float): Nivel de confianza (entre 0 y 1). Por defecto 0.95.

    Retorna:
        Tuple[float, float]: Limite inferior y superior del intervalo de confianza.

    Lanza:
        ValueError: Si los datos están vacíos o solo contienen NaNs.
    """
    data = _validate_data(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - h, mean + h
