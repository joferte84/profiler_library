# utils.py

import re
import json
import datetime
import os
import gzip
import shutil
import functools
import time
import unicodedata
from typing import Any, Union, Dict, List

# Funciones de Validación

def is_valid_email(email: str) -> bool:
    """
    Verifica si una cadena de texto es un correo electrónico válido.

    Parámetros:
        email (str): Dirección de correo electrónico a validar.

    Retorna:
        bool: True si es un correo electrónico válido, False en caso contrario.

    Ejemplo:
        >>> is_valid_email("usuario@example.com")
        True
        >>> is_valid_email("usuario@ejemplo")
        False
    """
    pattern = r'^[A-Za-z0-9\._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_date(date_string: str, date_format: str = "%Y-%m-%d") -> bool:
    """
    Verifica si una cadena de texto corresponde a una fecha válida según el formato especificado.

    Parámetros:
        date_string (str): Cadena de texto que representa una fecha.
        date_format (str): Formato de fecha esperado (por defecto "%Y-%m-%d").

    Retorna:
        bool: True si la fecha es válida, False en caso contrario.

    Ejemplo:
        >>> is_valid_date("2023-10-15")
        True
        >>> is_valid_date("15/10/2023", date_format="%d/%m/%Y")
        True
        >>> is_valid_date("2023/15/10")
        False
    """
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False

def is_valid_json(json_string: str) -> bool:
    """
    Verifica si una cadena de texto es un JSON válido.

    Parámetros:
        json_string (str): Cadena de texto a validar.

    Retorna:
        bool: True si es un JSON válido, False en caso contrario.

    Ejemplo:
        >>> is_valid_json('{"nombre": "Juan", "edad": 30}')
        True
        >>> is_valid_json('{"nombre": "Juan", "edad": }')
        False
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

# Funciones de Manejo de Archivos

def read_file(filepath: str, mode: str = 'r') -> str:
    """
    Lee el contenido de un archivo y lo retorna como una cadena de texto.

    Parámetros:
        filepath (str): Ruta del archivo.
        mode (str): Modo de lectura (por defecto 'r').

    Retorna:
        str: Contenido del archivo.

    Lanza:
        IOError: Si ocurre un error al leer el archivo.

    Ejemplo:
        >>> contenido = read_file("archivo.txt")
        >>> print(contenido)
    """
    try:
        with open(filepath, mode, encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error al leer el archivo {filepath}: {e}")
        raise

def write_file(filepath: str, content: str, mode: str = 'w') -> None:
    """
    Escribe una cadena de texto en un archivo.

    Parámetros:
        filepath (str): Ruta del archivo.
        content (str): Contenido a escribir.
        mode (str): Modo de escritura (por defecto 'w').

    Lanza:
        IOError: Si ocurre un error al escribir en el archivo.

    Ejemplo:
        >>> write_file("archivo.txt", "Hola, mundo!")
    """
    try:
        with open(filepath, mode, encoding='utf-8') as file:
            file.write(content)
    except IOError as e:
        print(f"Error al escribir en el archivo {filepath}: {e}")
        raise

def compress_file(input_filepath: str, output_filepath: str) -> None:
    """
    Comprime un archivo usando gzip.

    Parámetros:
        input_filepath (str): Ruta del archivo a comprimir.
        output_filepath (str): Ruta donde se guardará el archivo comprimido.

    Ejemplo:
        >>> compress_file("datos.txt", "datos.txt.gz")
    """
    with open(input_filepath, 'rb') as f_in:
        with gzip.open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def decompress_file(input_filepath: str, output_filepath: str) -> None:
    """
    Descomprime un archivo gzip.

    Parámetros:
        input_filepath (str): Ruta del archivo comprimido.
        output_filepath (str): Ruta donde se guardará el archivo descomprimido.

    Ejemplo:
        >>> decompress_file("datos.txt.gz", "datos.txt")
    """
    with gzip.open(input_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Funciones de Tiempo y Fecha

def convert_datetime_format(date_string: str, input_format: str, output_format: str) -> str:
    """
    Convierte una fecha de un formato a otro.

    Parámetros:
        date_string (str): Fecha en formato de entrada.
        input_format (str): Formato de la fecha de entrada.
        output_format (str): Formato deseado de salida.

    Retorna:
        str: Fecha en el formato de salida.

    Lanza:
        ValueError: Si la fecha no coincide con el formato de entrada.

    Ejemplo:
        >>> convert_datetime_format("15/10/2023", "%d/%m/%Y", "%Y-%m-%d")
        '2023-10-15'
    """
    try:
        dt = datetime.datetime.strptime(date_string, input_format)
        return dt.strftime(output_format)
    except ValueError as e:
        print(f"Error al convertir la fecha: {e}")
        raise

def time_difference(start_time: str, end_time: str, time_format: str = "%H:%M:%S") -> datetime.timedelta:
    """
    Calcula la diferencia de tiempo entre dos horas.

    Parámetros:
        start_time (str): Hora de inicio.
        end_time (str): Hora de fin.
        time_format (str): Formato de las horas (por defecto "%H:%M:%S").

    Retorna:
        datetime.timedelta: Diferencia de tiempo entre las dos horas.

    Ejemplo:
        >>> delta = time_difference("08:30:00", "12:45:00")
        >>> print(delta)
        4:15:00
    """
    t1 = datetime.datetime.strptime(start_time, time_format)
    t2 = datetime.datetime.strptime(end_time, time_format)
    return t2 - t1

# Decoradores Utilitarios

def timer(func):
    """
    Decorador que mide el tiempo de ejecución de una función.

    Parámetros:
        func (callable): Función a decorar.

    Retorna:
        callable: Función decorada.

    Ejemplo:
        @timer
        def procesar_datos():
            # Código de procesamiento
            pass
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Función '{func.__name__}' ejecutada en {run_time:.4f} segundos")
        return value
    return wrapper_timer

def simple_cache(func):
    """
    Decorador que almacena en caché los resultados de una función para entradas específicas.

    Parámetros:
        func (callable): Función a decorar.

    Retorna:
        callable: Función decorada con caché simple.

    Ejemplo:
        @simple_cache
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    cache = {}

    @functools.wraps(func)
    def wrapper_cache(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return wrapper_cache

# Funciones de Procesamiento de Cadenas

def remove_whitespace(text: str) -> str:
    """
    Elimina espacios en blanco al inicio y al final de una cadena y reduce múltiples espacios internos a uno solo.

    Parámetros:
        text (str): Cadena de texto a procesar.

    Retorna:
        str: Cadena de texto sin espacios redundantes.

    Ejemplo:
        >>> remove_whitespace("  Hola   mundo  ")
        'Hola mundo'
    """
    return ' '.join(text.strip().split())

def normalize_text(text: str) -> str:
    """
    Normaliza una cadena de texto eliminando acentos y convirtiendo a minúsculas.

    Parámetros:
        text (str): Cadena de texto a normalizar.

    Retorna:
        str: Cadena de texto normalizada.

    Ejemplo:
        >>> normalize_text("Canción")
        'cancion'
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return text

# Funciones Matemáticas Adicionales

def calculate_percentage(part: float, whole: float) -> float:
    """
    Calcula el porcentaje que representa 'part' del 'whole'.

    Parámetros:
        part (float): Parte del total.
        whole (float): Total.

    Retorna:
        float: Porcentaje correspondiente.

    Lanza:
        ValueError: Si 'whole' es cero.

    Ejemplo:
        >>> calculate_percentage(50, 200)
        25.0
    """
    if whole == 0:
        raise ValueError("El total no puede ser cero.")
    return (part / whole) * 100
