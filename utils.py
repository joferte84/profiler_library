# utils.py

import re
import json
import datetime
from typing import Any, Union, Dict, List

def is_valid_email(email: str) -> bool:
    """
    Verifica si una cadena de texto es un correo electrónico válido.

    Parámetros:
        email (str): Dirección de correo electrónico a validar.

    Retorna:
        bool: True si es un correo electrónico válido, False en caso contrario.
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def is_valid_date(date_string: str, date_format: str = "%Y-%m-%d") -> bool:
    """
    Verifica si una cadena de texto corresponde a una fecha válida según el formato especificado.

    Parámetros:
        date_string (str): Cadena de texto que representa una fecha.
        date_format (str): Formato de fecha esperado (por defecto "%Y-%m-%d").

    Retorna:
        bool: True si la fecha es válida, False en caso contrario.
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
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

import os
import gzip
import shutil

def read_file(filepath: str, mode: str = 'r') -> str:
    """
    Lee el contenido de un archivo y lo retorna como una cadena de texto.

    Parámetros:
        filepath (str): Ruta del archivo.
        mode (str): Modo de lectura (por defecto 'r').

    Retorna:
        str: Contenido del archivo.
    """
    with open(filepath, mode, encoding='utf-8') as file:
        return file.read()

def write_file(filepath: str, content: str, mode: str = 'w') -> None:
    """
    Escribe una cadena de texto en un archivo.

    Parámetros:
        filepath (str): Ruta del archivo.
        content (str): Contenido a escribir.
        mode (str): Modo de escritura (por defecto 'w').
    """
    with open(filepath, mode, encoding='utf-8') as file:
        file.write(content)

def compress_file(input_filepath: str, output_filepath: str) -> None:
    """
    Comprime un archivo usando gzip.

    Parámetros:
        input_filepath (str): Ruta del archivo a comprimir.
        output_filepath (str): Ruta donde se guardará el archivo comprimido.
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
    """
    with gzip.open(input_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def convert_datetime_format(date_string: str, input_format: str, output_format: str) -> str:
    """
    Convierte una fecha de un formato a otro.

    Parámetros:
        date_string (str): Fecha en formato de entrada.
        input_format (str): Formato de la fecha de entrada.
        output_format (str): Formato deseado de salida.

    Retorna:
        str: Fecha en el formato de salida.
    """
    dt = datetime.datetime.strptime(date_string, input_format)
    return dt.strftime(output_format)

def time_difference(start_time: str, end_time: str, time_format: str = "%H:%M:%S") -> datetime.timedelta:
    """
    Calcula la diferencia de tiempo entre dos horas.

    Parámetros:
        start_time (str): Hora de inicio.
        end_time (str): Hora de fin.
        time_format (str): Formato de las horas (por defecto "%H:%M:%S").

    Retorna:
        datetime.timedelta: Diferencia de tiempo entre las dos horas.
    """
    t1 = datetime.datetime.strptime(start_time, time_format)
    t2 = datetime.datetime.strptime(end_time, time_format)
    return t2 - t1

import functools
import time

def timer(func):
    """
    Decorador que mide el tiempo de ejecución de una función.

    Parámetros:
        func (callable): Función a decorar.

    Retorna:
        callable: Función decorada.
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
