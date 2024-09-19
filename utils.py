import re
import json
import datetime
from typing import Any, Union, Dict, List
import gzip
import shutil
import functools
import time

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

    Lanza:
        ValueError: Si la fecha no coincide con el formato de entrada.
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
    """
    t1 = datetime.datetime.strptime(start_time, time_format)
    t2 = datetime.datetime.strptime(end_time, time_format)
    return t2 - t1

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
