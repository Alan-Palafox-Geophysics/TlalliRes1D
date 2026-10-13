"""
rutas_sev.py
==================================================================
Resolución de rutas del proyecto.

Las carpetas del proyecto (``src``, ``app``) empiezan por un
dígito y contienen un punto, así que **no son nombres de paquete
válidos en Python** y no pueden importarse con ``import``. Este módulo
resuelve esa restricción registrando ambas carpetas en ``sys.path``,
de modo que los módulos se importen de forma plana
(``import lectura_sev``) sin importar desde cuál de las dos carpetas
se ejecute el código.

Se importa una sola vez, al inicio de los módulos que necesitan cruzar
la frontera entre carpetas (p. ej. ``pipeline_sev`` en ``src``, que
usa ``almacenamiento_sev`` en ``app``). Es idempotente: importarlo
varias veces no duplica entradas en ``sys.path``.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Carpeta raíz del proyecto (la que contiene ``src`` y ``app``).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

#: Carpetas de código que deben ser importables de forma plana.
CARPETAS_CODIGO = ("src", "app")

#: Carpeta de datos de entrada del proyecto.
RUTA_DATOS = RAIZ_PROYECTO / "1.data"


def registrar_rutas() -> list[str]:
    """Añade las carpetas de código del proyecto a ``sys.path``.

    Returns
    -------
    list[str]
        Rutas efectivamente registradas (en orden de prioridad).
    """
    registradas: list[str] = []
    for carpeta in CARPETAS_CODIGO:
        ruta = str(RAIZ_PROYECTO / carpeta)
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
        else:
            pass
        registradas.append(ruta)
    return registradas


# Se ejecuta al importar: es el propósito del módulo.
registrar_rutas()
