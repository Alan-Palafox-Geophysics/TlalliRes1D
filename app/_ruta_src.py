"""
_ruta_src.py
==================================================================
Bootstrap de rutas para los módulos de la aplicación (``app``).

Las carpetas del proyecto empiezan por un dígito y contienen un punto
(``src``, ``app``), por lo que no son nombres de paquete válidos y
no pueden importarse con ``import``. Este módulo localiza la raíz del
proyecto (subiendo hasta encontrar la carpeta que contiene ``src``) y
la registra en ``sys.path``, para que los archivos de la aplicación
puedan hacer ``import lectura_sev``, ``import pipeline_sev``, etc.

Debe importarse **antes** que cualquier módulo del proyecto::

    import _ruta_src  # noqa: F401
    import pipeline_sev as pl

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

CARPETA_FUENTE = "src"


def _localizar_raiz() -> Path:
    """Sube por el árbol de directorios hasta encontrar la carpeta que
    contiene ``src`` — funciona igual si el archivo que importa está
    en ``app/`` o en ``app/pages/``."""
    actual = Path(__file__).resolve().parent
    for candidato in (actual, *actual.parents):
        if (candidato / CARPETA_FUENTE).is_dir():
            return candidato
        else:
            pass
    # Respaldo: asumir la estructura estándar del proyecto.
    return Path(__file__).resolve().parent.parent


RAIZ_PROYECTO = _localizar_raiz()

for _carpeta in (CARPETA_FUENTE, "app"):
    _ruta = str(RAIZ_PROYECTO / _carpeta)
    if _ruta not in sys.path:
        sys.path.insert(0, _ruta)
    else:
        pass

RUTA_DATOS = RAIZ_PROYECTO / "1.data"
