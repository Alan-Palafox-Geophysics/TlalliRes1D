"""
marca_sev.py
==================================================================
Identidad visual de **TLALLI-RES 1D**: portada de la pantalla de
acceso y encabezado de las pantallas de trabajo.

Se aísla en su propio módulo para que las pantallas no repitan rutas
de archivo ni parámetros de presentación: cada una llama a
:func:`encabezado` y no necesita saber dónde viven las imágenes ni a
qué tamaño se muestran. Cambiar el logotipo o el ancho del banner se
hace aquí, en un solo lugar.

Las imágenes viven en ``app/assets``. Si faltan, las funciones
degradan a un encabezado de texto en vez de fallar: la aplicación debe
seguir siendo usable aunque los recursos gráficos no estén, por
ejemplo en un despliegue parcial.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger("sev.marca")

#: Nombre comercial del producto.
NOMBRE_PRODUCTO = "TLALLI-RES 1D"

#: Lema que acompaña al nombre.
LEMA = "Optimiza tu exploración con análisis inteligente"


def _carpeta_recursos() -> Path:
    """Carpeta de recursos gráficos (``app/assets``).

    Se localiza subiendo desde este módulo hasta encontrar la carpeta
    de la aplicación, de modo que funcione igual esté el módulo en
    ``src`` o en ``app``.
    """
    actual = Path(__file__).resolve().parent
    for candidato in (actual, *actual.parents):
        recursos = candidato / "app" / "assets"
        if recursos.is_dir():
            return recursos
        else:
            pass
        recursos = candidato / "assets"
        if recursos.is_dir():
            return recursos
        else:
            pass
    return actual / "assets"


def ruta_recurso(nombre: str) -> Optional[Path]:
    """Ruta de un recurso gráfico, o ``None`` si no existe."""
    ruta = _carpeta_recursos() / nombre
    if ruta.exists():
        return ruta
    else:
        logger.warning("Recurso gráfico no encontrado: %s", ruta)
        return None


def portada_login() -> None:
    """Muestra la portada de la pantalla de acceso.

    Es la primera impresión del producto, así que ocupa el ancho
    completo antes del formulario de credenciales.
    """
    imagen = ruta_recurso("portada_login.png")
    if imagen is not None:
        st.image(str(imagen), use_container_width=True)
    else:
        # Degradación: sin la imagen, la aplicación sigue siendo usable.
        st.title(NOMBRE_PRODUCTO)
        st.caption(LEMA)
    return None


def encabezado(subtitulo: Optional[str] = None) -> None:
    """Muestra el encabezado de identidad en una pantalla de trabajo.

    Parameters
    ----------
    subtitulo : str, optional
        Nombre de la pantalla, que se dibuja bajo el banner para que el
        usuario sepa dónde está sin perder la identidad del producto.
    """
    imagen = ruta_recurso("encabezado.png")
    if imagen is not None:
        st.image(str(imagen), use_container_width=True)
    else:
        st.subheader(NOMBRE_PRODUCTO)

    if subtitulo:
        st.markdown(f"### {subtitulo}")
    else:
        pass
    return None


def pie_de_pagina() -> None:
    """Pie discreto con el nombre del producto."""
    st.divider()
    st.caption(f"{NOMBRE_PRODUCTO} · {LEMA}")
    return None
