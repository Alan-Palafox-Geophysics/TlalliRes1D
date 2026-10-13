"""
estado_app.py
==================================================================
Estado de sesión y acceso a datos de la aplicación Streamlit.

Mismo principio que TlalliRes2D: ``st.session_state`` guarda **solo
punteros** (ruta del proyecto, sondeo activo, configuración); los datos
se releen de disco con un caché acotado (``max_entries`` y ``ttl``),
para que la memoria no crezca con la campaña.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import almacenamiento_sev as alm
import escalas_color_tre as esc
import paquete_sev as pk
import pipeline_sev as pl
from almacenamiento_sev import Proyecto

logger = logging.getLogger("sev.app")

CLAVE_PROYECTO = "ruta_proyecto"
CLAVE_SONDEO = "sondeo_activo"
CLAVE_CONFIG = "config_pipeline"
CLAVE_LOGS = "logs_ejecucion"
CLAVE_ESPACIO = "espacio_trabajo"
CLAVE_SIN_DESCARGAR = "cambios_sin_descargar"
CLAVE_TOKEN = "token_cancelacion"
CLAVE_EJECUTANDO = "pipeline_ejecutando"
CLAVE_MODO_ESCALA = "modo_escala_color"
CLAVE_N_NIVELES_ESCALA = "n_niveles_escala"

CACHE_MAX_ENTRADAS = 8
CACHE_TTL_SEGUNDOS = 900


def inicializar_estado() -> None:
    for clave, valor in {
        CLAVE_PROYECTO: None, CLAVE_SONDEO: None, CLAVE_CONFIG: pl.ConfigPipeline(), CLAVE_LOGS: [],
    }.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor
        else:
            pass


# ----------------------------------------------------------------------
# Espacio de trabajo temporal (ver paquete_sev)
# ----------------------------------------------------------------------


def _identificador_sesion() -> str:
    import uuid

    if "id_espacio_sesion" not in st.session_state:
        st.session_state["id_espacio_sesion"] = uuid.uuid4().hex[:16]
    else:
        pass
    return st.session_state["id_espacio_sesion"]


def espacio_trabajo() -> Path:
    if CLAVE_ESPACIO not in st.session_state:
        pk.limpiar_espacios_caducados()
        st.session_state[CLAVE_ESPACIO] = str(pk.crear_espacio(_identificador_sesion()))
    else:
        pass
    ruta = Path(st.session_state[CLAVE_ESPACIO])
    if not ruta.exists():
        ruta = pk.crear_espacio(_identificador_sesion())
        st.session_state[CLAVE_ESPACIO] = str(ruta)
    else:
        pk.marcar_actividad(ruta)
    return ruta


def liberar_espacio_trabajo() -> None:
    ruta = st.session_state.get(CLAVE_ESPACIO)
    if ruta:
        pk.eliminar_espacio(ruta)
        st.session_state.pop(CLAVE_ESPACIO, None)
    else:
        pass
    st.session_state[CLAVE_SIN_DESCARGAR] = False


def marcar_cambios_sin_descargar(hay: bool = True) -> None:
    st.session_state[CLAVE_SIN_DESCARGAR] = hay


def hay_cambios_sin_descargar() -> bool:
    return bool(st.session_state.get(CLAVE_SIN_DESCARGAR, False))


# ----------------------------------------------------------------------
# Proyecto, sondeo y configuración
# ----------------------------------------------------------------------


def fijar_proyecto(proyecto: Proyecto) -> None:
    st.session_state[CLAVE_PROYECTO] = str(proyecto.ruta)
    st.session_state[CLAVE_SONDEO] = None
    limpiar_cache_datos()
    registrar_log(f"Proyecto activo: {proyecto.nombre}")
    # Parámetros de la campaña guardados en el proyecto: se usan por omisión.
    guardados = pl.cargar_parametros(proyecto)
    if guardados is not None:
        st.session_state[CLAVE_CONFIG] = guardados
        registrar_log("Parámetros de la campaña cargados del proyecto.")
    else:
        pass


def obtener_proyecto() -> Optional[Proyecto]:
    ruta = st.session_state.get(CLAVE_PROYECTO)
    if not ruta:
        return None
    else:
        pass
    try:
        return alm.abrir_proyecto(ruta)
    except alm.ProyectoNoEncontradoError:
        st.session_state[CLAVE_PROYECTO] = None
        return None


def fijar_sondeo(sondeo: Optional[str]) -> None:
    st.session_state[CLAVE_SONDEO] = sondeo


def obtener_sondeo() -> Optional[str]:
    return st.session_state.get(CLAVE_SONDEO)


def obtener_config() -> pl.ConfigPipeline:
    config = st.session_state.get(CLAVE_CONFIG)
    if config is None:
        config = pl.ConfigPipeline()
        st.session_state[CLAVE_CONFIG] = config
    else:
        pass
    return config


def fijar_config(config: pl.ConfigPipeline) -> None:
    st.session_state[CLAVE_CONFIG] = config


# ----------------------------------------------------------------------
# Registro
# ----------------------------------------------------------------------


def registrar_log(mensaje: str, nivel: str = "info") -> None:
    from datetime import datetime

    st.session_state.setdefault(CLAVE_LOGS, [])
    st.session_state[CLAVE_LOGS].append(
        {"hora": datetime.now().strftime("%H:%M:%S"), "nivel": nivel, "mensaje": mensaje})
    st.session_state[CLAVE_LOGS] = st.session_state[CLAVE_LOGS][-200:]
    logger.info("[app] %s", mensaje)


def registrar_resultado(resultado: pl.ResultadoEtapa) -> None:
    nivel = "info" if resultado else "error"
    registrar_log(f"[{resultado.etapa}] {resultado.mensaje} ({resultado.duracion_s:.1f}s)", nivel)
    if resultado and resultado.etapa not in ("paquete", "proyecto"):
        marcar_cambios_sin_descargar(True)
        st.session_state.pop("rango_color_campana", None)
    else:
        pass
    for advertencia in resultado.advertencias:
        registrar_log(f"[{resultado.etapa}] ⚠ {advertencia}", "warning")


def mostrar_logs(altura: int = 220) -> None:
    logs = st.session_state.get(CLAVE_LOGS, [])
    if not logs:
        st.caption("Sin actividad registrada en esta sesión.")
        return None
    else:
        pass
    iconos = {"info": "•", "warning": "⚠", "error": "✖"}
    texto = "\n".join(f"{iconos.get(e['nivel'], '•')} [{e['hora']}] {e['mensaje']}" for e in reversed(logs))
    st.text_area("Registro de ejecución", texto, height=altura, disabled=True, key=f"logs_{len(logs)}")
    return None


CLAVE_AVISO = "aviso_pendiente"


def avisar(resultado: pl.ResultadoEtapa, cancelado: bool = False) -> None:
    """Guarda el resultado de una acción para mostrarlo tras ``st.rerun()``.

    Las acciones relanzan la página para que las tarjetas, el selector de
    sondeos y el panel de descarga —dibujados antes del botón— reflejen
    lo que acaba de pasar. Sin esto, el mensaje se perdería en el
    relanzamiento.
    """
    tipo = "exito" if resultado else ("cancelado" if cancelado else "error")
    st.session_state[CLAVE_AVISO] = {"tipo": tipo, "mensaje": resultado.mensaje,
                                     "advertencias": list(resultado.advertencias)}


def mostrar_aviso() -> None:
    """Muestra (una sola vez) el aviso guardado por :func:`avisar`."""
    aviso = st.session_state.pop(CLAVE_AVISO, None)
    if aviso is None:
        return None
    else:
        pass
    if aviso["tipo"] == "exito":
        st.success(aviso["mensaje"])
    else:
        if aviso["tipo"] == "cancelado":
            st.warning(aviso["mensaje"])
        else:
            st.error(aviso["mensaje"])
    for advertencia in aviso["advertencias"][:6]:
        st.warning(advertencia, icon="⚠️")
    if len(aviso["advertencias"]) > 6:
        st.caption(f"… y {len(aviso['advertencias']) - 6} advertencia(s) más en el registro de ejecución.")
    else:
        pass
    return None


def mostrar_advertencias(resultado: pl.ResultadoEtapa, maximo: int = 6) -> None:
    """Muestra las advertencias de un resultado bajo su mensaje."""
    for advertencia in resultado.advertencias[:maximo]:
        st.warning(advertencia, icon="⚠️")
    if len(resultado.advertencias) > maximo:
        st.caption(f"… y {len(resultado.advertencias) - maximo} advertencia(s) más en el registro.")
    else:
        pass


# ----------------------------------------------------------------------
# Cancelación
# ----------------------------------------------------------------------


def obtener_token_cancelacion() -> pl.TokenCancelacion:
    if CLAVE_TOKEN not in st.session_state:
        st.session_state[CLAVE_TOKEN] = pl.TokenCancelacion()
    else:
        pass
    return st.session_state[CLAVE_TOKEN]


def solicitar_detener() -> None:
    obtener_token_cancelacion().solicitar_cancelacion()
    registrar_log("Detención solicitada por el usuario.", "warning")


def reiniciar_token_cancelacion() -> None:
    obtener_token_cancelacion().reiniciar()


def marcar_ejecutando(activo: bool) -> None:
    st.session_state[CLAVE_EJECUTANDO] = activo


# ----------------------------------------------------------------------
# Carga de datos cacheada
# ----------------------------------------------------------------------


def limpiar_cache_datos() -> None:
    cargar_tabla_cacheada.clear()
    cargar_campana_cacheada.clear()


@st.cache_data(max_entries=CACHE_MAX_ENTRADAS, ttl=CACHE_TTL_SEGUNDOS, show_spinner=False)
def cargar_tabla_cacheada(ruta_proyecto: str, etapa: str, tabla: str, sondeo: str,
                          marca: float = 0.0) -> Optional[pd.DataFrame]:
    return alm.cargar_tabla(alm.abrir_proyecto(ruta_proyecto), etapa, tabla, sondeo)


def cargar_tabla(etapa: str, tabla: str, sondeo: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Tabla del sondeo activo (o del indicado), con invalidación por
    fecha de modificación del archivo."""
    proyecto = obtener_proyecto()
    nombre = sondeo if sondeo is not None else obtener_sondeo()
    if proyecto is None or nombre is None:
        return None
    else:
        pass
    ruta = proyecto.ruta_tabla(etapa, tabla, nombre)
    if not ruta.exists():
        return None
    else:
        pass
    return cargar_tabla_cacheada(str(proyecto.ruta), etapa, tabla, str(nombre), marca=ruta.stat().st_mtime)


@st.cache_data(max_entries=4, ttl=CACHE_TTL_SEGUNDOS, show_spinner=False)
def cargar_campana_cacheada(ruta_proyecto: str, etapa: str, tabla: str, marca: float) -> pd.DataFrame:
    return alm.consultar_multisondeo(alm.abrir_proyecto(ruta_proyecto), etapa, tabla)


def cargar_campana(etapa: str, tabla: str) -> pd.DataFrame:
    proyecto = obtener_proyecto()
    if proyecto is None:
        return pd.DataFrame()
    else:
        pass
    archivos = list(proyecto.ruta_etapa(etapa).glob(f"SEV=*/{tabla}.parquet"))
    marca = max((a.stat().st_mtime for a in archivos), default=0.0) + len(archivos)
    return cargar_campana_cacheada(str(proyecto.ruta), etapa, tabla, marca)


# ----------------------------------------------------------------------
# Escala de color de campaña
# ----------------------------------------------------------------------


def rango_color_proyecto() -> Optional[list]:
    proyecto = obtener_proyecto()
    if proyecto is None:
        return None
    else:
        pass
    if "rango_color_campana" not in st.session_state:
        st.session_state["rango_color_campana"] = alm.rango_color_campana(proyecto)
    else:
        pass
    return st.session_state["rango_color_campana"]


def escala_color_activa(rango: Optional[list] = None):
    """Escala de resistividad (``escalas_color_tre``) para las secciones."""
    modo = st.session_state.get(CLAVE_MODO_ESCALA, esc.MODO_ESCALA_DEFECTO)
    n = int(st.session_state.get(CLAVE_N_NIVELES_ESCALA, esc.N_NIVELES_DEFECTO))
    return esc.construir_escala(modo if modo != esc.MODO_CUSTOMIZABLE else esc.MODO_ESCALA_DEFECTO,
                                rango or rango_color_proyecto(), n_niveles=n)


# ----------------------------------------------------------------------
# Utilidades de interfaz
# ----------------------------------------------------------------------


def tabla_legible(df: pd.DataFrame, cifras: int = 4) -> pd.DataFrame:
    """Copia para mostrar: números con ``cifras`` significativas, ∞ para
    el espesor del semiespacio y vacío para lo que no aplica.

    Se convierte a texto columna por columna para que Arrow no tenga que
    adivinar el tipo de una columna con números y "∞" mezclados.
    """
    import numpy as np

    salida = df.copy()
    for columna in salida.columns:
        serie = salida[columna]
        if pd.api.types.is_bool_dtype(serie) or not pd.api.types.is_numeric_dtype(serie):
            continue
        else:
            pass
        salida[columna] = [
            "∞" if (isinstance(v, float) and np.isinf(v)) else ("" if pd.isna(v) else f"{v:.{cifras}g}")
            for v in serie
        ]
    return salida


def sondeos_disponibles() -> list[str]:
    proyecto = obtener_proyecto()
    if proyecto is None:
        return []
    else:
        pass
    df = alm.listar_sondeos(proyecto)
    return df["sondeo"].astype(str).tolist() if len(df) else []


def selector_sondeo(etiqueta: str = "Sondeo activo", key: str = "sel_sondeo") -> Optional[str]:
    sondeos = sondeos_disponibles()
    if not sondeos:
        st.info("No hay sondeos cargados en este proyecto todavía.")
        return None
    else:
        pass
    activo = obtener_sondeo()
    indice = sondeos.index(activo) if activo in sondeos else 0
    seleccion = st.selectbox(etiqueta, sondeos, index=indice, key=key)
    fijar_sondeo(seleccion)
    return seleccion


def requiere_proyecto() -> Optional[Proyecto]:
    proyecto = obtener_proyecto()
    if proyecto is None:
        st.warning("Primero cree o abra un proyecto en la pantalla **Proyecto y datos**.")
        return None
    else:
        return proyecto


def mostrar_estado_pipeline(sondeo: str) -> None:
    proyecto = obtener_proyecto()
    if proyecto is None:
        return None
    else:
        pass
    estado = pl.estado_sondeo(proyecto, sondeo)
    columnas = st.columns(len(pl.ETIQUETAS_ETAPA))
    for columna, (clave, etiqueta) in zip(columnas, pl.ETIQUETAS_ETAPA.items()):
        columna.markdown(f"{'✅' if estado.get(clave) else '⬜'} **{etiqueta}**")
    return None


def panel_descarga_proyecto(usuario: Optional[str] = None) -> None:
    """Barra lateral: estado del proyecto y botón de descarga ``.isev``."""
    import acciones_app as acc

    proyecto = obtener_proyecto()
    if proyecto is None:
        return None
    else:
        pass
    with st.sidebar:
        st.divider()
        st.markdown("### 💾 Proyecto")
        st.caption(f"**{proyecto.nombre}**")
        if hay_cambios_sin_descargar():
            st.warning("Hay trabajo sin descargar. Al cerrar la sesión se perderá si no "
                       "guarda el archivo `.isev`.", icon="⚠️")
        else:
            pass
        resultado = acc.accion_empaquetar_proyecto(proyecto, usuario=usuario)
        if resultado:
            if st.download_button("⬇️ Descargar proyecto (.isev)", resultado.datos["contenido"],
                                  file_name=resultado.datos["nombre"], mime="application/octet-stream",
                                  use_container_width=True, key="btn_descargar_proyecto"):
                marcar_cambios_sin_descargar(False)
            else:
                pass
            st.caption(f"{resultado.metricas['tamano_mb']} MB · {resultado.metricas['n_sondeos']} sondeo(s)")
        else:
            st.caption("Aún no hay nada que descargar.")
    return None


def ejecutar_con_progreso(funcion, texto_inicial: str = "Procesando…"):
    """Ejecuta una acción larga con barra de progreso y token de parada.

    ``funcion(callback, token)`` debe aceptar el callback
    ``(texto, fraccion, resultado)`` y el token de cancelación.
    """
    barra = st.progress(0.0, text=texto_inicial)
    reiniciar_token_cancelacion()
    token = obtener_token_cancelacion()
    marcar_ejecutando(True)

    def _cb(texto, fraccion, resultado):
        barra.progress(min(max(float(fraccion), 0.0), 1.0), text=texto)
        registrar_resultado(resultado)

    try:
        resultado = funcion(_cb, token)
    finally:
        marcar_ejecutando(False)
        barra.empty()
    limpiar_cache_datos()
    return resultado, token
