"""
acciones_app.py
==================================================================
**Acciones** de la aplicación: la lógica de cada botón.

Cada función devuelve un :class:`pipeline_sev.ResultadoEtapa`
(booleano-evaluable) y no importa Streamlit, de modo que todo lo que
hace un botón se prueba desde código (ver ``test_app.py``).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

import _ruta_src  # noqa: F401
import almacenamiento_sev as alm
import pipeline_sev as pl
from almacenamiento_sev import Proyecto
from pipeline_sev import ConfigPipeline, ResultadoEtapa, TokenCancelacion

logger = logging.getLogger("sev.acciones")


# ----------------------------------------------------------------------
# Proyecto
# ----------------------------------------------------------------------


def accion_crear_proyecto(ruta_base, nombre: str, cliente: Optional[str] = None,
                          anio: Optional[str] = None) -> ResultadoEtapa:
    if not str(nombre).strip():
        return ResultadoEtapa(False, "proyecto", "El nombre del proyecto no puede estar vacío.")
    else:
        pass
    try:
        proyecto = alm.crear_proyecto(ruta_base, nombre, cliente=cliente, anio=anio)
    except OSError as exc:
        return ResultadoEtapa(False, "proyecto", f"No se pudo crear el proyecto: {exc}")
    return ResultadoEtapa(True, "proyecto", f"Proyecto '{proyecto.nombre}' listo.",
                          datos={"proyecto": proyecto})


def accion_abrir_proyecto(ruta) -> ResultadoEtapa:
    try:
        proyecto = alm.abrir_proyecto(ruta)
    except alm.ProyectoNoEncontradoError as exc:
        return ResultadoEtapa(False, "proyecto", str(exc))
    return ResultadoEtapa(True, "proyecto", f"Proyecto '{proyecto.nombre}' abierto.",
                          datos={"proyecto": proyecto})


def guardar_archivo_subido(proyecto: Proyecto, nombre_archivo: str, contenido: bytes) -> Path:
    """Conserva el archivo subido dentro del proyecto (``_entradas``)."""
    carpeta = proyecto.ruta / alm.CARPETA_ENTRADAS
    carpeta.mkdir(exist_ok=True)
    destino = carpeta / Path(nombre_archivo).name
    destino.write_bytes(contenido)
    return destino


def accion_eliminar_sondeo(proyecto: Proyecto, sondeo: str) -> ResultadoEtapa:
    try:
        alm.eliminar_sondeo(proyecto, sondeo)
    except OSError as exc:
        return ResultadoEtapa(False, "proyecto", f"No se pudo eliminar: {exc}")
    return ResultadoEtapa(True, "proyecto", f"Sondeo '{sondeo}' eliminado del proyecto.")


# ----------------------------------------------------------------------
# Etapas
# ----------------------------------------------------------------------


def accion_cargar_datos(proyecto: Proyecto, ruta_archivo, config: Optional[ConfigPipeline] = None) -> ResultadoEtapa:
    return pl.ejecutar_lectura(proyecto, ruta_archivo, config, guardar=True)


def accion_cargar_coordenadas(proyecto: Proyecto, ruta_o_bytes) -> ResultadoEtapa:
    return pl.cargar_coordenadas(proyecto, ruta_o_bytes)


def accion_editar_coordenadas(proyecto: Proyecto, tabla: pd.DataFrame) -> ResultadoEtapa:
    validas = tabla.dropna(subset=["X", "Y"])
    n = alm.fijar_coordenadas(proyecto, validas)
    return ResultadoEtapa(True, "coordenadas", f"Coordenadas actualizadas en {n} sondeo(s).")


def accion_ejecutar_archivo(proyecto: Proyecto, ruta_archivo, config: ConfigPipeline,
                            hasta_etapa: str = "parametros", callback_progreso: Optional[Callable] = None,
                            token_cancelacion: Optional[TokenCancelacion] = None) -> ResultadoEtapa:
    """Botón «Correr pipeline»: lectura y todas las etapas de todos los sondeos."""
    salida = pl.ejecutar_pipeline_archivo(proyecto, ruta_archivo, config, hasta_etapa,
                                          callback_progreso, token_cancelacion)
    lectura = salida["lectura"]
    if not lectura:
        return lectura
    else:
        pass
    fallidos = []
    advertencias = list(lectura.advertencias)
    for nombre, etapas in salida["sondeos"].items():
        for etapa, resultado in etapas.items():
            advertencias += [f"{nombre}/{etapa}: {a}" for a in resultado.advertencias]
            if not resultado:
                fallidos.append(f"{nombre} ({etapa}): {resultado.mensaje}")
            else:
                pass
    n = len(salida["sondeos"])
    if fallidos:
        return ResultadoEtapa(False, "pipeline", f"{n - len(fallidos)} de {n} sondeos completos. Fallos: "
                              + " | ".join(fallidos), advertencias=advertencias,
                              datos={"sondeos": lectura.datos["sondeos"]})
    else:
        return ResultadoEtapa(True, "pipeline", f"{n} sondeo(s) procesado(s) hasta «{hasta_etapa}».",
                              advertencias=advertencias, datos={"sondeos": lectura.datos["sondeos"]})


def accion_reprocesar(proyecto: Proyecto, sondeos: list[str], config: ConfigPipeline,
                      desde_etapa: str = "promedio", hasta_etapa: str = "parametros",
                      callback_progreso: Optional[Callable] = None,
                      token_cancelacion: Optional[TokenCancelacion] = None) -> ResultadoEtapa:
    """Rehace etapas de uno o varios sondeos ya cargados."""
    fallidos, advertencias = [], []
    for i, nombre in enumerate(sondeos):
        def _cb(etapa, indice, total, resultado, _i=i, _n=nombre):
            if callback_progreso is not None:
                callback_progreso(f"{_n} · {pl.ETIQUETAS_ETAPA[etapa]}", (_i + indice / total) / len(sondeos), resultado)
            else:
                pass

        etapas = pl.ejecutar_pipeline_sondeo(proyecto, nombre, config, desde_etapa, hasta_etapa, _cb, token_cancelacion)
        for etapa, resultado in etapas.items():
            advertencias += [f"{nombre}/{etapa}: {a}" for a in resultado.advertencias]
            if not resultado:
                fallidos.append(f"{nombre} ({etapa}): {resultado.mensaje}")
            else:
                pass
        if token_cancelacion is not None and token_cancelacion.cancelado:
            break
        else:
            pass
    if fallidos:
        return ResultadoEtapa(False, "pipeline", " | ".join(fallidos), advertencias=advertencias)
    else:
        return ResultadoEtapa(True, "pipeline", f"{len(sondeos)} sondeo(s) reprocesado(s).",
                              advertencias=advertencias)


def accion_unidades(proyecto: Proyecto, n_max: int = 6) -> ResultadoEtapa:
    return pl.ejecutar_unidades(proyecto, n_max=n_max)


def accion_perfil(proyecto: Proyecto, nombre: str, vertices, buffer: float, lci: bool,
                  lam: float, acoplamiento: float, n_capas: int = 25) -> ResultadoEtapa:
    if not str(nombre).strip():
        return ResultadoEtapa(False, "perfil", "Dé un nombre al perfil.")
    else:
        pass
    return pl.ejecutar_perfil(proyecto, alm.sanear_id(nombre), vertices, buffer=buffer, lci=lci,
                              lam=lam, acoplamiento=acoplamiento, n_capas=n_capas)


# ----------------------------------------------------------------------
# Paquete
# ----------------------------------------------------------------------


def accion_empaquetar_proyecto(proyecto: Proyecto, usuario: Optional[str] = None) -> ResultadoEtapa:
    import paquete_sev as pk

    try:
        sondeos = alm.listar_sondeos(proyecto)["sondeo"].astype(str).tolist()
        contenido = pk.empaquetar_proyecto(proyecto, sondeos, usuario=usuario)
    except (OSError, ValueError) as exc:
        return ResultadoEtapa(False, "paquete", f"No se pudo empaquetar el proyecto: {exc}")
    return ResultadoEtapa(
        True, "paquete", f"Proyecto listo para descargar ({len(contenido) / 1024 / 1024:.2f} MB).",
        datos={"contenido": contenido, "nombre": pk.nombre_paquete(proyecto.nombre)},
        metricas={"n_sondeos": len(sondeos), "tamano_mb": round(len(contenido) / 1024 / 1024, 3)},
    )


def accion_restaurar_proyecto(contenido: bytes, espacio) -> ResultadoEtapa:
    import paquete_sev as pk

    try:
        manifiesto = pk.leer_manifiesto(contenido)
        ruta = pk.restaurar_proyecto(contenido, espacio)
        proyecto = alm.abrir_proyecto(ruta)
    except (pk.PaqueteError, alm.ProyectoNoEncontradoError) as exc:
        return ResultadoEtapa(False, "paquete", str(exc))
    n = len(alm.listar_sondeos(proyecto))
    return ResultadoEtapa(True, "paquete",
                          f"Proyecto '{proyecto.nombre}' restaurado con {n} sondeo(s) "
                          f"(generado el {manifiesto.get('generado_en', '¿?')}).",
                          datos={"proyecto": proyecto})


# ----------------------------------------------------------------------
# Consultas
# ----------------------------------------------------------------------


def resumen_proyecto(proyecto: Proyecto) -> dict[str, Any]:
    sondeos = alm.listar_sondeos(proyecto)
    n_completos = 0
    for nombre in sondeos["sondeo"].astype(str):
        if all(pl.estado_sondeo(proyecto, nombre).values()):
            n_completos += 1
        else:
            pass
    return {
        "nombre": proyecto.nombre, "cliente": proyecto.cliente,
        "n_sondeos": len(sondeos), "n_completos": n_completos,
        "n_con_ip": int(sondeos["tiene_ip"].sum()) if len(sondeos) else 0,
        "n_con_coordenadas": int(sondeos["x"].notna().sum()) if len(sondeos) else 0,
        "tamano_mb": alm.tamano_proyecto(proyecto)["total_mb"],
    }


def metricas_sondeo(proyecto: Proyecto, sondeo: str) -> dict[str, Any]:
    df = alm.obtener_metricas(proyecto, sondeo)
    salida: dict[str, Any] = {}
    for _, fila in df.iterrows():
        clave = f"{fila['etapa']}.{fila['clave']}"
        salida[clave] = float(fila["valor_num"]) if pd.notna(fila["valor_num"]) else fila["valor_texto"]
    return salida
