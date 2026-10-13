"""
pipeline_sev.py
==================================================================
Orquestador del pipeline SEV/IP: funciones puras y probables desde
código, sin Streamlit (mismo contrato que TlalliRes2D).

Etapas por sondeo
--------------------
==============  ==========================================================
Etapa           Qué hace
==============  ==========================================================
``crudo``       Lectura, detección de arreglo, recálculo de K, R y ρa.
``promedio``    Corrección de empalmes y media geométrica de direcciones.
``procesado``   Detección y tratamiento de anómalos, modelo de error.
``inversion``   Zohdy → Occam (pyGIMLi) → Dar Zarrouk + BIC → bloques
                (pyGIMLi) → IP.
``parametros``  S, T, ρL, ρT, λ, ρm y corrosividad.
==============  ==========================================================

Además, a nivel de campaña: unidades geoeléctricas (GMM) y perfiles 2D
(sección interpolada y LCI).

Cada ``ejecutar_*`` devuelve un :class:`ResultadoEtapa`
booleano-evaluable.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

import rutas_sev  # noqa: F401
import almacenamiento_sev as alm
from almacenamiento_sev import Proyecto
from inversion_sev import ConfigInversion
from procesamiento_sev import ConfigPreproceso

logger = logging.getLogger("sev.pipeline")


class CancelacionSolicitada(Exception):
    """Se solicitó detener la ejecución."""


class TokenCancelacion:
    """Señal de cancelación cooperativa (ver TlalliRes2D).

    El pipeline la consulta entre etapas, entre sondeos y entre pruebas
    de Optuna, así que la parada surte efecto en segundos.
    """

    def __init__(self) -> None:
        self._evento = threading.Event()

    def solicitar_cancelacion(self) -> None:
        self._evento.set()

    def reiniciar(self) -> None:
        self._evento.clear()

    @property
    def cancelado(self) -> bool:
        return self._evento.is_set()

    def verificar(self) -> None:
        if self._evento.is_set():
            raise CancelacionSolicitada("Ejecución detenida por el usuario.")
        else:
            pass


@dataclass
class ResultadoEtapa:
    """Resultado de una etapa (``bool(resultado)`` = éxito)."""

    exito: bool
    etapa: str
    mensaje: str = ""
    datos: dict[str, Any] = field(default_factory=dict)
    metricas: dict[str, Any] = field(default_factory=dict)
    duracion_s: float = 0.0
    advertencias: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.exito


@dataclass
class ConfigPipeline:
    """Parámetros de ejecución de todo el pipeline.

    Attributes
    ----------
    preproceso : ConfigPreproceso
    inversion : ConfigInversion
    arreglo_forzado : str, optional
        Código de arreglo (``SCH``, ``WE``, ``DD``, ``PD``, ``PP``) o
        ``None`` para detección automática.
    autoguardado : bool
    prof_parametros : float, optional
        Profundidad de cálculo de Dar Zarrouk (``None`` = DOI).
    prof_corrosion : float
        Profundidad del suelo evaluado para corrosividad.
    umbrales_corrosion : tuple
        Umbrales de clase de corrosividad (Ω·m).
    """

    preproceso: ConfigPreproceso = field(default_factory=ConfigPreproceso)
    inversion: ConfigInversion = field(default_factory=ConfigInversion)
    arreglo_forzado: Optional[str] = None
    autoguardado: bool = True
    prof_parametros: Optional[float] = None
    prof_corrosion: float = 3.0
    umbrales_corrosion: tuple = (5.0, 10.0, 20.0, 100.0)


ORDEN_ETAPAS = ("crudo", "promedio", "procesado", "inversion", "parametros")

ETIQUETAS_ETAPA = {
    "crudo": "1 · Lectura",
    "promedio": "2 · Promedio",
    "procesado": "3 · Preproceso",
    "inversion": "4 · Inversión",
    "parametros": "5 · Parámetros",
}


def _cronometrar(funcion: Callable[[], ResultadoEtapa], etapa: str) -> ResultadoEtapa:
    """Ejecuta una etapa capturando tiempo, advertencias y errores."""
    inicio = time.time()
    with warnings.catch_warnings(record=True) as registro:
        warnings.simplefilter("always")
        try:
            resultado = funcion()
        except CancelacionSolicitada as exc:
            resultado = ResultadoEtapa(exito=False, etapa=etapa, mensaje=str(exc))
        except Exception as exc:  # la interfaz nunca debe caerse por una etapa
            logger.exception("Etapa %s falló", etapa)
            resultado = ResultadoEtapa(exito=False, etapa=etapa,
                                       mensaje=f"{type(exc).__name__}: {exc}")
    resultado.duracion_s = time.time() - inicio
    ruidosas = ("Degrees of freedom", "Mean of empty slice", "invalid value", "divide by zero")
    resultado.advertencias += [
        str(w.message) for w in registro
        if not any(r in str(w.message) for r in ruidosas)
    ][:5]
    return resultado


# ======================================================================
# Etapa 1: lectura
# ======================================================================


def ejecutar_lectura(
    proyecto: Proyecto, ruta_archivo: Union[str, Path], config: Optional[ConfigPipeline] = None,
    guardar: bool = True, sondeos: Optional[list[str]] = None,
) -> ResultadoEtapa:
    """Lee un archivo (uno o varios sondeos) y guarda la etapa ``crudo``.

    Returns
    -------
    ResultadoEtapa
        ``datos['sondeos']``: lista de sondeos leídos.
    """
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import lectura_sev as lec
        import procesamiento_sev as pr

        tabla = lec.leer_sev(ruta_archivo, nombre_sondeo_defecto=Path(ruta_archivo).stem)
        nombres = [s for s in tabla["Sondeo"].unique() if sondeos is None or s in sondeos]
        advertencias: list[str] = []
        detecciones = {}
        for nombre in nombres:
            sub = tabla[tabla["Sondeo"] == nombre]
            crudo, deteccion, metricas, adv = pr.preparar_crudo(sub, arreglo_forzado=cfg.arreglo_forzado)
            detecciones[nombre] = deteccion.mensaje
            advertencias += [f"{nombre}: {a}" for a in adv]
            if guardar:
                alm.registrar_sondeo(
                    proyecto, nombre, archivo_origen=Path(ruta_archivo).name,
                    arreglo=deteccion.arreglo, convencion=deteccion.convencion,
                    n_registros=metricas["n_registros"], n_direcciones=metricas["n_direcciones"],
                    ab2_max=metricas["ab2_max"], tiene_ip=metricas["tiene_ip"],
                )
                alm.guardar_tabla(proyecto, crudo, "crudo", "lectura", nombre, metricas=metricas)
            else:
                pass
        return ResultadoEtapa(
            exito=True, etapa="crudo",
            mensaje=f"{len(nombres)} sondeo(s) leído(s): {', '.join(map(str, nombres))}.",
            datos={"sondeos": [str(n) for n in nombres], "detecciones": detecciones},
            metricas={"n_sondeos": len(nombres), "n_registros": int(len(tabla))},
            advertencias=advertencias,
        )

    return _cronometrar(_ejecutar, "crudo")


def cargar_coordenadas(proyecto: Proyecto, ruta: Union[str, Path, bytes]) -> ResultadoEtapa:
    """Lee un archivo ``Sondeo, X, Y, Z`` y lo asigna a los sondeos."""

    def _ejecutar() -> ResultadoEtapa:
        import lectura_sev as lec

        coords = lec.leer_coordenadas(ruta)
        n = alm.fijar_coordenadas(proyecto, coords)
        faltan = sorted(set(coords["Sondeo"]) - set(alm.listar_sondeos(proyecto)["sondeo"]))
        adv = [f"Sondeos del archivo de coordenadas que no están en el proyecto: {faltan}"] if faltan else []
        return ResultadoEtapa(True, "coordenadas", f"Coordenadas asignadas a {n} sondeo(s).",
                              advertencias=adv)

    return _cronometrar(_ejecutar, "coordenadas")


# ======================================================================
# Etapas 2–5 de un sondeo
# ======================================================================


def _requerir(proyecto: Proyecto, etapa: str, tabla: str, sondeo: str) -> pd.DataFrame:
    df = alm.cargar_tabla(proyecto, etapa, tabla, sondeo)
    if df is None:
        raise ValueError(f"El sondeo '{sondeo}' no tiene la etapa '{etapa}': ejecútela antes.")
    else:
        return df


def ejecutar_promedio(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                      guardar: bool = True) -> ResultadoEtapa:
    """Empalmes + media geométrica de direcciones."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import procesamiento_sev as pr

        crudo = _requerir(proyecto, "crudo", "lectura", sondeo)
        pp = cfg.preproceso
        corregido, informe = pr.corregir_empalmes(crudo, umbral=pp.umbral_empalme, activar=pp.corregir_empalmes,
                                                  arreglo=arreglo_sondeo(proyecto, sondeo))
        promedio, metricas, adv = pr.promediar_direcciones(corregido)
        metricas["n_empalmes_corregidos"] = int(sum(1 for i in informe if i["aplicado"]))
        if guardar:
            alm.guardar_tabla(proyecto, corregido, "crudo", "lectura", sondeo)
            alm.guardar_tabla(proyecto, pd.DataFrame(informe if informe else [{"Direccion": "", "factor": 1.0}]),
                              "crudo", "empalmes", sondeo)
            alm.guardar_tabla(proyecto, promedio, "promedio", "promedio", sondeo, metricas=metricas)
        else:
            pass
        mensaje = (f"Curva promedio de {metricas['n_direcciones']} dirección(es), "
                   f"{metricas['n_puntos']} puntos; {metricas['n_empalmes_corregidos']} empalme(s) corregido(s).")
        return ResultadoEtapa(True, "promedio", mensaje, metricas=metricas, advertencias=adv,
                              datos={"promedio": promedio})

    return _cronometrar(_ejecutar, "promedio")


def ejecutar_procesamiento(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                           guardar: bool = True) -> ResultadoEtapa:
    """Detección y tratamiento de anómalos."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import procesamiento_sev as pr

        promedio = _requerir(proyecto, "promedio", "promedio", sondeo)
        arreglo = arreglo_sondeo(proyecto, sondeo)
        procesado, metricas, adv = pr.preprocesar(promedio, cfg.preproceso, arreglo=arreglo)
        metricas["config"] = json.dumps(cfg.preproceso.as_dict(), default=str)
        if guardar:
            alm.guardar_tabla(proyecto, procesado, "procesado", "procesado", sondeo, metricas=metricas)
        else:
            pass
        if metricas["requirio_preproceso"]:
            mensaje = (f"{metricas['n_anomalos']} dato(s) de resistividad y "
                       f"{metricas['n_ip_anomalos']} de IP tratados ({metricas['tratamiento']}).")
        else:
            mensaje = "No requirió preprocesamiento: la curva preprocesada es la observada."
        return ResultadoEtapa(True, "procesado", mensaje, metricas=metricas, advertencias=adv,
                              datos={"procesado": procesado})

    return _cronometrar(_ejecutar, "procesado")


def ejecutar_inversion(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                       guardar: bool = True, token_cancelacion: Optional[TokenCancelacion] = None) -> ResultadoEtapa:
    """Inversión completa de un sondeo."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import inversion_sev as inv

        procesado = _requerir(proyecto, "procesado", "procesado", sondeo)
        verificar = token_cancelacion.verificar if token_cancelacion is not None else None
        res = inv.invertir_sondeo(procesado, cfg.inversion, verificar_cancelacion=verificar)
        metricas = dict(res.metricas)
        metricas["config"] = json.dumps(cfg.inversion.as_dict(), default=str)
        if guardar:
            for tabla, df in (("multicapa", res.multicapa), ("simplificado", res.simplificado),
                              ("ajuste", res.ajuste), ("seleccion", res.seleccion),
                              ("lambdas", res.lambdas), ("zohdy", res.zohdy)):
                if df is not None:
                    alm.guardar_tabla(proyecto, df, "inversion", tabla, sondeo)
                else:
                    pass
            alm.registrar_metricas(proyecto, sondeo, "inversion", metricas)
        else:
            pass
        m = res.metricas
        mensaje = (f"Multicapa χ²={m['multicapa_chi2']:.2f} (λ={m['multicapa_lambda']:.1f}); "
                   f"simplificado {m['simplificado_n_capas']} capas χ²={m['simplificado_chi2']:.2f}; "
                   f"DOI≈{m['doi']:.1f} m.")
        return ResultadoEtapa(True, "inversion", mensaje, metricas=metricas,
                              advertencias=res.advertencias, datos={"resultado": res})

    return _cronometrar(_ejecutar, "inversion")


def ejecutar_parametros(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                        guardar: bool = True) -> ResultadoEtapa:
    """Parámetros de Dar Zarrouk y corrosividad."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import parametros_sev as pa

        multicapa = _requerir(proyecto, "inversion", "multicapa", sondeo)
        simplificado = _requerir(proyecto, "inversion", "simplificado", sondeo)
        doi = metrica(proyecto, sondeo, "inversion", "doi")
        tabla, resumen = pa.parametros_sondeo(
            multicapa, simplificado, profundidad=cfg.prof_parametros, doi=doi,
            prof_corrosion=cfg.prof_corrosion, umbrales_corrosion=cfg.umbrales_corrosion,
        )
        if guardar:
            alm.guardar_tabla(proyecto, tabla, "parametros", "parametros", sondeo, metricas=resumen)
        else:
            pass
        adv = []
        if abs(resumen["diferencia_S_pct"]) > 25 or abs(resumen["diferencia_T_pct"]) > 25:
            adv.append(
                f"S o T difieren más de un 25 % entre el modelo multicapa y el simplificado "
                f"(ΔS={resumen['diferencia_S_pct']:.0f} %, ΔT={resumen['diferencia_T_pct']:.0f} %): "
                "el simplificado puede estar sobre-simplificado."
            )
        else:
            pass
        mensaje = (f"S={resumen['S']:.3g} S, T={resumen['T']:.3g} Ω·m², ρm={resumen['rho_m']:.3g} Ω·m "
                   f"hasta {resumen['profundidad']:.1f} m; corrosividad {resumen['corrosividad'].lower()}.")
        return ResultadoEtapa(True, "parametros", mensaje, metricas=resumen, advertencias=adv,
                              datos={"tabla": tabla})

    return _cronometrar(_ejecutar, "parametros")


# ======================================================================
# Campaña
# ======================================================================


def ejecutar_unidades(proyecto: Proyecto, n_max: int = 6) -> ResultadoEtapa:
    """Unidades geoeléctricas de toda la campaña (mezcla gaussiana)."""

    def _ejecutar() -> ResultadoEtapa:
        import parametros_sev as pa

        capas = alm.consultar_multisondeo(proyecto, "inversion", "simplificado")
        if len(capas) == 0:
            raise ValueError("Ningún sondeo tiene modelo simplificado todavía.")
        else:
            pass
        capas = capas.rename(columns={"SEV": "Sondeo"})
        tabla, resumen = pa.unidades_geoelectricas(capas, n_max=n_max)
        alm.guardar_tabla(proyecto, tabla, "parametros", "unidades", alm.CAMPANA)
        alm.guardar_tabla(proyecto, resumen, "parametros", "resumen_unidades", alm.CAMPANA)
        return ResultadoEtapa(True, "unidades",
                              f"{tabla['unidad'].nunique()} unidad(es) geoeléctrica(s) en "
                              f"{tabla['Sondeo'].nunique()} sondeo(s).", datos={"resumen": resumen})

    return _cronometrar(_ejecutar, "unidades")


def ejecutar_perfil(
    proyecto: Proyecto, nombre: str, vertices: list[list[float]], buffer: float = 100.0,
    lci: bool = True, lam: float = 20.0, acoplamiento: float = 1.0, n_capas: int = 25,
    token_cancelacion: Optional[TokenCancelacion] = None,
) -> ResultadoEtapa:
    """Proyecta los sondeos sobre la ruta y, si se pide, invierte la LCI."""

    def _ejecutar() -> ResultadoEtapa:
        import perfil_sev as pf

        coords = alm.coordenadas(proyecto)
        if len(coords) < 2:
            raise ValueError("Hacen falta coordenadas de al menos dos sondeos (pantalla 1).")
        else:
            pass
        proyectados = pf.proyectar(coords, vertices, buffer=buffer)
        dentro = proyectados[proyectados["dentro"]].reset_index(drop=True)
        alm.guardar_tabla(proyecto, proyectados, "perfiles", "perfil_sondeos", nombre)
        alm.guardar_tabla(proyecto, pd.DataFrame(vertices, columns=["X", "Y"]), "perfiles", "perfil_ruta", nombre)
        if len(dentro) < 2:
            raise ValueError(f"Solo {len(dentro)} sondeo(s) caen dentro del buffer de {buffer} m.")
        else:
            pass
        mensaje = f"{len(dentro)} sondeos proyectados sobre la ruta '{nombre}'."
        metricas: dict[str, Any] = {"n_sondeos": len(dentro), "buffer": buffer}
        if lci:
            if token_cancelacion is not None:
                token_cancelacion.verificar()
            else:
                pass
            datos, iniciales = {}, {}
            for s in dentro["Sondeo"].astype(str):
                p = alm.cargar_tabla(proyecto, "procesado", "procesado", s)
                m = alm.cargar_tabla(proyecto, "inversion", "multicapa", s)
                if p is not None:
                    datos[s] = p
                else:
                    pass
                if m is not None:
                    iniciales[s] = m
                else:
                    pass
            res = pf.invertir_lci(dentro, datos, iniciales, lam=lam, acoplamiento=acoplamiento, n_capas=n_capas)
            alm.guardar_tabla(proyecto, res["modelo"], "perfiles", "perfil_lci", nombre)
            alm.guardar_tabla(proyecto, res["ajuste"], "perfiles", "perfil_ajuste", nombre)
            metricas.update({"lci_chi2": res["chi2"], "lci_lam": lam, "lci_acoplamiento": acoplamiento})
            mensaje += f" LCI: χ²={res['chi2']:.2f}."
        else:
            alm.eliminar_tabla(proyecto, "perfiles", "perfil_lci", nombre)
        alm.registrar_metricas(proyecto, nombre, "perfiles", metricas)
        return ResultadoEtapa(True, "perfil", mensaje, metricas=metricas)

    return _cronometrar(_ejecutar, "perfil")


def listar_perfiles(proyecto: Proyecto) -> list[str]:
    carpeta = proyecto.ruta_etapa("perfiles")
    return sorted(p.name.split("=", 1)[1] for p in carpeta.glob("SEV=*") if (p / "perfil_ruta.parquet").exists())


# ======================================================================
# Pipeline
# ======================================================================


def ejecutar_pipeline_sondeo(
    proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
    desde_etapa: str = "promedio", hasta_etapa: str = "parametros",
    callback_progreso: Optional[Callable] = None, token_cancelacion: Optional[TokenCancelacion] = None,
) -> dict[str, ResultadoEtapa]:
    """Ejecuta las etapas ``desde..hasta`` (posteriores a la lectura)."""
    cfg = config or ConfigPipeline()
    orden = list(ORDEN_ETAPAS)
    a_ejecutar = orden[orden.index(desde_etapa): orden.index(hasta_etapa) + 1]
    a_ejecutar = [e for e in a_ejecutar if e != "crudo"]
    resultados: dict[str, ResultadoEtapa] = {}
    for indice, etapa in enumerate(a_ejecutar, start=1):
        if token_cancelacion is not None and token_cancelacion.cancelado:
            resultados[etapa] = ResultadoEtapa(False, etapa, "Ejecución detenida por el usuario.")
            break
        else:
            pass
        if etapa == "promedio":
            resultado = ejecutar_promedio(proyecto, sondeo, cfg, guardar=cfg.autoguardado)
        else:
            if etapa == "procesado":
                resultado = ejecutar_procesamiento(proyecto, sondeo, cfg, guardar=cfg.autoguardado)
            else:
                if etapa == "inversion":
                    resultado = ejecutar_inversion(proyecto, sondeo, cfg, guardar=cfg.autoguardado,
                                                   token_cancelacion=token_cancelacion)
                else:
                    resultado = ejecutar_parametros(proyecto, sondeo, cfg, guardar=cfg.autoguardado)
        resultado.mensaje = f"[{sondeo}] {resultado.mensaje}"
        resultados[etapa] = resultado
        if callback_progreso is not None:
            callback_progreso(etapa, indice, len(a_ejecutar), resultado)
        else:
            pass
        if not resultado:
            break
        else:
            pass
    return resultados


def ejecutar_pipeline_archivo(
    proyecto: Proyecto, ruta_archivo: Union[str, Path], config: Optional[ConfigPipeline] = None,
    hasta_etapa: str = "parametros", callback_progreso: Optional[Callable] = None,
    token_cancelacion: Optional[TokenCancelacion] = None,
) -> dict[str, Any]:
    """Lee un archivo y procesa todos sus sondeos hasta ``hasta_etapa``.

    ``callback_progreso(texto, fraccion, resultado)`` se invoca tras cada
    etapa de cada sondeo.

    Returns
    -------
    dict
        ``{'lectura': ResultadoEtapa, 'sondeos': {nombre: {etapa: ResultadoEtapa}}}``
    """
    cfg = config or ConfigPipeline()
    lectura = ejecutar_lectura(proyecto, ruta_archivo, cfg, guardar=True)
    salida: dict[str, Any] = {"lectura": lectura, "sondeos": {}}
    if callback_progreso is not None:
        callback_progreso("Lectura", 0.05, lectura)
    else:
        pass
    if not lectura or hasta_etapa == "crudo":
        return salida
    else:
        pass

    nombres = lectura.datos["sondeos"]
    for i, nombre in enumerate(nombres):
        def _cb(etapa, indice, total, resultado, _i=i):
            if callback_progreso is not None:
                fraccion = 0.05 + 0.95 * (_i + indice / total) / len(nombres)
                callback_progreso(f"{nombre} · {ETIQUETAS_ETAPA[etapa]}", fraccion, resultado)
            else:
                pass

        salida["sondeos"][nombre] = ejecutar_pipeline_sondeo(
            proyecto, nombre, cfg, "promedio", hasta_etapa, _cb, token_cancelacion)
        if token_cancelacion is not None and token_cancelacion.cancelado:
            break
        else:
            pass
    return salida


# ======================================================================
# Consultas
# ======================================================================


def estado_sondeo(proyecto: Proyecto, sondeo: str) -> dict[str, bool]:
    return {
        "crudo": alm.existe_tabla(proyecto, "crudo", "lectura", sondeo),
        "promedio": alm.existe_tabla(proyecto, "promedio", "promedio", sondeo),
        "procesado": alm.existe_tabla(proyecto, "procesado", "procesado", sondeo),
        "inversion": alm.existe_tabla(proyecto, "inversion", "simplificado", sondeo),
        "parametros": alm.existe_tabla(proyecto, "parametros", "parametros", sondeo),
    }


def metrica(proyecto: Proyecto, sondeo: str, etapa: str, clave: str) -> Optional[float]:
    df = alm.obtener_metricas(proyecto, sondeo, etapa)
    fila = df[df["clave"] == clave]
    if len(fila) == 0 or pd.isna(fila["valor_num"].iloc[0]):
        return None
    else:
        return float(fila["valor_num"].iloc[0])


def arreglo_sondeo(proyecto: Proyecto, sondeo: str) -> str:
    df = alm.listar_sondeos(proyecto)
    fila = df[df["sondeo"] == str(sondeo)]
    if len(fila) == 0 or not fila["arreglo"].iloc[0]:
        return "SCH"
    else:
        return str(fila["arreglo"].iloc[0])
