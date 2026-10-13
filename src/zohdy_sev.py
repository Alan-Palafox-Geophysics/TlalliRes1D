"""
zohdy_sev.py
==================================================================
Inversión automática de Zohdy (1989) para curvas de SEV de cualquier
arreglo.

Idea del método
------------------
Zohdy construye un modelo con **tantas capas como datos**: cada punto
de la curva aporta una capa cuya resistividad inicial es la aparente
medida y cuya base está a una profundidad proporcional a la apertura.
Después hace dos cosas:

1. **Búsqueda de profundidad**: escala todas las profundidades por un
   mismo factor hasta minimizar el RMS entre curva observada y
   calculada. Aquí se hace con una búsqueda en rejilla logarítmica
   (0.3–3) en vez de la reducción sucesiva del 10 % original, que puede
   detenerse en un mínimo local.
2. **Corrección de resistividades**: multiplica cada capa por el
   cociente observado/calculado del dato que la originó, iterando
   mientras el RMS disminuya.

Generalización a arreglos no Schlumberger
--------------------------------------------
La abscisa de cada dato no es AB/2 sino la **profundidad mediana de
investigación de Edwards** de su geometría real
(:func:`geometria_sev.profundidad_mediana`). En Schlumberger equivale a
0.38·AB/2 —la búsqueda de factor absorbe la diferencia— y en
dipolo-dipolo, polo-dipolo y polo-polo da una abscisa físicamente
comparable, que es lo que el método necesita.

El resultado es un modelo inicial excelente para la inversión
regularizada: ya reproduce la curva con un RMS bajo y no presupone el
número de capas.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import forward_sev as fw

logger = logging.getLogger("sev.zohdy")


@dataclass
class ResultadoZohdy:
    """Modelo de Zohdy y su historial.

    ``n_ab`` es el denominador de la profundidad z = AB/n cuando se usa
    :func:`invertir_zohdy_ab` (None con el factor sobre Edwards), y
    ``busqueda`` la tabla (n, RMS) de la búsqueda.
    """

    espesores: np.ndarray
    resistividades: np.ndarray
    respuesta: np.ndarray
    factor_profundidad: float
    rms: float
    historial_rms: list[float] = field(default_factory=list)
    n_ab: float | None = None
    busqueda: list[tuple[float, float]] = field(default_factory=list)


def _corregir_resistividades(geom, rhoa, espesores, max_iter: int = 30):
    """Corrección iterativa de Zohdy: ρ_i ← ρ_i·(ρa_obs/ρa_calc) del dato
    que originó la capa, mientras el RMS disminuya."""
    rho = rhoa.copy()
    calculada = fw.respuesta(geom, espesores, rho)
    historial = [_rms_log(rhoa, calculada)]
    mejor_rho, mejor_calc = rho.copy(), calculada.copy()
    for _ in range(max_iter):
        cociente = np.clip(rhoa / calculada, 0.2, 5.0)
        rho = np.clip(rho * cociente, 1e-3, 1e7)
        calculada = fw.respuesta(geom, espesores, rho)
        rms = _rms_log(rhoa, calculada)
        historial.append(rms)
        if rms < min(historial[:-1]):
            mejor_rho, mejor_calc = rho.copy(), calculada.copy()
        else:
            pass
        if len(historial) > 3 and historial[-3] - rms < 0.01:
            break
        else:
            pass
    return mejor_rho, mejor_calc, historial


def invertir_zohdy_ab(
    geom: fw.GeometriaSondeo,
    rhoa: np.ndarray,
    longitud_ab: np.ndarray,
    n_min: float = 5.0,
    n_max: float = 8.0,
    paso: float = 0.125,
    max_iter: int = 30,
) -> ResultadoZohdy:
    """Zohdy con profundidad de atribución z = AB/n.

    Para cada n de ``n_min`` a ``n_max`` (paso ``paso``) se asigna a cada
    dato la base de su capa en AB/n, se corrigen las resistividades
    (Zohdy completo) y se mide el RMS **final**. Gana el n con menor RMS:
    a diferencia de elegir la profundidad con las resistividades sin
    corregir, la comparación es entre modelos ya ajustados.

    La relación empírica AB/6 de la tomografía coincide con la
    profundidad mediana de investigación de Edwards (AB/5.3 en
    Schlumberger, AB/5.8 en Wenner); la búsqueda permite que los datos
    la confirmen o la corrijan.

    Parameters
    ----------
    longitud_ab : array
        Distancia AB de cada dato (en arreglos sin AB definida, una
        longitud equivalente, p. ej. 5.3·z_e).
    """
    rhoa = np.asarray(rhoa, float)
    L = np.asarray(longitud_ab, float)
    if np.any(np.diff(L) <= 0):
        raise ValueError("AB debe ser estrictamente creciente en la curva de Zohdy.")
    else:
        pass
    mejor = None
    busqueda = []
    for n in np.arange(n_min, n_max + 1e-9, paso):
        espesores = np.diff(np.concatenate([[0.0], L[:-1] / n]))
        rho, calc, historial = _corregir_resistividades(geom, rhoa, espesores, max_iter)
        rms = float(min(historial))
        busqueda.append((float(n), rms))
        if mejor is None or rms < mejor[1]:
            mejor = (float(n), rms, espesores, rho, calc, historial)
        else:
            pass
    n, rms, espesores, rho, calc, historial = mejor
    logger.info("Zohdy AB/n: n=%.3f, RMS %.2f%%", n, rms)
    return ResultadoZohdy(
        espesores=espesores, resistividades=rho, respuesta=calc,
        factor_profundidad=float("nan"), rms=rms, historial_rms=historial,
        n_ab=n, busqueda=busqueda,
    )


def _rms_log(obs: np.ndarray, calc: np.ndarray) -> float:
    """RMS relativo en porcentaje (sobre log ρa)."""
    return float(100.0 * np.sqrt(np.mean(np.log(obs / calc) ** 2)))


def invertir_zohdy(
    geom: fw.GeometriaSondeo,
    rhoa: np.ndarray,
    abscisa: np.ndarray,
    max_iter: int = 30,
    factores: np.ndarray | None = None,
) -> ResultadoZohdy:
    """Inversión de Zohdy.

    Parameters
    ----------
    geom : GeometriaSondeo
        Geometría de cada dato (curva monovaluada, ordenada por abscisa).
    rhoa : array
        Resistividad aparente observada.
    abscisa : array
        Profundidad mediana de investigación de cada dato
        (estrictamente creciente).
    max_iter : int
        Iteraciones máximas de corrección de resistividad.
    """
    rhoa = np.asarray(rhoa, float)
    abscisa = np.asarray(abscisa, float)
    if np.any(np.diff(abscisa) <= 0):
        raise ValueError("La abscisa de Zohdy debe ser estrictamente creciente.")
    else:
        pass
    factores = np.logspace(np.log10(0.3), np.log10(3.0), 31) if factores is None else factores

    # --- 1. Factor de profundidad ---
    mejor_factor, mejor_rms = 1.0, np.inf
    for factor in factores:
        bases = factor * abscisa[:-1]
        espesores = np.diff(np.concatenate([[0.0], bases]))
        rms = _rms_log(rhoa, fw.respuesta(geom, espesores, rhoa))
        if rms < mejor_rms:
            mejor_factor, mejor_rms = float(factor), rms
        else:
            pass
    espesores = np.diff(np.concatenate([[0.0], mejor_factor * abscisa[:-1]]))

    # --- 2. Corrección de resistividades ---
    rho = rhoa.copy()
    calculada = fw.respuesta(geom, espesores, rho)
    historial = [_rms_log(rhoa, calculada)]
    mejor_rho, mejor_calc = rho.copy(), calculada.copy()
    for _ in range(max_iter):
        cociente = np.clip(rhoa / calculada, 0.2, 5.0)
        rho = np.clip(rho * cociente, 1e-3, 1e7)
        calculada = fw.respuesta(geom, espesores, rho)
        rms = _rms_log(rhoa, calculada)
        historial.append(rms)
        if rms < min(historial[:-1]):
            mejor_rho, mejor_calc = rho.copy(), calculada.copy()
        else:
            pass
        # Se detiene cuando deja de mejorar de forma apreciable.
        if len(historial) > 3 and historial[-3] - rms < 0.01:
            break
        else:
            pass

    logger.info("Zohdy: factor=%.2f, RMS %.2f%% -> %.2f%% en %d iteraciones",
                mejor_factor, historial[0], min(historial), len(historial) - 1)
    return ResultadoZohdy(
        espesores=espesores, resistividades=mejor_rho, respuesta=mejor_calc,
        factor_profundidad=mejor_factor, rms=float(min(historial)), historial_rms=historial,
    )
