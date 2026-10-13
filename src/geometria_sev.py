"""
geometria_sev.py
==================================================================
Geometría de los arreglos de SEV: posiciones de electrodos, factor
geométrico, detección del arreglo y profundidad mediana de
investigación.

Arreglos soportados
----------------------
====  ======================  ==============================================
Code  Arreglo                 Posiciones (origen en el centro del sondeo)
====  ======================  ==============================================
SCH   Schlumberger            A,B = ∓AB/2 · M,N = ∓MN/2
WE    Wenner                  A,M,N,B = −1.5a, −0.5a, 0.5a, 1.5a
DD    Dipolo-dipolo axial     dipolos de longitud *d* separados *n·d*
PD    Polo-dipolo             A en 0, B al infinito, M = n·d, N = (n+1)·d
PP    Polo-polo               A en 0, M = r, B y N al infinito
====  ======================  ==============================================

Convenciones de columnas
---------------------------
El formato ``Sondeo, ID, a, AB/2, MN/2, K…`` no fija cómo se escribe la
separación de los arreglos dipolares en la columna ``AB/2``: cada
operador la anota a su manera (distancia entre centros de dipolo, *n·a*,
el propio *n*…). En vez de imponer una, el módulo **prueba todas las
convenciones razonables y se queda con la que reproduce la columna K
del archivo**. Es un clasificador sencillo con una ventaja enorme: el
arreglo y la convención se deciden con evidencia, y el error de K que
queda es a la vez un control de calidad de la geometría de campo.

Si el archivo no trae K, se usan reglas geométricas (Wenner si
AB/2 = 1.5·a; Schlumberger si MN ≪ AB) y, en los arreglos dipolares, la
convención por defecto (distancia entre centros).

Profundidad mediana de investigación
---------------------------------------
:func:`profundidad_mediana` calcula la profundidad de Edwards (1977)
para **cualquier** geometría con la característica de investigación en
profundidad de un semiespacio homogéneo (Roy y Apparao, 1971). Para
dos electrodos a distancia *r*, la fracción de señal aportada por el
terreno por encima de *z* es::

    C(z; r) = 1 − r / sqrt(r² + 4 z²)

y para un arreglo de cuatro electrodos se combina con los mismos
pesos que el factor geométrico. Reproduce los valores tabulados
(Wenner 0.519·a, polo-polo 0.867·a, dipolo-dipolo n=1 0.416·a) y es la
abscisa que usa Zohdy para arreglos que no son Schlumberger.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sev.geometria")

#: Distancia que representa "infinito" para el modelado (1/r despreciable).
INFINITO = 1.0e9

ARREGLOS: dict[str, str] = {
    "SCH": "Schlumberger",
    "WE": "Wenner",
    "DD": "Dipolo-dipolo",
    "PD": "Polo-dipolo",
    "PP": "Polo-polo",
}

#: Error mediano de K (relativo) por debajo del cual la detección se
#: considera confirmada por el archivo.
TOLERANCIA_K = 0.03
#: Error mediano (en ln K) por encima del cual la columna K del archivo no
#: se considera un factor geométrico: se ignora para detectar el arreglo.
UMBRAL_K_CONFIABLE = 0.25


def _valido(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def _dipolo(a: np.ndarray, mn2: np.ndarray) -> np.ndarray:
    """Longitud de dipolo: columna ``a`` o, si falta, ``2·MN/2``."""
    return np.where(_valido(a), a, 2.0 * mn2)


# ----------------------------------------------------------------------
# Posiciones (A, B, M, N) de cada convención
# ----------------------------------------------------------------------
# Cada función recibe arrays (a, ab2, mn2) y devuelve cuatro arrays de
# posiciones sobre el eje del sondeo. Los electrodos al infinito llevan
# ±INFINITO.


def _sch(a, ab2, mn2):
    mn = np.where(_valido(mn2), mn2, np.where(_valido(a), a / 2.0, ab2 / 5.0))
    return -ab2, ab2, -mn, mn


def _we_a(a, ab2, mn2):
    aa = np.where(_valido(a), a, 2.0 * ab2 / 3.0)
    return -1.5 * aa, 1.5 * aa, -0.5 * aa, 0.5 * aa


def _we_ab2(a, ab2, mn2):
    aa = 2.0 * ab2 / 3.0
    return -1.5 * aa, 1.5 * aa, -0.5 * aa, 0.5 * aa


def _dd(d, n):
    centros = (n + 1.0) * d
    return (-centros / 2 - d / 2, -centros / 2 + d / 2, centros / 2 - d / 2, centros / 2 + d / 2)


def _dd_centros(a, ab2, mn2):
    d = _dipolo(a, mn2)
    return _dd(d, ab2 / d - 1.0)


def _dd_sep(a, ab2, mn2):
    d = _dipolo(a, mn2)
    return _dd(d, ab2 / d)


def _dd_n(a, ab2, mn2):
    return _dd(_dipolo(a, mn2), ab2)


def _dd_semicentros(a, ab2, mn2):
    d = _dipolo(a, mn2)
    return _dd(d, 2.0 * ab2 / d - 1.0)


def _pd(d, n):
    cero = np.zeros_like(d)
    return cero, np.full_like(d, -INFINITO), n * d, (n + 1.0) * d


def _pd_centro(a, ab2, mn2):
    d = _dipolo(a, mn2)
    return _pd(d, ab2 / d - 0.5)


def _pd_am(a, ab2, mn2):
    d = _dipolo(a, mn2)
    return _pd(d, ab2 / d)


def _pd_n(a, ab2, mn2):
    return _pd(_dipolo(a, mn2), ab2)


def _pp(r):
    cero = np.zeros_like(r)
    return cero, np.full_like(r, -INFINITO), r, np.full_like(r, INFINITO)


def _pp_ab2(a, ab2, mn2):
    return _pp(ab2)


def _pp_a(a, ab2, mn2):
    return _pp(np.where(_valido(a), a, ab2))


def _pp_2ab2(a, ab2, mn2):
    return _pp(2.0 * ab2)


@dataclass(frozen=True)
class Convencion:
    """Una forma de leer las columnas de distancia de un arreglo."""

    arreglo: str
    clave: str
    descripcion: str
    posiciones: Callable


CONVENCIONES: tuple[Convencion, ...] = (
    Convencion("SCH", "estandar", "A,B = ∓AB/2; M,N = ∓MN/2", _sch),
    Convencion("WE", "a", "espaciamiento a de la columna 'a'", _we_a),
    Convencion("WE", "ab2", "a = 2/3 · AB/2", _we_ab2),
    Convencion("DD", "centros", "AB/2 = distancia entre centros de dipolo = (n+1)·a", _dd_centros),
    Convencion("DD", "sep", "AB/2 = separación entre dipolos = n·a", _dd_sep),
    Convencion("DD", "n", "AB/2 = factor n", _dd_n),
    Convencion("DD", "semicentros", "AB/2 = mitad de la distancia entre centros", _dd_semicentros),
    Convencion("PD", "centro", "AB/2 = distancia de A al centro de MN = (n+½)·a", _pd_centro),
    Convencion("PD", "am", "AB/2 = distancia AM = n·a", _pd_am),
    Convencion("PD", "n", "AB/2 = factor n", _pd_n),
    Convencion("PP", "ab2", "r = AB/2", _pp_ab2),
    Convencion("PP", "a", "r = a", _pp_a),
    Convencion("PP", "2ab2", "r = 2·AB/2", _pp_2ab2),
)

#: Convención por defecto de cada arreglo cuando no hay K para decidir.
CONVENCION_DEFECTO = {"SCH": "estandar", "WE": "a", "DD": "centros", "PD": "centro", "PP": "ab2"}


def buscar_convencion(arreglo: str, clave: str) -> Convencion:
    for conv in CONVENCIONES:
        if conv.arreglo == arreglo and conv.clave == clave:
            return conv
        else:
            pass
    raise KeyError(f"Convención desconocida: {arreglo}/{clave}")


# ----------------------------------------------------------------------
# Distancias y factor geométrico
# ----------------------------------------------------------------------


def distancias(A, B, M, N) -> dict[str, np.ndarray]:
    """Distancias AM, AN, BM, BN (con los infinitos acotados a
    :data:`INFINITO`, que es lo que necesita el modelado)."""
    def _d(p, q):
        dist = np.abs(np.asarray(q, float) - np.asarray(p, float))
        lejos = (np.abs(p) >= INFINITO) | (np.abs(q) >= INFINITO)
        return np.where(lejos, INFINITO, np.maximum(dist, 1e-6))

    return {"am": _d(A, M), "an": _d(A, N), "bm": _d(B, M), "bn": _d(B, N)}


def factor_geometrico(am, an, bm, bn) -> np.ndarray:
    """K = 2π / (1/AM − 1/AN − 1/BM + 1/BN), en valor absoluto."""
    inv = 1.0 / np.asarray(am) - 1.0 / np.asarray(an) - 1.0 / np.asarray(bm) + 1.0 / np.asarray(bn)
    with np.errstate(divide="ignore"):
        return np.abs(2.0 * np.pi / inv)


def geometria(df: pd.DataFrame, arreglo: str, convencion: str) -> pd.DataFrame:
    """Posiciones, distancias, K y profundidad mediana de cada fila.

    Returns
    -------
    pandas.DataFrame
        Columnas ``xA, xB, xM, xN, am, an, bm, bn, K_geom, z_mediana``,
        alineadas con ``df``.
    """
    conv = buscar_convencion(arreglo, convencion)
    a = df["a"].to_numpy(float)
    ab2 = df["AB2"].to_numpy(float)
    mn2 = df["MN2"].to_numpy(float)
    # AB/2 ausente pero 'a' presente (p. ej. Wenner escrito solo con a).
    ab2 = np.where(_valido(ab2), ab2, 1.5 * a)

    A, B, M, N = (np.asarray(p, float) for p in conv.posiciones(a, ab2, mn2))
    dist = distancias(A, B, M, N)
    salida = pd.DataFrame({
        "xA": A, "xB": B, "xM": M, "xN": N, **dist,
    }, index=df.index)
    salida["K_geom"] = factor_geometrico(dist["am"], dist["an"], dist["bm"], dist["bn"])
    salida["z_mediana"] = profundidad_mediana(dist["am"], dist["an"], dist["bm"], dist["bn"])
    return salida


# ----------------------------------------------------------------------
# Profundidad mediana de investigación (Edwards, 1977)
# ----------------------------------------------------------------------


def _fraccion_somera(z: np.ndarray, am, an, bm, bn) -> np.ndarray:
    """Fracción de la señal aportada por el terreno entre 0 y z."""
    total = np.zeros_like(z, dtype=float)
    pesos = np.zeros_like(z, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _acumular(z, am, an, bm, bn, total, pesos)


def _acumular(z, am, an, bm, bn, total, pesos):
    for r, signo in ((am, 1.0), (an, -1.0), (bm, -1.0), (bn, 1.0)):
        r = np.asarray(r, float)
        activo = r < INFINITO
        w = np.where(activo, signo / np.where(activo, r, 1.0), 0.0)
        c = np.where(activo, 1.0 - r / np.sqrt(r ** 2 + 4.0 * z ** 2), 0.0)
        total = total + w * c
        pesos = pesos + w
    return total / pesos


def profundidad_mediana(am, an, bm, bn) -> np.ndarray:
    """Profundidad a la que el terreno somero aporta el 50 % de la señal.

    Se resuelve por bisección en log(z), vectorizada sobre todas las
    mediciones.
    """
    am = np.asarray(am, float)
    escala = np.fmin.reduce([am, np.asarray(an, float), np.asarray(bm, float),
                             np.asarray(bn, float)])
    bajo = np.log(escala * 1e-3)
    alto = np.log(escala * 1e3)
    for _ in range(60):
        medio = 0.5 * (bajo + alto)
        f = _fraccion_somera(np.exp(medio), am, an, bm, bn)
        arriba = f > 0.5
        alto = np.where(arriba, medio, alto)
        bajo = np.where(arriba, bajo, medio)
    return np.exp(0.5 * (bajo + alto))


# ----------------------------------------------------------------------
# Detección del arreglo
# ----------------------------------------------------------------------


@dataclass
class Deteccion:
    """Resultado de la detección de arreglo y convención."""

    arreglo: str
    convencion: str
    error_k: float
    confirmada: bool
    mensaje: str
    #: Columna del archivo que contiene el factor geométrico ("K", "R", o
    #: "" si ninguna lo reproduce y se usó la geometría).
    columna_k: str = "K"

    def as_dict(self) -> dict:
        return {
            "arreglo": self.arreglo, "convencion": self.convencion,
            "error_k": self.error_k, "confirmada": self.confirmada,
            "mensaje": self.mensaje,
        }


def _error_k_max(df: pd.DataFrame, conv: Convencion, columna: str = "K") -> float:
    """Peor error (ln K) entre la columna y la geometría, fila por fila."""
    k_col = df[columna].to_numpy(float)
    k_geo = geometria(df, conv.arreglo, conv.clave)["K_geom"].to_numpy(float)
    ok = _valido(k_col) & _valido(k_geo)
    return float(np.max(np.abs(np.log(k_col[ok] / k_geo[ok])))) if ok.any() else float("nan")


def _error_k(df: pd.DataFrame, conv: Convencion, columna: str = "K") -> float:
    k_col = df[columna].to_numpy(float)
    if not _valido(k_col).any():
        return float("nan")
    else:
        pass
    k_geo = geometria(df, conv.arreglo, conv.clave)["K_geom"].to_numpy(float)
    ok = _valido(k_col) & _valido(k_geo)
    if not ok.any():
        return float("nan")
    else:
        pass
    return float(np.median(np.abs(np.log(k_col[ok] / k_geo[ok]))))


def _regla_geometrica(df: pd.DataFrame) -> str:
    """Arreglo más probable sin K, por la forma de las distancias."""
    a = df["a"].to_numpy(float)
    ab2 = df["AB2"].to_numpy(float)
    mn2 = df["MN2"].to_numpy(float)
    ok = _valido(a) & _valido(ab2)
    if ok.sum() >= 2 and np.median(np.abs(ab2[ok] / (1.5 * a[ok]) - 1.0)) < 0.05:
        return "WE"
    else:
        pass
    ok = _valido(mn2) & _valido(ab2)
    if ok.sum() >= 2 and np.median(mn2[ok] / ab2[ok]) <= 0.34:
        return "SCH"
    else:
        pass
    if _valido(a).any() and not _valido(ab2).any():
        return "WE"
    else:
        pass
    return "SCH"


def detectar_arreglo(df: pd.DataFrame, forzado: Optional[str] = None) -> Deteccion:
    """Decide arreglo y convención de un sondeo.

    Parameters
    ----------
    df : DataFrame
        Registros de un sondeo (todas sus direcciones).
    forzado : str, optional
        Código de arreglo impuesto por el usuario; solo se elige entre
        sus convenciones.
    """
    candidatos = [c for c in CONVENCIONES if forzado in (None, "", "AUTO") or c.arreglo == forzado]

    def elegir(columna: str):
        errores = [(c, _error_k(df, c, columna)) for c in candidatos]
        con_k = [(c, e) for c, e in errores if np.isfinite(e)]
        if not con_k:
            return None, float("nan")
        else:
            pass
        error_min = min(e for _, e in con_k)
        # Empates: varias convenciones dan la misma K (Wenner y polo-polo con
        # r = a dan 2πa; Wenner es un Schlumberger con MN = AB/3). Entre las
        # que reproducen K igual de bien en la mediana se prefiere (1) la que
        # lo reproduce en **todas** las filas (si alguna lectura no tiene
        # geometría Wenner exacta, la posición real de los electrodos AB/2,
        # MN/2 es la buena) y (2) el arreglo que indica la geometría de las
        # distancias, con Wenner antes que Schlumberger.
        empatadas = [(c, e) for c, e in con_k if e <= error_min + 1e-3]
        preferido = _regla_geometrica(df)
        orden = {preferido: 0, "WE": 1, "SCH": 2}

        def clave_orden(par):
            conv, err = par
            return (round(_error_k_max(df, conv, columna), 3), orden.get(conv.arreglo, 3), err)

        mejor, error = min(empatadas, key=clave_orden)
        return mejor, error

    if "K" in df and _valido(df["K"].to_numpy(float)).any():
        mejor, error = elegir("K")
    else:
        mejor, error = None, float("nan")
    if mejor is not None and error <= UMBRAL_K_CONFIABLE:
        confirmada = error <= TOLERANCIA_K
        if confirmada:
            mensaje = (f"{ARREGLOS[mejor.arreglo]} ({mejor.descripcion}): la geometría "
                       f"reproduce la columna K con error mediano {100 * error:.1f} %.")
        else:
            mensaje = (f"Ninguna convención reproduce bien la columna K (mejor: "
                       f"{ARREGLOS[mejor.arreglo]}, {mejor.descripcion}, error "
                       f"{100 * error:.0f} %). Revise K o fuerce el arreglo.")
        return Deteccion(mejor.arreglo, mejor.clave, float(error), confirmada, mensaje)
    else:
        pass

    # La columna K no es un factor geométrico (o no existe). ¿Está K en otra
    # columna? Un intercambio de columnas K ↔ R es frecuente al exportar.
    if mejor is not None:
        if "R" in df and _valido(df["R"].to_numpy(float)).any():
            mejor_r, error_r = elegir("R")
        else:
            mejor_r, error_r = None, float("nan")
        if mejor_r is not None and error_r <= TOLERANCIA_K:
            return Deteccion(
                mejor_r.arreglo, mejor_r.clave, float(error_r), True,
                f"La columna K no es un factor geométrico (error mediano {100 * error:.0f} % con cualquier "
                f"arreglo), pero la columna R sí lo es: {ARREGLOS[mejor_r.arreglo]} "
                f"({mejor_r.descripcion}), error {100 * error_r:.1f} %. Parece que K y R están "
                "intercambiadas en el archivo; se usa R como K.",
                columna_k="R",
            )
        else:
            pass
        arreglo = forzado if forzado in ARREGLOS else _regla_geometrica(df)
        clave = CONVENCION_DEFECTO[arreglo]
        conv = buscar_convencion(arreglo, clave)
        return Deteccion(
            arreglo, clave, float(error), False,
            f"La columna K no corresponde a ningún arreglo (mejor error mediano {100 * error:.0f} %): se "
            f"ignora y se asume {ARREGLOS[arreglo]} ({conv.descripcion}) por la geometría de AB/2, MN/2 y a. "
            "Revise el contenido de la columna K.",
            columna_k="",
        )
    else:
        pass

    arreglo = forzado if forzado in ARREGLOS else _regla_geometrica(df)
    clave = CONVENCION_DEFECTO[arreglo]
    conv = buscar_convencion(arreglo, clave)
    return Deteccion(
        arreglo, clave, float("nan"), False,
        f"Sin columna K: se asume {ARREGLOS[arreglo]} ({conv.descripcion}).",
    )
