"""
perfil_sev.py
==================================================================
Perfiles 2D a partir de SEV: proyección sobre una ruta, sección
interpolada e inversión lateralmente restringida (LCI) con pyGIMLi.

Proyección
-------------
La ruta es una polilínea de vértices (X, Y) que dibuja el usuario. Cada
sondeo se proyecta sobre el segmento más cercano; su **cadenamiento**
(distancia a lo largo de la ruta) es la abscisa del perfil, y la
distancia perpendicular decide si entra (``distancia ≤ buffer``).

Sección interpolada
----------------------
Cada modelo 1D se muestrea en una rejilla de profundidad y se
interpola **en log ρ** a lo largo del cadenamiento. Es rápida y sirve
para cualquier modelo (multicapa, simplificado, IP), pero no usa los
datos: si dos sondeos vecinos llegaron a modelos equivalentes distintos,
la sección muestra un cambio lateral que puede ser solo equivalencia.

Inversión lateralmente restringida (LCI)
-------------------------------------------
Todos los sondeos del perfil se invierten **juntos**: el modelo es una
rejilla (sondeo × capa) en una malla 2D de pyGIMLi, cuya regularización
de suavidad liga cada celda con su vecina vertical (dentro del sondeo)
y con su vecina lateral (el sondeo contiguo). El peso relativo es el
**acoplamiento lateral**. La respuesta de cada columna es la de su
sondeo con su propia geometría, así que el jacobiano es diagonal por
bloques y se arma con las derivadas analíticas de :mod:`forward_sev`.

El efecto práctico es el que se busca en un perfil: la información de
un sondeo bien resuelto ayuda a resolver el vecino, y los cambios
laterales que aparecen son los que los datos exigen, no artefactos de
la equivalencia de cada SEV por separado (Auken y Christiansen, 2004).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import pygimli as pg

import forward_sev as fw
import inversion_sev as inv

logger = logging.getLogger("sev.perfil")


class PerfilError(Exception):
    """El perfil no se pudo construir."""


# ======================================================================
# Proyección
# ======================================================================


def proyectar(
    sondeos: pd.DataFrame, vertices: Sequence[Sequence[float]], buffer: float = np.inf,
) -> pd.DataFrame:
    """Proyecta sondeos (``Sondeo, X, Y, Z``) sobre una polilínea.

    Returns
    -------
    DataFrame
        Entrada más ``s`` (cadenamiento), ``distancia``, ``Xp``, ``Yp``,
        ``dentro``; ordenado por ``s``.
    """
    v = np.asarray(vertices, float)
    if v.ndim != 2 or len(v) < 2:
        raise PerfilError("La ruta necesita al menos dos vértices (X, Y).")
    else:
        pass
    segmentos = v[1:] - v[:-1]
    longitudes = np.hypot(segmentos[:, 0], segmentos[:, 1])
    acumulado = np.concatenate([[0.0], np.cumsum(longitudes)])

    filas = []
    for _, fila in sondeos.iterrows():
        p = np.array([fila["X"], fila["Y"]], float)
        mejor = None
        for i, (a, d, L) in enumerate(zip(v[:-1], segmentos, longitudes)):
            if L <= 0:
                continue
            else:
                pass
            t = float(np.clip(np.dot(p - a, d) / L ** 2, 0.0, 1.0))
            q = a + t * d
            dist = float(np.hypot(*(p - q)))
            if mejor is None or dist < mejor[0]:
                mejor = (dist, acumulado[i] + t * L, q)
            else:
                pass
        filas.append({**fila.to_dict(), "distancia": mejor[0], "s": mejor[1],
                      "Xp": mejor[2][0], "Yp": mejor[2][1], "dentro": mejor[0] <= buffer})
    salida = pd.DataFrame(filas).sort_values("s").reset_index(drop=True)
    salida.attrs["longitud"] = float(acumulado[-1])
    return salida


# ======================================================================
# Sección interpolada
# ======================================================================


def seccion_interpolada(
    proyectados: pd.DataFrame,
    modelos: dict[str, pd.DataFrame],
    columna: str = "rho",
    z_max: Optional[float] = None,
    n_s: int = 160,
    n_z: int = 120,
    usar_log: bool = True,
    dois: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Sección en una rejilla regular (cadenamiento × elevación).

    Parameters
    ----------
    proyectados : DataFrame
        Salida de :func:`proyectar` (solo filas ``dentro``).
    modelos : dict
        ``{sondeo: modelo}`` con ``z_techo, z_base`` y ``columna``.
    z_max : float, optional
        Profundidad máxima de la sección; por defecto la mayor DOI.

    Returns
    -------
    dict
        ``s`` (n_s), ``elev`` (n_z), ``valores`` (n_z × n_s, NaN fuera
        del terreno o bajo la profundidad máxima), ``topo`` (n_s),
        ``doi`` (n_s) y los sondeos usados.
    """
    usados = proyectados[proyectados["Sondeo"].astype(str).isin(modelos)].reset_index(drop=True)
    if len(usados) < 2:
        raise PerfilError("Hacen falta al menos dos sondeos invertidos sobre la ruta.")
    else:
        pass

    if z_max is None:
        if dois:
            z_max = float(np.nanmax([dois.get(str(s), np.nan) for s in usados["Sondeo"]]))
        else:
            z_max = float(np.nanmax([modelos[str(s)]["z_techo"].max() * 1.3 for s in usados["Sondeo"]]))
    else:
        pass
    if not np.isfinite(z_max) or z_max <= 0:
        z_max = 50.0
    else:
        pass

    z = np.linspace(0.0, z_max, n_z)
    columnas = []
    for _, fila in usados.iterrows():
        modelo = modelos[str(fila["Sondeo"])]
        bases = modelo["z_base"].to_numpy(float)[:-1]
        valores = fw.muestrear_modelo(np.diff(np.concatenate([[0.0], bases])),
                                      modelo[columna].to_numpy(float), z)
        valores = np.log10(np.maximum(valores, 1e-6)) if usar_log else valores
        columnas.append(valores)
    matriz = np.column_stack(columnas)  # (n_z, n_sondeos)

    s_sondeos = usados["s"].to_numpy(float)
    s = np.linspace(s_sondeos.min(), s_sondeos.max(), n_s)
    topo = np.interp(s, s_sondeos, usados["Z"].to_numpy(float))
    en_profundidad = np.vstack([np.interp(s, s_sondeos, matriz[i]) for i in range(n_z)])

    elev = np.linspace(topo.max(), topo.min() - z_max, n_z)
    valores = np.full((n_z, n_s), np.nan)
    for j in range(n_s):
        profundidad = topo[j] - elev
        dentro = (profundidad >= 0) & (profundidad <= z_max)
        valores[dentro, j] = np.interp(profundidad[dentro], z, en_profundidad[:, j])

    doi_s = None
    if dois:
        doi_s = np.interp(s, s_sondeos, [dois.get(str(x), np.nan) for x in usados["Sondeo"]])
    else:
        pass
    return {"s": s, "elev": elev, "valores": valores, "topo": topo, "doi": doi_s,
            "sondeos": usados, "z_max": z_max, "usar_log": usar_log}


# ======================================================================
# LCI con pyGIMLi
# ======================================================================


class FopLCI(pg.frameworks.MeshModelling):
    """Operador de varios sondeos sobre una malla (capa × sondeo)."""

    def __init__(self, geometrias: list[fw.GeometriaSondeo], espesores: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.geometrias = geometrias
        self.espesores = np.asarray(espesores, float)
        self.n_s = len(geometrias)
        self.n_z = len(self.espesores) + 1
        bases = np.cumsum(self.espesores)
        y = -np.concatenate([[0.0], bases, [bases[-1] * 1.5]])
        self._malla = pg.createGrid(x=np.arange(self.n_s + 1, dtype=float), y=y)
        self.setMesh(self._malla)
        self._cortes = np.cumsum([0] + [len(g) for g in geometrias])

    def _columnas(self, modelo) -> np.ndarray:
        # Celdas ordenadas con x (sondeo) más rápido: índice = capa·n_s + sondeo.
        return np.asarray(modelo, float).reshape(self.n_z, self.n_s)

    def response(self, modelo):
        m = self._columnas(modelo)
        partes = [fw.respuesta(g, self.espesores, m[:, i]) for i, g in enumerate(self.geometrias)]
        return pg.Vector(np.concatenate(partes))

    def createJacobian(self, modelo):
        m = self._columnas(modelo)
        J = np.zeros((self._cortes[-1], self.n_z * self.n_s))
        for i, g in enumerate(self.geometrias):
            ra, _jt, jr = fw.jacobiano_analitico(g, self.espesores, m[:, i])
            filas = slice(self._cortes[i], self._cortes[i + 1])
            J[filas, np.arange(self.n_z) * self.n_s + i] = jr
        self._J = pg.Matrix(np.ascontiguousarray(J))
        self.setJacobian(self._J)


def invertir_lci(
    proyectados: pd.DataFrame,
    datos: dict[str, pd.DataFrame],
    modelos_iniciales: dict[str, pd.DataFrame],
    lam: float = 20.0,
    acoplamiento: float = 1.0,
    n_capas: int = 25,
    z_max: Optional[float] = None,
) -> dict[str, Any]:
    """Inversión conjunta de los sondeos de un perfil.

    Parameters
    ----------
    datos : dict
        ``{sondeo: tabla preprocesada}`` (``usar``, ``rhoa_proc``,
        ``error_rel``, ``am..bn``, ``z_mediana``).
    modelos_iniciales : dict
        ``{sondeo: modelo multicapa}`` para el arranque.
    acoplamiento : float
        Peso de las restricciones laterales relativo a las verticales
        (1 = iguales; >1 perfiles más continuos; <1 sondeos más libres).
    """
    usados = proyectados[proyectados["Sondeo"].astype(str).isin(datos)].reset_index(drop=True)
    if len(usados) < 2:
        raise PerfilError("La LCI necesita al menos dos sondeos preprocesados sobre la ruta.")
    else:
        pass

    geometrias, d, e, nombres = [], [], [], []
    z_min_global, z_max_global = np.inf, 0.0
    for nombre in usados["Sondeo"].astype(str):
        tabla = datos[nombre]
        tabla = tabla[tabla["usar"]]
        geometrias.append(fw.GeometriaSondeo.desde_tabla(tabla))
        d.append(tabla["rhoa_proc"].to_numpy(float))
        e.append(tabla["error_rel"].to_numpy(float))
        nombres.append(nombre)
        z_min_global = min(z_min_global, float(tabla["z_mediana"].min()))
        z_max_global = max(z_max_global, float(tabla["z_mediana"].max()))

    fondo = z_max if z_max else 2.0 * z_max_global
    espesores = inv.capas_logaritmicas(max(z_min_global / 2.0, 0.05), fondo, n_capas)
    centros = np.concatenate([[0.0], np.cumsum(espesores)]) + np.concatenate([espesores / 2, [espesores[-1]]])

    inicial = np.zeros((len(espesores) + 1, len(nombres)))
    for i, nombre in enumerate(nombres):
        modelo = modelos_iniciales.get(nombre)
        if modelo is not None:
            bases = modelo["z_base"].to_numpy(float)[:-1]
            inicial[:, i] = fw.muestrear_modelo(np.diff(np.concatenate([[0.0], bases])),
                                                modelo["rho"].to_numpy(float), centros)
        else:
            inicial[:, i] = float(np.median(d[i]))

    fop = FopLCI(geometrias, espesores)
    fop.modelTrans = pg.trans.TransLogLU(inv.RHO_MIN, inv.RHO_MAX)
    inversor = pg.Inversion(fop=fop, verbose=False)
    inversor.dataTrans = pg.trans.TransLog()
    inversor.maxIter = 15
    # zWeight pondera las restricciones verticales frente a las laterales.
    fop.setRegionProperties("*", zWeight=1.0 / max(acoplamiento, 1e-3))
    datos_todos = np.concatenate(d)
    error_todos = np.concatenate(e)
    modelo = np.asarray(inversor.run(datos_todos, error_todos, lam=lam,
                                     startModel=inicial.ravel()), float)
    respuesta = np.asarray(fop.response(modelo), float)
    m = modelo.reshape(len(espesores) + 1, len(nombres))

    techos = np.concatenate([[0.0], np.cumsum(espesores)])
    bases = np.concatenate([np.cumsum(espesores), [np.inf]])
    filas_modelo = []
    filas_ajuste = []
    for i, nombre in enumerate(nombres):
        s = float(usados.loc[i, "s"])
        for j in range(len(techos)):
            filas_modelo.append({"Sondeo": nombre, "s": s, "capa": j + 1, "z_techo": techos[j],
                                 "z_base": bases[j], "rho": m[j, i]})
        tramo = slice(fop._cortes[i], fop._cortes[i + 1])
        tabla = datos[nombre][datos[nombre]["usar"]]
        for ab2, zm, obs, calc, err in zip(tabla["AB2"], tabla["z_mediana"], d[i], respuesta[tramo], e[i]):
            filas_ajuste.append({"Sondeo": nombre, "s": s, "AB2": ab2, "z_mediana": zm, "rhoa_obs": obs,
                                 "rhoa_lci": calc, "error_rel": err})
    ajuste = pd.DataFrame(filas_ajuste)
    chi2 = float(np.mean((np.log(ajuste["rhoa_obs"] / ajuste["rhoa_lci"]) / ajuste["error_rel"]) ** 2))
    por_sondeo = ajuste.groupby("Sondeo").apply(
        lambda g: float(np.mean((np.log(g["rhoa_obs"] / g["rhoa_lci"]) / g["error_rel"]) ** 2)),
        include_groups=False,
    ).rename("chi2").reset_index()
    return {
        "modelo": pd.DataFrame(filas_modelo), "ajuste": ajuste, "chi2": chi2,
        "chi2_por_sondeo": por_sondeo, "espesores": espesores, "lam": lam,
        "acoplamiento": acoplamiento,
    }


def modelos_desde_lci(modelo_lci: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convierte la tabla de la LCI en ``{sondeo: modelo}``."""
    return {str(nombre): grupo.sort_values("capa").reset_index(drop=True)
            for nombre, grupo in modelo_lci.groupby("Sondeo")}


def ruta_automatica(coords: pd.DataFrame) -> list[list[float]]:
    """Ruta recta por el eje principal de los sondeos (PCA).

    Útil como punto de partida: los extremos son las proyecciones más
    lejanas de los sondeos sobre su dirección de mayor dispersión.
    """
    xy = coords[["X", "Y"]].to_numpy(float)
    centro = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centro, full_matrices=False)
    eje = vt[0]
    t = (xy - centro) @ eje
    margen = 0.02 * (t.max() - t.min() + 1.0)
    a = centro + (t.min() - margen) * eje
    b = centro + (t.max() + margen) * eje
    return [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]
