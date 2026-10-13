"""
forward_sev.py
==================================================================
Modelado directo 1D (medio estratificado) para cualquier arreglo de
cuatro electrodos, sobre el núcleo C++ de pyGIMLi.

Por qué no se usa ``pygimli.physics.ves.VESModelling`` directamente
----------------------------------------------------------------------
``VESModelling`` (pyGIMLi 1.5–1.6) pasa las distancias al constructor
C++ en el orden ``(am, bm, an, bn)``, mientras que el constructor de
``DC1dModelling`` las espera como ``(am, an, bm, bn)``. En Schlumberger
y Wenner simétricos ``an == bm`` y el intercambio no se nota, pero en
dipolo-dipolo y polo-dipolo el resultado es incorrecto. Aquí se llama al
núcleo C++ con el orden correcto; :func:`verificar_forward` lo compara
con la solución analítica de imágenes de dos capas (error < 1e-4 %).

Además, el núcleo C++ requiere ``pg.Vector`` explícitos: pasarle arrays
de NumPy provoca un fallo de segmentación en algunas versiones.

Dos motores de modelado directo
----------------------------------
``"numpy"`` (por defecto)
    Transformada de resistividad de Koefoed evaluada con el filtro de
    Hankel J0 de 401 puntos de Key (2009), **vectorizada sobre todas
    las mediciones y sobre lotes de modelos a la vez**. Se resta la
    contribución primaria (ρ₁/r) antes de filtrar, lo que hace exacta
    la parte dominante del potencial. Es ~3 veces más rápida que el
    núcleo C++ por respuesta y, sobre todo, calcula el jacobiano
    completo en **una sola llamada** (todos los modelos perturbados en
    un lote), ~30 veces más rápido. Concuerda con pyGIMLi a 1e-4
    (error frente a la solución analítica < 1e-5 incluso con contrastes 100:1).
``"pygimli"``
    Núcleo C++ de pyGIMLi (``DC1dModelling``). Se conserva como
    referencia y para verificación cruzada.

En ambos casos la **inversión** la hace el marco ``pg.Inversion`` de
pyGIMLi (Gauss-Newton con regularización, transformaciones log, línea
de búsqueda); lo que cambia es solo quién evalúa la física.

Contenido
------------
- :class:`GeometriaSondeo`: distancias AM, AN, BM, BN de un sondeo.
- :func:`respuesta`: resistividad aparente de un modelo de capas.
- :func:`jacobiano_log`: sensibilidades ∂ln ρa / ∂ln p por diferencias.
- :func:`cargabilidad_aparente`: IP aparente por la relación de Seigel.
- :class:`FopMulticapa` y :class:`FopBloques`: operadores para el
  marco de inversión de pyGIMLi (espesores fijos / libres).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pygimli as pg

from filtro_hankel_sev import BASE as _BASE_HANKEL, J0 as _J0_HANKEL

logger = logging.getLogger("sev.forward")

pg.setLogLevel(40)  # solo errores: pyGIMLi es muy verboso por defecto

#: Motor de modelado directo activo (``"numpy"`` o ``"pygimli"``).
MOTOR = "numpy"
MOTORES = ("numpy", "pygimli")


def fijar_motor(motor: str) -> None:
    """Selecciona el motor de modelado directo de todo el módulo."""
    global MOTOR
    if motor in MOTORES:
        MOTOR = motor
    else:
        raise ValueError(f"Motor desconocido: {motor}. Opciones: {MOTORES}")


def _vec(x) -> "pg.Vector":
    return pg.Vector(np.ascontiguousarray(np.asarray(x, dtype=float)))


@dataclass
class GeometriaSondeo:
    """Distancias entre electrodos de cada medición de un sondeo."""

    am: np.ndarray
    an: np.ndarray
    bm: np.ndarray
    bn: np.ndarray

    @classmethod
    def desde_tabla(cls, df: pd.DataFrame) -> "GeometriaSondeo":
        """Desde una tabla con columnas ``am, an, bm, bn``."""
        return cls(*(df[c].to_numpy(float) for c in ("am", "an", "bm", "bn")))

    def __len__(self) -> int:
        return len(self.am)

    def subconjunto(self, mascara) -> "GeometriaSondeo":
        return GeometriaSondeo(self.am[mascara], self.an[mascara], self.bm[mascara], self.bn[mascara])

    @property
    def k(self) -> np.ndarray:
        inv = 1 / self.am - 1 / self.an - 1 / self.bm + 1 / self.bn
        return np.abs(2 * np.pi / inv)

    def vectores(self):
        return _vec(self.am), _vec(self.an), _vec(self.bm), _vec(self.bn)


# ----------------------------------------------------------------------
# Respuesta directa
# ----------------------------------------------------------------------


def respuesta_lote(geom: GeometriaSondeo, espesores, resistividades) -> np.ndarray:
    """Resistividad aparente de un **lote** de modelos (motor NumPy).

    Parameters
    ----------
    espesores : array (P, n−1)
    resistividades : array (P, n)

    Returns
    -------
    array (P, N)
        Una respuesta por modelo.
    """
    thk = np.atleast_2d(np.asarray(espesores, float))
    rho = np.atleast_2d(np.asarray(resistividades, float))
    r = np.stack([geom.am, geom.an, geom.bm, geom.bn])            # (4, N)
    activo = r < 1e8
    signo = np.array([1.0, -1.0, -1.0, 1.0])[:, None]
    w = np.where(activo, signo / np.where(activo, r, 1.0), 0.0)    # (4, N)
    unicas, inverso = np.unique(np.where(activo, r, 1.0), return_inverse=True)
    inverso = inverso.reshape(r.shape)
    lam = _BASE_HANKEL[None, :] / unicas[:, None]                  # (R, F)

    T = np.broadcast_to(rho[:, -1][:, None, None], (rho.shape[0],) + lam.shape).copy()
    for i in range(thk.shape[1] - 1, -1, -1):
        th = np.tanh(lam[None] * thk[:, i][:, None, None])
        rho_i = rho[:, i][:, None, None]
        T = (T + rho_i * th) / (1.0 + T * th / rho_i)
    rho1 = rho[:, 0][:, None]
    integral = (rho1 + (T - rho1[..., None]) @ _J0_HANKEL)[:, inverso]   # (P, 4, N)
    return (w[None] * integral).sum(axis=1) / w.sum(axis=0)[None]


def jacobiano_analitico(geom: GeometriaSondeo, espesores, resistividades, con_espesores: bool = True):
    """Respuesta y derivadas **analíticas** ∂ρa/∂h y ∂ρa/∂ρ (motor NumPy).

    Se derivan las recurrencias de Koefoed capa por capa
    (T_i = (T' + ρ_i t)/(1 + T' t/ρ_i), t = tanh(λ h_i))::

        ∂T_i/∂T' = (1 − t²) / D²
        ∂T_i/∂ρ_i = t (1 + 2 T' t/ρ_i + T'²/ρ_i²) / D²
        ∂T_i/∂h_i = λ (1 − t²) (ρ_i − T'²/ρ_i) / D²,   D = 1 + T' t/ρ_i

    La regla de la cadena se acumula con un producto acumulado sobre
    todas las capas a la vez y el filtro de Hankel se aplica en una sola
    multiplicación matricial: el jacobiano completo cuesta lo mismo que
    dos o tres respuestas, sin error de diferencias finitas.

    Returns
    -------
    (ra, d_espesores, d_resistividades)
        ``ra`` (N,), ``d_espesores`` (N, n−1) o ``None`` si
        ``con_espesores`` es falso, ``d_resistividades`` (N, n).
    """
    thk = np.asarray(espesores, float)
    rho = np.asarray(resistividades, float)
    n = len(rho)
    r = np.stack([geom.am, geom.an, geom.bm, geom.bn])
    activo = r < 1e8
    signo = np.array([1.0, -1.0, -1.0, 1.0])[:, None]
    w = np.where(activo, signo / np.where(activo, r, 1.0), 0.0)
    # El núcleo solo depende de la distancia: se evalúa una vez por
    # distancia distinta (en Schlumberger y Wenner AM = BN y AN = BM, así
    # que hay la mitad de distancias que de términos) y se reparte después.
    unicas, inverso = np.unique(np.where(activo, r, 1.0), return_inverse=True)
    inverso = inverso.reshape(r.shape)
    lam = _BASE_HANKEL[None, :] / unicas[:, None]          # (R, F)

    forma = (n,) + lam.shape
    d_sig = np.ones(forma)          # ∂T_i/∂T_{i+1} (la última fila queda en 1)
    d_rho = np.empty(forma)         # ∂T_i/∂ρ_i
    d_thk = np.empty((max(n - 1, 0),) + lam.shape) if con_espesores else None
    d_rho[n - 1] = 1.0
    T = np.full(lam.shape, rho[-1])
    for i in range(n - 2, -1, -1):
        t = np.tanh(lam * thk[i])
        ti = T * t / rho[i]
        D2 = (1.0 + ti) ** 2
        uno_t2 = 1.0 - t * t
        d_sig[i] = uno_t2 / D2
        d_rho[i] = t * (1.0 + 2.0 * ti + (T / rho[i]) ** 2) / D2
        if con_espesores:
            d_thk[i] = lam * uno_t2 * (rho[i] - T * T / rho[i]) / D2
        else:
            pass
        T = (T + rho[i] * t) / (1.0 + ti)

    G = w.sum(axis=0)
    integral = (rho[0] + (T - rho[0]) @ _J0_HANKEL)[inverso]
    ra = (w * integral).sum(axis=0) / G

    # Producto acumulado de ∂T/∂T' desde la superficie: P_j = Π_{i<j} d_sig_i.
    P = np.empty(forma)
    P[0] = 1.0
    if n > 1:
        P[1:] = np.cumprod(d_sig[:-1], axis=0)
    else:
        pass
    kernel = (P * d_rho) @ _J0_HANKEL                  # (n, R)
    kernel[0] = 1.0 + ((P[0] * d_rho[0]) - 1.0) @ _J0_HANKEL
    J_rho = ((w[None] * kernel[:, inverso]).sum(axis=1) / G[None]).T
    J_thk = None
    if con_espesores and n > 1:
        kernel_h = (P[:-1] * d_thk) @ _J0_HANKEL
        J_thk = ((w[None] * kernel_h[:, inverso]).sum(axis=1) / G[None]).T
    else:
        pass
    return ra, J_thk, J_rho


def _respuesta_pygimli(geom: GeometriaSondeo, thk: np.ndarray, rho: np.ndarray) -> np.ndarray:
    fop = pg.core.DC1dModelling(len(rho), *geom.vectores())
    return np.asarray(fop.response(_vec(np.concatenate([thk, rho]))))


def respuesta(geom: GeometriaSondeo, espesores, resistividades,
              motor: Optional[str] = None) -> np.ndarray:
    """Resistividad aparente de un modelo de capas.

    Parameters
    ----------
    espesores : array (n−1)
        Espesores de las capas finitas (m).
    resistividades : array (n)
        Resistividad de cada capa, la última es el semiespacio.
    motor : {'numpy', 'pygimli'}, optional
        Por defecto, el motor activo del módulo (:data:`MOTOR`).
    """
    rho = np.asarray(resistividades, float)
    thk = np.asarray(espesores, float)
    if len(rho) == 1:
        # Semiespacio = dos capas iguales (el núcleo exige al menos dos).
        thk = np.array([1.0])
        rho = np.array([rho[0], rho[0]])
    else:
        pass
    if (motor or MOTOR) == "pygimli":
        return _respuesta_pygimli(geom, thk, rho)
    else:
        return respuesta_lote(geom, thk[None, :], rho[None, :])[0]


def respuesta_fija(geom: GeometriaSondeo, espesores, resistividades) -> np.ndarray:
    """Igual que :func:`respuesta`, con el operador de espesores fijos."""
    fop = pg.core.DC1dRhoModelling(_vec(espesores), *geom.vectores())
    return np.asarray(fop.response(_vec(resistividades)))


def jacobiano_log(
    geom: GeometriaSondeo, espesores, resistividades,
    incluir_espesores: bool = True, paso: float = 1e-3,
) -> np.ndarray:
    """Sensibilidad ∂ln ρa / ∂ln p por diferencias centradas.

    Las columnas son ``[espesores..., resistividades...]`` si
    ``incluir_espesores``, o solo las resistividades.

    Con el motor NumPy se usa la derivada analítica
    (:func:`jacobiano_analitico`); con el de pyGIMLi, diferencias
    centradas. La
    sensibilidad logarítmica es exactamente la que necesitan la relación
    de Seigel para IP, la covarianza del modelo por capas y la
    profundidad de investigación.
    """
    thk = np.asarray(espesores, float)
    rho = np.asarray(resistividades, float)
    n_thk = len(thk)
    if MOTOR == "numpy" and len(rho) >= 2:
        ra, j_thk, j_rho = jacobiano_analitico(geom, thk, rho, con_espesores=incluir_espesores)
        jl_rho = j_rho * rho[None, :] / ra[:, None]
        if incluir_espesores:
            return np.hstack([j_thk * thk[None, :] / ra[:, None], jl_rho])
        else:
            return jl_rho
    else:
        pass
    base = np.concatenate([thk, rho])
    indices = list(range(len(base))) if incluir_espesores else list(range(n_thk, len(base)))
    lote = np.repeat(base[None, :], 2 * len(indices), axis=0)
    for fila, j in enumerate(indices):
        lote[2 * fila, j] *= np.exp(paso)
        lote[2 * fila + 1, j] *= np.exp(-paso)
    if MOTOR == "pygimli":
        respuestas = np.array([_respuesta_pygimli(geom, m[:n_thk], m[n_thk:]) for m in lote])
    else:
        respuestas = respuesta_lote(geom, lote[:, :n_thk], lote[:, n_thk:])
    log_r = np.log(respuestas)
    return ((log_r[0::2] - log_r[1::2]) / (2 * paso)).T


def cargabilidad_aparente(
    geom: GeometriaSondeo, espesores, resistividades, cargabilidades_mvv,
) -> np.ndarray:
    """Cargabilidad aparente (mV/V) por la relación exacta de Seigel (1959).

    ma = 1 − ρa(ρ) / ρa(ρ / (1 − m))

    con m en fracción (mV/V / 1000). Es la versión no lineal de la
    aproximación ma ≈ Σ (∂ln ρa/∂ln ρj)·mj que se usa para invertir.
    """
    m = np.clip(np.asarray(cargabilidades_mvv, float) / 1000.0, 0.0, 0.95)
    rho = np.asarray(resistividades, float)
    base = respuesta(geom, espesores, rho)
    cargada = respuesta(geom, espesores, rho / (1.0 - m))
    return 1000.0 * (1.0 - base / cargada)


def verificar_forward() -> float:
    """Error relativo máximo frente a la solución analítica de dos capas
    (serie de imágenes) en dipolo-dipolo, polo-dipolo y polo-polo, con
    los dos motores."""
    rho1, rho2, h = 100.0, 10.0, 5.0
    k = (rho2 - rho1) / (rho2 + rho1)

    def potencial(r):
        r = np.asarray(r, float)
        suma = 1.0 / r
        for n in range(1, 400):
            suma = suma + 2 * k ** n / np.sqrt(r ** 2 + (2 * n * h) ** 2)
        return np.where(r >= 1e8, 0.0, rho1 / (2 * np.pi) * suma)

    a = 10.0
    n = np.arange(1, 9, dtype=float)
    grande = 1e9
    casos = (
        (np.zeros(8), np.full(8, a), a + n * a, 2 * a + n * a),
        (np.zeros(8), np.full(8, -grande), n * a, n * a + a),
        (np.zeros(8), np.full(8, -grande), n * a, np.full(8, grande)),
    )
    peor = 0.0
    for A, B, M, N in casos:
        am, an, bm, bn = (np.where((np.abs(p) >= grande) | (np.abs(q) >= grande), grande, np.abs(q - p))
                          for p, q in ((A, M), (A, N), (B, M), (B, N)))
        g = 1 / am - 1 / an - 1 / bm + 1 / bn
        analitica = 2 * np.pi * (potencial(am) - potencial(an) - potencial(bm) + potencial(bn)) / g
        geo = GeometriaSondeo(am, an, bm, bn)
        for motor in MOTORES:
            numerica = respuesta(geo, [h], [rho1, rho2], motor=motor)
            peor = max(peor, float(np.max(np.abs(numerica / analitica - 1))))
    return peor


# ----------------------------------------------------------------------
# Operadores para el marco de inversión de pyGIMLi
# ----------------------------------------------------------------------


class FopMulticapa(pg.frameworks.MeshModelling):
    """Modelo de muchas capas de espesor fijo (inversión suave / Occam).

    La malla 1D de pyGIMLi da la regularización de suavidad vertical de
    primer orden sin más configuración.
    """

    def __init__(self, geom: GeometriaSondeo, espesores, **kwargs):
        super().__init__(**kwargs)
        self.geom = geom
        self.espesores = np.asarray(espesores, float)
        self._fwd = pg.core.DC1dRhoModelling(_vec(self.espesores), *geom.vectores())
        self._malla = pg.meshtools.createMesh1D(len(self.espesores) + 1)
        self.setMesh(self._malla)

    def response(self, modelo):
        if MOTOR == "pygimli":
            return self._fwd.response(_vec(modelo))
        else:
            return _vec(respuesta_lote(self.geom, self.espesores[None, :],
                                       np.asarray(modelo, float)[None, :])[0])

    def createJacobian(self, modelo):
        """Jacobiano en unidades naturales (∂ρa/∂ρ), en un solo lote."""
        rho = np.asarray(modelo, float)
        jl = jacobiano_log(self.geom, self.espesores, rho, incluir_espesores=False)
        ra = respuesta(self.geom, self.espesores, rho)
        self._J = pg.Matrix(np.ascontiguousarray(jl * ra[:, None] / rho[None, :]))
        self.setJacobian(self._J)

    def createStartModel(self, datos):
        return pg.Vector(len(self.espesores) + 1, float(np.median(datos)))


class FopBloques(pg.frameworks.Block1DModelling):
    """Modelo de pocas capas con espesores y resistividades libres."""

    def __init__(self, geom: GeometriaSondeo, n_capas: int, **kwargs):
        self.geom = geom
        super().__init__(nLayers=n_capas, **kwargs)
        self.setMultiThreadJacobian(1)

    def response(self, modelo):
        m = np.asarray(modelo, float)
        n = self.nLayers
        return _vec(respuesta(self.geom, m[: n - 1], m[n - 1:]))

    def response_mt(self, modelo, i=0):
        return self.response(modelo)

    def createJacobian(self, modelo):
        """Jacobiano en unidades naturales, en un solo lote."""
        m = np.asarray(modelo, float)
        n = self.nLayers
        jl = jacobiano_log(self.geom, m[: n - 1], m[n - 1:], incluir_espesores=True)
        ra = respuesta(self.geom, m[: n - 1], m[n - 1:])
        self._J = pg.Matrix(np.ascontiguousarray(jl * ra[:, None] / m[None, :]))
        self.setJacobian(self._J)


def profundidades(espesores) -> np.ndarray:
    """Profundidad de la base de cada capa finita."""
    return np.cumsum(np.asarray(espesores, float))


def muestrear_modelo(espesores, resistividades, z: np.ndarray) -> np.ndarray:
    """Valor del modelo de capas en las profundidades ``z``."""
    bases = profundidades(espesores)
    indice = np.searchsorted(bases, np.asarray(z, float), side="right")
    return np.asarray(resistividades, float)[np.clip(indice, 0, len(resistividades) - 1)]
