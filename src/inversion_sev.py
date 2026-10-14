"""
inversion_sev.py
==================================================================
Inversión 1D de resistividad y cargabilidad de un SEV.

Flujo
--------
::

    curva preprocesada
        │
        ├─► Zohdy (modelo inicial, N capas = N datos)
        │
        ├─► pyGIMLi Occam  (espesores de Zohdy fijos, suavidad vertical,
        │                    λ elegido por Optuna o principio de discrepancia)
        │        = MODELO MULTICAPA  ──► DOI (Oldenburg y Li, 1999)
        │
        ├─► Reducción de Dar Zarrouk  para k = 2…kmax capas
        │        (segmentación óptima por programación dinámica)
        │        └─► pyGIMLi Marquardt (espesores y ρ libres) por cada k
        │                └─► BIC → número de capas
        │        = MODELO SIMPLIFICADO  ──► covarianza / equivalencia
        │
        └─► IP: inversión lineal de Seigel sobre cada modelo
                 (mismo jacobiano, cargabilidad acotada a ≥ 0)

Decisiones de diseño
-----------------------
- **Elección de λ** (Optuna o discrepancia): el objetivo es que el
  ajuste quede al nivel del ruido (χ² ≈ 1): con λ menor el modelo
  ajusta ruido, con λ mayor desperdicia información.
- **Número de capas por BIC**: es la parte que en los programas
  tradicionales decide el intérprete a ojo. El criterio de información
  bayesiano (BIC = Φ + p·ln N) penaliza cada capa añadida; si dos
  modelos difieren en menos de 2 unidades de BIC se elige el más
  simple (Kass y Raftery, 1995).
- **Dos arranques por k**: la reducción clásica de Dar Zarrouk
  (h = √(T·S), ρ = √(T/S)) y la que conserva el espesor del grupo
  (ρ = √(ρT·ρL)); se invierten ambas y se queda la de menor χ². La
  primera conserva exactamente S y T del grupo; la segunda, su
  profundidad. Ninguna es mejor siempre, y evaluar las dos cuesta
  milisegundos.
- **Equivalencia**: la covarianza linealizada del modelo por capas
  dice qué parámetro está realmente resuelto en cada capa (h y ρ, solo
  S, solo T, o nada), en vez de presentar valores puntuales con una
  precisión que no tienen.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pygimli as pg

import forward_sev as fw
import zohdy_sev as zh

logger = logging.getLogger("sev.inversion")


class InversionSEVError(Exception):
    """La inversión no pudo completarse."""


#: Límites físicos de resistividad para las transformaciones log (Ω·m).
RHO_MIN, RHO_MAX = 0.01, 1.0e6


# ======================================================================
# Utilidades
# ======================================================================


def _chi2(obs, calc, error_rel) -> float:
    return float(np.mean((np.log(obs / calc) / error_rel) ** 2))


def _rms(obs, calc) -> float:
    return float(100.0 * np.sqrt(np.mean(np.log(obs / calc) ** 2)))


def rugosidad(rho: np.ndarray) -> float:
    """RMS de los saltos de log10 ρ entre capas consecutivas."""
    if len(rho) < 2:
        return 0.0
    else:
        return float(np.sqrt(np.mean(np.diff(np.log10(rho)) ** 2)))


def extender_capas(espesores, resistividades, z_fondo: float, max_extra: int = 8):
    """Añade capas bajo la última base hasta ``z_fondo`` con paso log.

    El paso es el cociente de profundidad de las dos últimas bases
    (acotado a 1.15–1.6). Las capas nuevas toman la resistividad del
    semiespacio original.
    """
    h = np.asarray(espesores, float)
    rho = np.asarray(resistividades, float)
    bases = np.cumsum(h)
    if len(bases) == 0 or bases[-1] >= z_fondo:
        return h, rho
    else:
        pass
    razon = float(np.clip(bases[-1] / bases[-2] if len(bases) > 1 else 1.3, 1.15, 1.6))
    nuevas = []
    z = bases[-1]
    while z * razon < z_fondo and len(nuevas) < max_extra:
        z = z * razon
        nuevas.append(z)
    if not nuevas:
        return h, rho
    else:
        pass
    todas = np.concatenate([bases, nuevas])
    return np.diff(np.concatenate([[0.0], todas])), np.concatenate([rho, np.full(len(nuevas), rho[-1])])


def capas_logaritmicas(z_min: float, z_max: float, n: int) -> np.ndarray:
    """Espesores de ``n−1`` capas con bases equiespaciadas en log(z)."""
    bases = np.logspace(np.log10(z_min), np.log10(z_max), n - 1)
    return np.diff(np.concatenate([[0.0], bases]))


def malla_desde_zohdy(espesores_zohdy, factor: float, z_e_max: float, n_objetivo: int,
                      z_e_min: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Malla del multicapa construida sobre las interfaces de Zohdy.

    Zohdy coloca la base de cada capa en f·z_e (f = factor de profundidad
    que mejor ajusta la curva, z_e = profundidad mediana de investigación
    de cada lectura). Ese factor calibra con los datos la escala de
    profundidades, así que:

    - la **profundidad máxima viable** es f·max(z_e): la profundidad a la
      que el dato de mayor apertura todavía aporta información; debajo
      solo queda el semiespacio;
    - las celdas del multicapa son las capas de Zohdy subdivididas en
      log z (≈ ``n_objetivo`` celdas en total), de modo que cada interfaz
      de Zohdy es un nodo de la malla y el modelo inicial entra **exacto**,
      sin remuestrearlo.

    Returns
    -------
    (espesores, z_viable)
    """
    bases = np.cumsum(np.asarray(espesores_zohdy, float))
    z_viable = float(max(factor * float(z_e_max), bases[-1] * 1.05 if len(bases) else 1.0))
    nodos = np.unique(np.concatenate([bases[bases < z_viable], [z_viable]]))
    # Primera celda: la más somera que ve el dato de menor apertura.
    z_tope = float(nodos[0]) if z_e_min is None else float(min(nodos[0], max(0.5 * factor * z_e_min, 0.05)))
    nodos = np.unique(np.concatenate([[z_tope], nodos]))
    por_tramo = max(int(np.ceil(n_objetivo / max(len(nodos) - 1, 1))), 1)
    interfaces = [nodos[0]]
    for a, b in zip(nodos[:-1], nodos[1:]):
        interfaces.extend(np.exp(np.linspace(np.log(a), np.log(b), por_tramo + 1))[1:])
    interfaces = np.asarray(interfaces, float)
    return np.diff(np.concatenate([[0.0], interfaces])), z_viable


def longitud_ab_equivalente(tabla: pd.DataFrame) -> np.ndarray:
    """Longitud AB de cada lectura para la profundidad z = AB/n de Zohdy.

    En arreglos con los electrodos de potencial entre los de corriente
    (Schlumberger, Wenner) es la distancia A–B real. En los demás
    (dipolares, polo) AB no tiene el mismo significado; se usa una
    longitud equivalente 5.3·z_e, de modo que AB/5.3 sea la profundidad
    de Edwards como en Schlumberger.
    """
    xa, xb = tabla["xA"].to_numpy(float), tabla["xB"].to_numpy(float)
    xm, xn = tabla["xM"].to_numpy(float), tabla["xN"].to_numpy(float)
    ze = tabla["z_mediana"].to_numpy(float)
    con_ab = (np.isfinite(xa) & np.isfinite(xb) & (np.minimum(xa, xb) < np.minimum(xm, xn))
              & (np.maximum(xm, xn) < np.maximum(xa, xb)))
    return np.where(con_ab, np.abs(xb - xa), 5.3 * ze)


def _inversor(fop) -> pg.Inversion:
    inv = pg.Inversion(fop=fop, verbose=False)
    inv.dataTrans = pg.trans.TransLog()
    return inv


# ======================================================================
# Occam (modelo multicapa)
# ======================================================================


#: Parámetro β del soporte de gradiente mínimo (en ln ρ): saltos mayores
#: que β se consideran interfaces y apenas se penalizan.
BETA_NITIDA = 0.05


def invertir_occam(
    geom: fw.GeometriaSondeo, rhoa: np.ndarray, error_rel: np.ndarray,
    espesores: np.ndarray, inicial: np.ndarray, lam: float,
    robusto: bool = False, referencia: bool = False, tipo_restriccion: int = 1,
    max_iter: int = 20, nitida: bool = False, beta: float = BETA_NITIDA,
    reponderaciones: int = 15, pesos_desde_inicial: bool = False,
) -> dict[str, Any]:
    """Inversión de espesores fijos con pyGIMLi (suave o nítida).

    Parameters
    ----------
    nitida : bool
        Regularización de **soporte de gradiente mínimo** (MGS;
        Portniaguine y Zhdanov, 1999; Vignoli et al., 2015). Se resuelve
        con reponderación iterativa **hasta converger** (hasta 15 vueltas;
        con 5 vueltas cortas el modelo quedaba a medio camino entre suave y
        por bloques, con los contrastes profundos desplazados hacia abajo).
        En cada vuelta, el peso de la
        restricción entre dos capas es β/√(Δm² + β²), de modo que los
        saltos grandes (interfaces) dejan de penalizarse y los pequeños
        se aplanan. El resultado es un modelo **por bloques con
        interfaces nítidas**, que es lo que la reducción de Dar Zarrouk
        necesita para encontrar las capas reales; la suavidad clásica
        (L2) reparte cada salto en varias capas y desdibuja las
        interfaces.
    referencia : bool
        Si ``True``, el modelo inicial es también referencia.

    Notas
    -----
    Se desactiva el criterio de parada de pyGIMLi en χ² < 1
    (``stopAtChi1``): con él la inversión se detiene en cuanto el ajuste
    alcanza el error asignado, aunque el modelo siga cambiando, y la
    búsqueda de λ comparaba modelos a medio converger.
    """
    fop = fw.FopMulticapa(geom, espesores)
    fop.modelTrans = pg.trans.TransLogLU(RHO_MIN, RHO_MAX)
    inv = _inversor(fop)
    d = np.asarray(rhoa, float)
    e = np.asarray(error_rel, float)
    modelo = np.clip(np.asarray(inicial, float), RHO_MIN * 1.01, RHO_MAX * 0.99)
    vueltas = reponderaciones if nitida else 1
    for vuelta in range(vueltas):
        # Con pesos_desde_inicial (continuación en λ) los pesos de MGS se
        # calculan desde la primera vuelta con el modelo de partida, para no
        # deshacer sus bloques con una vuelta de suavidad L2.
        if vuelta > 0 or (nitida and pesos_desde_inicial):
            salto = np.diff(np.log(modelo))
            inv.setConstraintWeights(pg.Vector(beta / np.sqrt(salto ** 2 + beta ** 2)))
        else:
            pass
        inv.maxIter = max_iter if vuelta == 0 else 8
        nuevo = np.asarray(inv.run(
            d, e, lam=float(lam), startModel=modelo,
            robustData=robusto, isReference=referencia, cType=tipo_restriccion,
            stopAtChi1=False, dPhi=1e-4,
        ), float)
        cambio = float(np.max(np.abs(np.log(nuevo / modelo))))
        modelo = nuevo
        # Reponderación convergida: el modelo ya no cambia (0.3 % en ρ).
        if vuelta >= 2 and cambio < 3e-3:
            break
        else:
            pass
    respuesta = np.asarray(fop.response(modelo), float)
    normalizado = np.log(d / respuesta) / e
    orden = np.argsort(geom.am + geom.bn, kind="stable")
    r = normalizado[orden]
    autocorrelacion = float(np.sum(r[1:] * r[:-1]) / max(np.sum(r * r), 1e-12)) if len(r) > 2 else 0.0
    return {
        "rho": modelo,
        "respuesta": respuesta,
        "autocorrelacion": autocorrelacion,
        "chi2": _chi2(rhoa, respuesta, error_rel),
        "rms": _rms(rhoa, respuesta),
        "rugosidad": rugosidad(modelo),
        "lam": float(lam),
    }


def ajuste_fisico_robusto(
    geom: fw.GeometriaSondeo, rhoa: np.ndarray, error_rel: np.ndarray, z_mediana: np.ndarray,
) -> np.ndarray:
    """Curva que mejor reproduce un medio 1D, insensible a datos aberrantes.

    Inversión suave con reponderación robusta de los datos (IRLS que
    imita la norma L1). La usa el preprocesamiento como referencia
    física para detectar anómalos.
    """
    z_min = max(float(np.min(z_mediana)) / 2.0, 1e-2)
    z_max = float(np.max(z_mediana)) * 2.0
    n = int(np.clip(len(rhoa), 12, 30))
    espesores = capas_logaritmicas(z_min, z_max, n)
    inicial = np.full(n, float(np.median(rhoa)))
    resultado = invertir_occam(geom, rhoa, error_rel, espesores, inicial, lam=20.0,
                               robusto=True, max_iter=15)
    return resultado["respuesta"]


def buscar_lambda(
    geom, rhoa, error_rel, espesores, inicial,
    metodo: str = "optuna", n_trials: int = 15, lam_manual: float = 20.0,
    robusto: bool = False, verificar_cancelacion: Optional[Callable[[], None]] = None,
    nitida: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Elige λ de la inversión multicapa y devuelve el mejor resultado.

    ``metodo``:

    - ``'optuna'``: búsqueda bayesiana (TPE) en log λ ∈ [0.1, 1000]
      minimizando |ln χ²| + 0.1·rugosidad + 2·max(ρ₁ − 0.2, 0), con
      tres semillas (3, 30, 300). ρ₁ es la autocorrelación de los
      residuales normalizados a lo largo de la curva: un χ² ≈ 1 con
      residuales correlacionados (rachas del mismo signo) significa que
      el modelo es demasiado suave y deja estructura sin explicar; el
      término lo penaliza aunque el χ² global parezca correcto.
    - ``'discrepancia'``: rejilla logarítmica; el mayor λ con χ² ≤ 1
      (el modelo más suave que ajusta al nivel del ruido).
    - ``'manual'``: el λ indicado.
    """
    cache: dict[float, dict[str, Any]] = {}

    def evaluar(lam: float) -> dict[str, Any]:
        clave = round(float(lam), 6)
        if clave not in cache:
            if verificar_cancelacion is not None:
                verificar_cancelacion()
            else:
                pass
            cache[clave] = invertir_occam(geom, rhoa, error_rel, espesores, inicial, lam, robusto=robusto,
                                          nitida=nitida)
        else:
            pass
        return cache[clave]

    def objetivo(res: dict[str, Any]) -> float:
        return (abs(np.log(max(res["chi2"], 1e-6))) + 0.1 * res["rugosidad"]
                + 2.0 * max(res.get("autocorrelacion", 0.0) - 0.2, 0.0))

    if metodo == "manual":
        mejor = evaluar(lam_manual)
    else:
        if nitida:
            # Regularización nítida: continuación en λ (de mayor a menor),
            # cada inversión parte del modelo del λ anterior. La función
            # objetivo de MGS no es convexa: arrancar cada λ desde cero da
            # configuraciones de bloques distintas y una búsqueda aleatoria
            # (Optuna) salta entre ellas; la continuación es estable y,
            # por partir de un modelo cercano, cada paso converge rápido.
            # λ se limita a 100: por encima, en la parte profunda (poco
            # sensible) manda la regularización y los contrastes se
            # desplazan hacia abajo.
            previo = np.asarray(inicial, float)
            for lam in np.logspace(2.0, -0.5, 8):
                if verificar_cancelacion is not None:
                    verificar_cancelacion()
                else:
                    pass
                res = invertir_occam(geom, rhoa, error_rel, espesores, previo, lam, robusto=robusto,
                                     nitida=True, pesos_desde_inicial=bool(cache))
                cache[round(float(lam), 6)] = res
                previo = res["rho"]
            if metodo == "discrepancia":
                ajustan = [r for r in cache.values() if r["chi2"] <= 1.0]
                mejor = max(ajustan, key=lambda r: r["lam"]) if ajustan else min(cache.values(), key=objetivo)
            else:
                mejor = min(cache.values(), key=objetivo)
        else:
            pass
    if metodo != "manual" and not nitida:
        if metodo == "discrepancia":
            for lam in np.logspace(-1, 3, 13):
                evaluar(lam)
            ajustan = [r for r in cache.values() if r["chi2"] <= 1.0]
            if ajustan:
                mejor = max(ajustan, key=lambda r: r["lam"])
            else:
                mejor = min(cache.values(), key=objetivo)
        else:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            estudio = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
            )
            for semilla in (3.0, 30.0, 300.0):
                estudio.enqueue_trial({"lam": semilla})

            def _trial(trial):
                return objetivo(evaluar(trial.suggest_float("lam", 0.1, 1000.0, log=True)))

            estudio.optimize(_trial, n_trials=max(int(n_trials), 3))
            mejor = min(cache.values(), key=objetivo)

    pruebas = pd.DataFrame([
        {"lam": r["lam"], "chi2": r["chi2"], "rms": r["rms"], "rugosidad": r["rugosidad"],
         "autocorrelacion": r["autocorrelacion"], "objetivo": objetivo(r)}
        for r in cache.values()
    ]).sort_values("lam").reset_index(drop=True)
    pruebas["seleccionado"] = np.isclose(pruebas["lam"], mejor["lam"])
    return mejor, pruebas


def _gauss_newton_referencia(
    geom, rhoa, error_rel, espesores, referencia: float, lam: float, alfa: float,
    z_escala: float, iteraciones: int = 15,
) -> np.ndarray:
    """Inversión Gauss-Newton con suavidad y atracción a una referencia,
    con los pesos **en profundidad física** de Oldenburg y Li (1999).

    Minimiza::

        ‖W(ln d − ln f(m))‖² + λ [ Σ (z₀/Δc_j)(m_{j+1} − m_j)²
                                   + α Σ (h_j/z₀)(m_j − m_ref)² ]

    con m = ln ρ, Δc la distancia entre centros de capa, h el espesor y
    z₀ una escala de profundidad (mediana de las profundidades de
    investigación). Al ponderar por espesor, las capas profundas —que
    son gruesas— quedan dominadas por el término de referencia salvo
    que los datos las controlen: es lo que hace que el índice R distinga
    lo resuelto de lo no resuelto. Sin esa ponderación, la suavidad
    prolonga hacia abajo el último valor resuelto en ambas inversiones
    y R sale ≈ 0 a cualquier profundidad.
    """
    h = np.asarray(espesores, float)
    n = len(h) + 1
    h_completo = np.concatenate([h, [h[-1] * 1.5]])
    techos = np.concatenate([[0.0], np.cumsum(h)])
    centros = techos + h_completo / 2.0
    peso_suave = z_escala / np.diff(centros)
    D = np.diff(np.eye(n), axis=0) * np.sqrt(peso_suave)[:, None]
    peso_ref = alfa * h_completo / z_escala
    W2 = 1.0 / np.asarray(error_rel, float) ** 2
    ln_d = np.log(np.asarray(rhoa, float))
    m_ref = np.full(n, np.log(referencia))
    m = m_ref.copy()
    R = lam * (D.T @ D + np.diag(peso_ref))

    def objetivo(mm):
        f = fw.respuesta(geom, espesores, np.exp(mm))
        dm = mm - m_ref
        return float(np.sum(W2 * (ln_d - np.log(f)) ** 2) + dm @ R @ dm - lam * dm @ (D.T @ D) @ dm
                     + lam * np.sum((D @ mm) ** 2))

    actual = objetivo(m)
    for _ in range(iteraciones):
        f = fw.respuesta(geom, espesores, np.exp(m))
        J = fw.jacobiano_log(geom, espesores, np.exp(m), incluir_espesores=False)
        A = J.T @ (J * W2[:, None]) + R
        g = J.T @ (W2 * (ln_d - np.log(f))) - lam * (D.T @ (D @ m)) - lam * peso_ref * (m - m_ref)
        paso = np.linalg.solve(A, g)
        t = 1.0
        valor = actual
        candidato = m
        while t > 1e-3:
            candidato = np.clip(m + t * paso, np.log(RHO_MIN), np.log(RHO_MAX))
            valor = objetivo(candidato)
            if valor < actual:
                break
            else:
                t *= 0.5
        if t <= 1e-3:
            break
        else:
            pass
        mejora = actual - valor
        m, actual = candidato, valor
        if mejora < 1e-4 * actual:
            break
        else:
            pass
    return np.exp(m)


def profundidad_investigacion(
    geom, rhoa, error_rel, espesores, lam: float, umbral: float = 0.2,
    z_min_valida: float = 0.0, alfa_relativo: float = 0.05, z_escala: Optional[float] = None,
) -> tuple[float, np.ndarray]:
    """DOI por el índice de Oldenburg y Li (1999).

    Se invierte dos veces con modelos de referencia homogéneos muy
    distintos (×0.2 y ×5 la mediana de la curva). Donde los datos
    controlan el modelo, ambas inversiones coinciden (R ≈ 0); donde no,
    cada una vuelve a su referencia (R ≈ 1)::

        R_j = (ln m1_j − ln m2_j) / (ln ref1 − ln ref2)

    La DOI es la profundidad a partir de la cual R supera el umbral y ya
    no vuelve a bajar de él.
    """
    mediana = float(np.median(rhoa))
    referencias = (mediana * 0.2, mediana * 5.0)
    # Con λ muy grande la suavidad domina también donde hay datos y la DOI
    # se acorta artificialmente: se acota a un rango donde el índice es
    # estable (probado entre 3 y 100 en sondeos sintéticos).
    lam = float(np.clip(lam, 1.0, 30.0))
    z0 = float(z_escala) if z_escala else float(np.sum(espesores)) / 10.0
    modelos = [
        _gauss_newton_referencia(geom, rhoa, error_rel, espesores, ref, lam, alfa_relativo, z0)
        for ref in referencias
    ]
    indice = np.abs(np.log(modelos[0]) - np.log(modelos[1])) / abs(np.log(referencias[0] / referencias[1]))
    techos = np.concatenate([[0.0], np.cumsum(espesores)])
    h_ultimo = float(espesores[-1]) * 1.5
    centros = techos + np.concatenate([espesores, [h_ultimo]]) / 2.0
    doi = float(centros[-1])
    for j in range(len(indice) - 1, -1, -1):
        if indice[j] <= umbral and centros[j] >= z_min_valida:
            if j + 1 < len(indice):
                # Interpolación en log z del cruce del umbral entre j y j+1.
                f = (umbral - indice[j]) / max(indice[j + 1] - indice[j], 1e-9)
                doi = float(np.exp(np.log(centros[j]) + f * (np.log(centros[j + 1]) - np.log(centros[j]))))
            else:
                doi = float(centros[j])
            break
        else:
            pass
    return doi, indice


# ======================================================================
# Dar Zarrouk: reducción de capas
# ======================================================================


def parametros_dar_zarrouk(espesores, resistividades) -> dict[str, float]:
    """S, T, ρL, ρT, λ y ρm de un paquete de capas finitas."""
    h = np.asarray(espesores, float)
    rho = np.asarray(resistividades, float)[: len(h)]
    H = float(h.sum())
    S = float(np.sum(h / rho))
    T = float(np.sum(h * rho))
    rho_l = H / S if S > 0 else np.nan
    rho_t = T / H if H > 0 else np.nan
    return {
        "H": H, "S": S, "T": T, "rho_L": rho_l, "rho_T": rho_t,
        "lambda": float(np.sqrt(rho_t / rho_l)) if rho_l > 0 else np.nan,
        "rho_m": float(np.sqrt(rho_l * rho_t)) if rho_l > 0 else np.nan,
    }


def segmentar(valores: np.ndarray, pesos: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Partición óptima de una serie en ``k`` tramos contiguos.

    Minimiza la suma ponderada de varianzas dentro de cada tramo
    (programación dinámica exacta, O(k·n²)). Es el equivalente 1D, con
    contigüidad, de un agrupamiento k-medias: agrupa capas con
    resistividad parecida sin romper el orden en profundidad.
    """
    v = np.asarray(valores, float)
    w = np.asarray(pesos, float)
    n = len(v)
    k = int(min(k, n))
    s0 = np.concatenate([[0.0], np.cumsum(w)])
    s1 = np.concatenate([[0.0], np.cumsum(w * v)])
    s2 = np.concatenate([[0.0], np.cumsum(w * v * v)])

    def costo(a: int, b: int) -> float:  # tramo [a, b)
        peso = s0[b] - s0[a]
        if peso <= 0:
            return 0.0
        else:
            return float(s2[b] - s2[a] - (s1[b] - s1[a]) ** 2 / peso)

    inf = np.inf
    dp = np.full((k + 1, n + 1), inf)
    corte = np.zeros((k + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0
    for g in range(1, k + 1):
        for b in range(g, n + 1):
            for a in range(g - 1, b):
                valor = dp[g - 1, a] + costo(a, b)
                if valor < dp[g, b]:
                    dp[g, b] = valor
                    corte[g, b] = a
                else:
                    pass
    tramos = []
    b = n
    for g in range(k, 0, -1):
        a = corte[g, b]
        tramos.append((a, b))
        b = a
    return tramos[::-1]


def reducir_dar_zarrouk(
    espesores: np.ndarray, resistividades: np.ndarray, k: int, conservar_espesor: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Reduce un modelo multicapa a ``k`` capas equivalentes.

    Las capas se agrupan por :func:`segmentar` sobre log ρ (pesando cada
    capa por su extensión en log z, que es como decrece la resolución).
    Cada grupo de capas finitas se sustituye por su equivalente de Dar
    Zarrouk::

        T = Σ ρ_i h_i      S = Σ h_i / ρ_i
        h = √(T·S)         ρ = √(T / S)         (reducción clásica)

    o, con ``conservar_espesor``, h = Σ h_i y ρ = √(ρT·ρL). El grupo que
    contiene el semiespacio se convierte en semiespacio con la media
    geométrica ponderada de sus resistividades.
    """
    h = np.asarray(espesores, float)
    rho = np.asarray(resistividades, float)
    bases = np.cumsum(h)
    techos = np.concatenate([[0.0], bases[:-1]])
    techo_ref = np.where(techos > 0, techos, h / 2.0)
    pesos_finitos = np.log((techo_ref + h) / techo_ref)
    pesos = np.concatenate([pesos_finitos, [float(np.median(pesos_finitos))]])
    tramos = segmentar(np.log10(rho), pesos, k)

    nuevos_h: list[float] = []
    nuevos_rho: list[float] = []
    for a, b in tramos:
        if b == len(rho):
            nuevos_rho.append(float(10 ** np.average(np.log10(rho[a:b]), weights=pesos[a:b])))
        else:
            dz = parametros_dar_zarrouk(h[a:b], rho[a:b])
            if conservar_espesor:
                nuevos_h.append(dz["H"])
                nuevos_rho.append(dz["rho_m"])
            else:
                nuevos_h.append(float(np.sqrt(dz["T"] * dz["S"])))
                nuevos_rho.append(float(np.sqrt(dz["T"] / dz["S"])))
    return np.array(nuevos_h), np.array(nuevos_rho), tramos


def invertir_bloques(
    geom, rhoa, error_rel, espesores0, resistividades0, lam: float = 10.0, max_iter: int = 30,
    limites_espesor: tuple[float, float] = (1e-2, 1e4),
    limites_rho: tuple[float, float] = (RHO_MIN, RHO_MAX),
) -> dict[str, Any]:
    """Inversión de Marquardt de un modelo de pocas capas (pyGIMLi).

    Los límites se imponen con transformaciones log acotadas: sin ellos,
    una capa delgada mal resuelta puede escaparse a espesores o
    resistividades absurdas (equivalencia) y arrastrar a las demás. El
    λ inicial alto y decreciente (×0.7 por iteración) hace que las
    primeras iteraciones sean pasos cortos, que es lo que estabiliza a
    Marquardt lejos del mínimo.
    """
    k = len(resistividades0)
    fop = fw.FopBloques(geom, k)
    fop.setRegionProperties(0, limits=limites_espesor, trans="log")
    fop.setRegionProperties(1, limits=limites_rho, trans="log")
    inv = _inversor(fop)
    inv.maxIter = max_iter
    h0 = np.clip(espesores0, limites_espesor[0] * 1.05, limites_espesor[1] * 0.95)
    r0 = np.clip(resistividades0, limites_rho[0] * 1.05, limites_rho[1] * 0.95)
    inicial = np.concatenate([h0, r0])
    modelo = np.asarray(inv.run(
        np.asarray(rhoa, float), np.asarray(error_rel, float),
        lam=lam, lambdaFactor=0.7, startModel=inicial, cType=0,
    ), float)
    respuesta = np.asarray(fop.response(modelo), float)
    chi2_inicial = _chi2(rhoa, fw.respuesta(geom, h0, r0), error_rel)
    chi2 = _chi2(rhoa, respuesta, error_rel)
    if not np.isfinite(chi2) or chi2 > chi2_inicial:
        # Marquardt empeoró el arranque: se conserva el modelo inicial.
        modelo, respuesta, chi2 = inicial, fw.respuesta(geom, h0, r0), chi2_inicial
    else:
        pass
    return {
        "espesores": modelo[: k - 1], "rho": modelo[k - 1:], "respuesta": respuesta,
        "chi2": chi2, "rms": _rms(rhoa, respuesta),
    }


def covarianza_bloques(geom, espesores, resistividades, error_rel, amortiguamiento: float = 1e-3) -> dict[str, np.ndarray]:
    """Incertidumbre linealizada (1σ, en ln) de h, ρ, S y T por capa.

    C = (Jᵀ W² J + εI)⁻¹ con J = ∂ln ρa/∂ln p y W = 1/error. Las
    correlaciones entre h y ρ de una misma capa dan la incertidumbre de
    S = h/ρ y T = h·ρ, que es lo que el principio de equivalencia deja
    bien determinado aunque h y ρ por separado no lo estén.
    """
    J = fw.jacobiano_log(geom, espesores, resistividades, incluir_espesores=True)
    W = 1.0 / np.asarray(error_rel, float)
    JW = J * W[:, None]
    C = np.linalg.pinv(JW.T @ JW + amortiguamiento * np.eye(J.shape[1]))
    n_h = len(espesores)
    var = np.diag(C)
    sig_h = np.sqrt(np.maximum(var[:n_h], 0))
    sig_rho = np.sqrt(np.maximum(var[n_h:], 0))
    sig_s = np.zeros(n_h)
    sig_t = np.zeros(n_h)
    for i in range(n_h):
        c_hr = C[i, n_h + i]
        sig_s[i] = np.sqrt(max(var[i] + var[n_h + i] - 2 * c_hr, 0))
        sig_t[i] = np.sqrt(max(var[i] + var[n_h + i] + 2 * c_hr, 0))
    return {"sigma_h": sig_h, "sigma_rho": sig_rho, "sigma_S": sig_s, "sigma_T": sig_t}


def _resolucion(sig_h: float, sig_rho: float, sig_s: float, sig_t: float) -> str:
    """Qué parámetro de la capa está resuelto (umbral 25 % en 1σ)."""
    bien = np.log(1.25)
    if sig_h <= bien and sig_rho <= bien:
        return "h y ρ"
    else:
        pass
    if sig_s <= bien and sig_s < sig_t:
        return "S (equivalencia)"
    else:
        pass
    if sig_t <= bien:
        return "T (equivalencia)"
    else:
        pass
    return "poco resuelta"


def _arranques_por_division(previo: dict[str, Any], z_fondo: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Arranques de k capas obtenidos al dividir una capa del mejor
    modelo de k−1 capas (las dos más gruesas en log z y el semiespacio).

    Complementan a los arranques de Dar Zarrouk: si el modelo suave
    escondió una capa (lo hace con capas delgadas de alto contraste),
    la división la introduce donde el modelo de k−1 capas tiene más
    espacio para ella, con contraste de ambos signos.
    """
    h = np.asarray(previo["espesores"], float)
    rho = np.asarray(previo["rho"], float)
    techos = np.concatenate([[0.0], np.cumsum(h)])
    extension = np.log((techos[1:] + h) / np.maximum(techos[:-1], h / 2.0))
    candidatas = list(np.argsort(extension)[::-1][:2])
    arranques = []
    for j in candidatas:
        # El factor 1 reproduce exactamente la respuesta del modelo de k−1
        # capas: garantiza que añadir una capa nunca empeore el ajuste por
        # caer en un mínimo local peor.
        for factor in (1.0, 0.3, 3.0):
            nuevos_h = np.concatenate([h[:j], [h[j] / 2.0, h[j] / 2.0], h[j + 1:]])
            nuevos_rho = np.concatenate([rho[:j + 1], [rho[j] * factor], rho[j + 1:]])
            arranques.append((nuevos_h, nuevos_rho))
    profundidad = techos[-1] if len(h) else 1.0
    nueva_base = min(max(profundidad * 1.6, profundidad + 1.0), z_fondo)
    for factor in (0.3, 3.0):
        arranques.append((np.concatenate([h, [max(nueva_base - profundidad, 0.5)]]),
                          np.concatenate([rho[:-1], [rho[-1] * factor, rho[-1]]])))
    return arranques


@dataclass
class RestriccionEspesor:
    """Espesor mínimo resoluble por un SEV.

    - La primera capa no puede ser más delgada que ``h_min_1`` (la
      profundidad de investigación de la apertura más corta: nada más
      somero está muestreado).
    - Cada capa siguiente debe tener al menos ``fraccion`` × la profundidad
      de su techo: la resolución vertical del SEV decae con la profundidad
      y una capa más delgada que eso no se distingue de sus vecinas (queda
      en equivalencia S o T). ``fraccion = 0`` desactiva esta parte.
    """

    h_min_1: float
    fraccion: float = 0.25

    def minimos(self, espesores) -> np.ndarray:
        h = np.asarray(espesores, float)
        techos = np.concatenate([[0.0], np.cumsum(h)[:-1]])
        return np.maximum(np.where(techos > 0, self.fraccion * techos, self.h_min_1), 1e-9)

    def cumple(self, espesores, tolerancia: float = 1e-3) -> bool:
        h = np.asarray(espesores, float)
        return bool(len(h) == 0 or np.all(h >= self.minimos(h) * (1 - tolerancia)))

    # Reparametrización: ln z_1 = ln h_min_1 + w_1 y
    # ln z_i = ln z_{i−1} + ln(1 + fraccion) + w_i, con w ≥ 0. Así cualquier
    # w ≥ 0 cumple la restricción y el optimizador acotado no puede violarla.
    def a_w(self, espesores) -> np.ndarray:
        z = np.cumsum(np.asarray(espesores, float))
        c = self._c(len(z))
        lnz = np.log(np.maximum(z, 1e-12))
        w = np.diff(np.concatenate([[np.log(self.h_min_1)], lnz])) - c
        return np.maximum(w, 0.0)

    def de_w(self, w) -> np.ndarray:
        lnz = np.log(self.h_min_1) + np.cumsum(np.asarray(w, float) + self._c(len(w)))
        return np.diff(np.concatenate([[0.0], np.exp(lnz)]))

    def _c(self, n: int) -> np.ndarray:
        return np.concatenate([[0.0], np.full(max(n - 1, 0), np.log1p(self.fraccion))])

    def proyectar(self, espesores) -> np.ndarray:
        """El modelo factible más cercano (en log z) a ``espesores``."""
        return self.de_w(self.a_w(espesores))


def ajustar_bloques_lm(
    geom, rhoa, error_rel, espesores0, resistividades0,
    limites_espesor: tuple[float, float], limites_rho: tuple[float, float], max_nfev: int = 50,
    restriccion: Optional[RestriccionEspesor] = None,
) -> dict[str, Any]:
    """Levenberg–Marquardt acotado (SciPy, región de confianza) para un
    modelo de pocas capas, con el jacobiano analítico en log.

    Es ~10 veces más rápido que :func:`invertir_bloques` porque no pasa
    por la maquinaria de pyGIMLi; se usa para explorar muchos arranques y
    luego el mejor se pule con pyGIMLi.
    """
    from scipy.optimize import least_squares

    log_obs = np.log(np.asarray(rhoa, float))
    w = 1.0 / np.asarray(error_rel, float)
    nh = len(espesores0)
    if restriccion is not None and nh > 0:
        return _ajustar_bloques_restringido(geom, log_obs, w, rhoa, error_rel, espesores0, resistividades0,
                                            limites_espesor, limites_rho, max_nfev, restriccion)
    else:
        pass
    lh = np.log(limites_espesor)
    lr = np.log(limites_rho)
    x0 = np.concatenate([
        np.clip(np.log(np.asarray(espesores0, float)), lh[0] + 0.01, lh[1] - 0.01),
        np.clip(np.log(np.asarray(resistividades0, float)), lr[0] + 0.01, lr[1] - 0.01),
    ])
    inf = np.concatenate([np.full(nh, lh[0]), np.full(len(resistividades0), lr[0])])
    sup = np.concatenate([np.full(nh, lh[1]), np.full(len(resistividades0), lr[1])])

    def residuos(x):
        return (log_obs - np.log(fw.respuesta(geom, np.exp(x[:nh]), np.exp(x[nh:])))) * w

    def jacobiano(x):
        return -fw.jacobiano_log(geom, np.exp(x[:nh]), np.exp(x[nh:]), incluir_espesores=True) * w[:, None]

    sol = least_squares(residuos, x0, jac=jacobiano, bounds=(inf, sup), method="trf",
                        max_nfev=max_nfev, xtol=1e-10, ftol=1e-12)
    h = np.exp(sol.x[:nh])
    rho = np.exp(sol.x[nh:])
    respuesta = fw.respuesta(geom, h, rho)
    return {"espesores": h, "rho": rho, "respuesta": respuesta,
            "chi2": _chi2(rhoa, respuesta, error_rel), "rms": _rms(rhoa, respuesta)}


def _ajustar_bloques_restringido(geom, log_obs, w, rhoa, error_rel, espesores0, resistividades0,
                                 limites_espesor, limites_rho, max_nfev, restriccion):
    """Como :func:`ajustar_bloques_lm`, con el espesor mínimo impuesto por
    reparametrización (ver :class:`RestriccionEspesor`)."""
    from scipy.optimize import least_squares

    nh = len(espesores0)
    lr = np.log(limites_rho)
    w_max = max(np.log(limites_espesor[1] / restriccion.h_min_1), 1.0)
    w0 = np.clip(restriccion.a_w(np.asarray(espesores0, float)), 1e-6, w_max - 1e-3)
    x0 = np.concatenate([w0, np.clip(np.log(np.asarray(resistividades0, float)), lr[0] + 0.01, lr[1] - 0.01)])
    inf = np.concatenate([np.zeros(nh), np.full(len(resistividades0), lr[0])])
    sup = np.concatenate([np.full(nh, w_max), np.full(len(resistividades0), lr[1])])

    def residuos(x):
        return (log_obs - np.log(fw.respuesta(geom, restriccion.de_w(x[:nh]), np.exp(x[nh:])))) * w

    def jacobiano(x):
        h = restriccion.de_w(x[:nh])
        z = np.cumsum(h)
        J = fw.jacobiano_log(geom, h, np.exp(x[nh:]), incluir_espesores=True)
        Jh = J[:, :nh]                                  # ∂lnρa/∂ln h_i
        # ∂lnρa/∂ln z_k = J_k·z_k/h_k − J_{k+1}·z_k/h_{k+1}
        Jz = Jh * (z / h)[None, :]
        Jz[:, :-1] -= Jh[:, 1:] * (z[:-1] / h[1:])[None, :]
        # ∂ln z_k/∂w_j = 1 si j ≤ k  →  suma acumulada inversa
        Jw = np.cumsum(Jz[:, ::-1], axis=1)[:, ::-1]
        return -np.hstack([Jw, J[:, nh:]]) * w[:, None]

    sol = least_squares(residuos, x0, jac=jacobiano, bounds=(inf, sup), method="trf",
                        max_nfev=max_nfev, xtol=1e-10, ftol=1e-12)
    h = restriccion.de_w(sol.x[:nh])
    rho = np.exp(sol.x[nh:])
    respuesta = fw.respuesta(geom, h, rho)
    return {"espesores": h, "rho": rho, "respuesta": respuesta,
            "chi2": _chi2(rhoa, respuesta, error_rel), "rms": _rms(rhoa, respuesta)}


def arranques_globales(
    k: int, espesores_mc, rho_mc, z_lim: tuple[float, float], n: int, semilla: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Arranques de ``k`` capas que exploran la posición de las interfaces.

    Las k−1 profundidades se muestrean por hipercubo latino (uniforme en
    log z dentro de ``z_lim``). La resistividad inicial de cada capa es la
    media geométrica del multicapa dentro de ella, así cada arranque ya
    es coherente con los datos y Marquardt solo tiene que mover las
    interfaces. Con eso la búsqueda escapa de los mínimos locales en que
    caen los arranques de Dar Zarrouk cuando el multicapa reparte un
    contraste entre varias capas.
    """
    from scipy.stats import qmc

    if k < 2 or n < 1:
        return []
    else:
        pass
    h_mc = np.asarray(espesores_mc, float)
    r_mc = np.asarray(rho_mc, float)
    techos_mc = np.concatenate([[0.0], np.cumsum(h_mc)])
    log_lo, log_hi = np.log(z_lim[0]), np.log(z_lim[1])
    muestras = qmc.LatinHypercube(d=k - 1, seed=semilla).random(n)
    arranques = []
    for u in muestras:
        z = np.sort(np.exp(log_lo + u * (log_hi - log_lo)))
        bordes = np.concatenate([[0.0], z, [np.inf]])
        rho = np.empty(k)
        for i in range(k):
            dentro = (techos_mc >= bordes[i]) & (techos_mc < bordes[i + 1])
            if dentro.any():
                rho[i] = float(np.exp(np.mean(np.log(r_mc[dentro]))))
            else:
                rho[i] = float(r_mc[max(np.searchsorted(techos_mc, bordes[i]) - 1, 0)])
        arranques.append((np.diff(np.concatenate([[0.0], z])), rho))
    return arranques


def distancia_al_multicapa(espesores, resistividades, espesores_mc, rho_mc, z_tope: float,
                           sensibilidad: Optional[np.ndarray] = None) -> float:
    """Diferencia RMS (en log10 ρ) entre un modelo de capas y el multicapa,
    medida sobre las celdas del multicapa hasta ``z_tope``.

    Con ``sensibilidad`` (norma de la columna del jacobiano ponderado de
    cada celda), cada celda pesa lo que los datos la ven: la parte
    profunda, donde el multicapa lo decide la regularización y no los
    datos, casi no cuenta. Sin ella, se pesa por la extensión en log z."""
    h_mc = np.asarray(espesores_mc, float)
    techos = np.concatenate([[0.0], np.cumsum(h_mc)])
    bases = np.concatenate([techos[1:], [techos[-1] * 1.5]])
    centros = np.sqrt(np.maximum(techos, h_mc[0] / 2.0) * bases)
    dentro = centros <= z_tope
    if sensibilidad is not None and len(sensibilidad) == len(centros):
        pesos = np.asarray(sensibilidad, float)[dentro]
    else:
        pesos = np.log(bases / np.maximum(techos, h_mc[0] / 2.0))[dentro]
    bloque = fw.muestrear_modelo(np.asarray(espesores, float), np.asarray(resistividades, float), centros[dentro])
    dif = np.log10(bloque) - np.log10(np.asarray(rho_mc, float)[dentro])
    return float(np.sqrt(np.sum(pesos * dif ** 2) / max(np.sum(pesos), 1e-12)))


def _elegir_equivalente(ajustes: list[dict[str, Any]], n_datos: int, espesores_mc, rho_mc,
                        z_tope: float, tolerancia_phi: float = 0.25,
                        sensibilidad: Optional[np.ndarray] = None,
                        criterio_desempate: str = "estructura") -> dict[str, Any]:
    """Entre los ajustes de k capas que explican los datos igual de bien
    (φ dentro de ``tolerancia_phi`` = 0.25 del mejor, ≈ medio σ), elige el de
    mínima estructura (contrastes menos extremos; por omisión) o el más
    parecido al multicapa.

    Es la forma de resolver la **equivalencia**: una capa conductora
    delgada y muy conductora y otra más gruesa y menos conductora con la
    misma S producen la misma curva; los datos no pueden distinguirlas y
    el multicapa regularizado (mínima estructura) decide.
    """
    phi_min = min(a["chi2"] for a in ajustes) * n_datos
    equivalentes = [a for a in ajustes if a["chi2"] * n_datos <= phi_min + tolerancia_phi]
    for a in equivalentes:
        a["distancia_multicapa"] = distancia_al_multicapa(a["espesores"], a["rho"], espesores_mc, rho_mc, z_tope,
                                                          sensibilidad)
        # Estructura: suma de los saltos² de ln ρ entre capas contiguas.
        a["estructura"] = float(np.sum(np.diff(np.log(np.asarray(a["rho"], float))) ** 2))
    if criterio_desempate == "estructura":
        # Mínima estructura: entre variantes equivalentes (misma S o T), la de
        # contrastes menos extremos — la más conservadora geológicamente.
        elegido = min(equivalentes, key=lambda a: a["estructura"])
    else:
        elegido = min(equivalentes, key=lambda a: a["distancia_multicapa"])
    elegido["n_equivalentes"] = len(equivalentes)
    return elegido


#: Contraste mínimo entre capas contiguas de la reducción de Dar Zarrouk
#: para considerarlas capas distintas (factor 1.5 en ρ).
CONTRASTE_ESTRUCTURAL = 1.5
#: Margen de criterio de información con el que se conserva el número de
#: capas que muestran Zohdy/multicapa: solo se descarta si los datos dan
#: evidencia fuerte en contra (ΔBIC > 6, escala de Kass y Raftery).
MARGEN_ESTRUCTURAL = 6.0


def k_estructural(espesores, resistividades, ks: list[int], z_fondo: float,
                  restriccion: Optional[RestriccionEspesor] = None,
                  contraste: float = CONTRASTE_ESTRUCTURAL) -> Optional[int]:
    """Número de capas que el modelo de referencia (Zohdy o multicapa)
    sostiene: el mayor k cuya reducción de Dar Zarrouk tiene todas sus capas
    distintas (contraste ≥ ``contraste`` con la vecina), dentro de la
    profundidad viable y con espesores resolubles."""
    h_f = np.asarray(espesores, float)
    r_f = np.asarray(resistividades, float)
    # Solo cuenta lo que hay hasta la profundidad viable.
    techos = np.concatenate([[0.0], np.cumsum(h_f)])
    n_dentro = int(np.sum(techos < z_fondo))
    h_f = h_f[: max(n_dentro - 1, 1)]
    r_f = r_f[: max(n_dentro, 2)]
    for k in sorted(ks, reverse=True):
        if len(r_f) <= k:
            continue
        else:
            pass
        h, r, _ = reducir_dar_zarrouk(h_f, r_f, k, conservar_espesor=False)
        cocientes = np.exp(np.abs(np.diff(np.log(r))))
        resoluble = restriccion is None or restriccion.cumple(h, tolerancia=0.5)
        if np.all(cocientes >= contraste) and resoluble:
            return int(k)
        else:
            pass
    return None


def modelo_por_profundidad(k: int, z_e: np.ndarray, rhoa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modelo inicial de k capas que depende solo de la profundidad.

    Las k − 1 interfaces dividen de forma logarítmicamente uniforme el
    intervalo de profundidades de investigación de Edwards
    [z_e mín, z_e máx]; la resistividad inicial de cada capa es la media
    geométrica de las ρa cuya z_e cae dentro de ella (si ninguna cae, la
    del dato más cercano).
    """
    z_e = np.asarray(z_e, float)
    rhoa = np.asarray(rhoa, float)
    bordes = np.exp(np.linspace(np.log(z_e.min()), np.log(z_e.max()), k + 1))[1:k]
    limites = np.concatenate([[0.0], bordes, [np.inf]])
    rho = np.empty(k)
    for i in range(k):
        dentro = (z_e >= limites[i]) & (z_e < limites[i + 1])
        if dentro.any():
            rho[i] = float(np.exp(np.mean(np.log(rhoa[dentro]))))
        else:
            centro = np.sqrt(max(limites[i], z_e.min()) * min(limites[i + 1], z_e.max()))
            rho[i] = float(rhoa[np.argmin(np.abs(np.log(z_e / centro)))])
    return np.diff(np.concatenate([[0.0], bordes])), rho


def invertir_bloques_optuna(
    geom, rhoa, error_rel, fuentes: dict[str, tuple[np.ndarray, np.ndarray]],
    espesores_mc, rho_mc, *, k_max: int = 8, k_fijo: int = 0, criterio: str = "BIC",
    n_trials: int = 40, restriccion: Optional[RestriccionEspesor] = None,
    z_max_datos: Optional[float] = None, z_min_datos: Optional[float] = None, z_viable: Optional[float] = None,
    modo: str = "dar_zarrouk", z_e: Optional[np.ndarray] = None, verificar_cancelacion=None,
    semilla: int = 42, n_exploratorios: int = 10, k_desde_estructura: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[str]]:
    """Pasos 3 y 4: modelo de bloques inicial y su inversión optimizada.

    **Paso 3 (modo ``'dar_zarrouk'``).** Cada fuente (Zohdy y multicapa) se
    reduce por Dar Zarrouk a 2…k capas (variantes clásica y con espesor
    conservado); cada reducción se ajusta brevemente y se queda la fuente
    cuyo mejor modelo tiene el menor criterio de información.
    (Modo ``'profundidad'``: el arranque es :func:`modelo_por_profundidad`
    con el k del usuario.)

    **Paso 4.** Optuna (TPE) explora los hiperparámetros de la inversión de
    bloques desde ese arranque: número de capas k, variante del arranque
    (Dar Zarrouk clásico / espesor conservado de la fuente elegida, de la
    otra fuente, o posiciones de interfaces exploratorias), escala de las
    profundidades del arranque y "contraste" del arranque
    (ρ = ρ̄·(ρ/ρ̄)^γ, que recorre la equivalencia S/T entre variantes más o
    menos extremas). Cada prueba se invierte por Levenberg–Marquardt con el
    espesor mínimo impuesto y se califica con BIC (o AIC). Al final: k por
    criterio de información con parsimonia, desempate entre modelos
    equivalentes, pulido con pyGIMLi y fusión de capas casi iguales.

    Returns
    -------
    (mejor, tabla_k, pruebas, proceso)
    """
    import optuna

    N = len(rhoa)
    rhoa = np.asarray(rhoa, float)
    error_rel = np.asarray(error_rel, float)
    k_lim = int(min(k_max, (N - 1) // 2))
    if k_lim < 2:
        raise InversionSEVError(f"Con {N} datos no se puede ajustar un modelo de 2 capas o más.")
    else:
        pass
    techos_mc = np.cumsum(espesores_mc)
    z_fondo = float(z_viable) if z_viable else (2.0 * float(z_max_datos) if z_max_datos else float(techos_mc[-1]))
    lim_h = (max(float(espesores_mc[0]) / 20.0, 1e-3), z_fondo)
    lim_r = (max(float(np.min(rhoa)) / 20.0, RHO_MIN), min(float(np.max(rhoa)) * 20.0, RHO_MAX))
    z_sup = float(z_viable) if z_viable else (1.5 * float(z_max_datos) if z_max_datos else float(techos_mc[-1]) / 2)
    z_inf = 0.5 * float(z_min_datos) if z_min_datos else float(espesores_mc[0])
    z_busqueda = (max(z_inf, lim_h[0] * 2.0), max(z_sup, 2.0 * z_inf))
    proceso: list[str] = []
    clave_ic = "AIC" if criterio == "AIC" else "BIC"

    def criterio_de(chi2: float, k: int) -> float:
        p = 2 * k - 1
        phi = chi2 * N
        return phi + p * np.log(N) if clave_ic == "BIC" else phi + 2 * p + 2 * p * (p + 1) / max(N - p - 1, 1)

    if k_fijo and 2 <= int(k_fijo) <= max(k_lim, 2):
        ks = [int(k_fijo)]
    else:
        ks = list(range(2, k_lim + 1))

    # ----- Paso 3: arranques por Dar Zarrouk y elección de la fuente -----
    evaluacion = []
    if modo == "profundidad":
        k_p = int(k_fijo) if k_fijo else min(5, k_lim)
        ks = [k_p]
        h0, r0 = modelo_por_profundidad(k_p, z_e, rhoa)
        r = ajustar_bloques_lm(geom, rhoa, error_rel, h0, r0, lim_h, lim_r, max_nfev=8, restriccion=restriccion)
        evaluacion.append({"fuente": "profundidad", "k": k_p, "variante": "log-uniforme en z_e",
                           "chi2": r["chi2"], clave_ic: criterio_de(r["chi2"], k_p)})
        fuente = "profundidad"
        proceso.append(
            f"3. Modelo de bloques por profundidad: {k_p} capas con interfaces log-uniformes entre z_e mín "
            f"({np.min(z_e):.2f} m) y z_e máx ({np.max(z_e):.1f} m); ρ inicial = media geométrica de las ρa de "
            f"cada intervalo. Interfaces iniciales: {', '.join(f'{v:.2f}' for v in np.cumsum(h0))} m."
        )
    else:
        for nombre, (h_f, r_f) in fuentes.items():
            for k in ks:
                if len(r_f) <= k:
                    continue
                else:
                    pass
                for conservar in (False, True):
                    h0, r0, _ = reducir_dar_zarrouk(h_f, r_f, k, conservar_espesor=conservar)
                    try:
                        r = ajustar_bloques_lm(geom, rhoa, error_rel, h0, r0, lim_h, lim_r, max_nfev=8,
                                               restriccion=restriccion)
                    except Exception:
                        continue
                    evaluacion.append({"fuente": nombre, "k": k,
                                       "variante": "espesor conservado" if conservar else "clásica",
                                       "chi2": r["chi2"], clave_ic: criterio_de(r["chi2"], k)})
        tabla_eval = pd.DataFrame(evaluacion)
        mejores = tabla_eval.groupby("fuente")[clave_ic].min().sort_values()
        fuente = str(mejores.index[0])
        proceso.append(
            "3. Modelo de bloques por Dar Zarrouk: se redujeron "
            + " y ".join(fuentes) + f" a {ks[0]}–{ks[-1]} capas (variantes clásica y con espesor conservado) y "
            "se ajustó brevemente cada reducción. Mejor "
            + ", ".join(f"{f}: {clave_ic} {v:.1f}" for f, v in mejores.items())
            + f" → se usa {fuente}."
        )
    otra = [f for f in fuentes if f != fuente]
    # Número de capas que sostiene la fuente elegida (Dar Zarrouk con
    # contraste ≥ 1.5 entre vecinas, dentro de la profundidad viable).
    k_dz = None
    if modo != "profundidad" and k_desde_estructura and not k_fijo:
        k_dz = k_estructural(fuentes[fuente][0], fuentes[fuente][1], ks, z_fondo, restriccion)
        if k_dz is not None:
            h_dz, r_dz, _ = reducir_dar_zarrouk(fuentes[fuente][0], fuentes[fuente][1], k_dz, False)
            proceso[-1] += (f" Estructura de {fuente} por Dar Zarrouk: {k_dz} capas distintas (contraste ≥ "
                            f"{CONTRASTE_ESTRUCTURAL:g} entre vecinas): interfaces "
                            + ", ".join(f"{v:.2f}" for v in np.cumsum(h_dz)) + " m; ρ "
                            + ", ".join(f"{v:.3g}" for v in r_dz) + " Ω·m.")
        else:
            pass
    else:
        pass

    # ----- Paso 4: Optuna sobre los hiperparámetros de la inversión -----
    arranques_base: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for k in ks:
        if modo == "profundidad":
            arranques_base[(k, "profundidad")] = modelo_por_profundidad(k, z_e, rhoa)
        else:
            h_f, r_f = fuentes[fuente]
            if len(r_f) > k:
                arranques_base[(k, "dz_clasica")] = reducir_dar_zarrouk(h_f, r_f, k, False)[:2]
                arranques_base[(k, "dz_espesor")] = reducir_dar_zarrouk(h_f, r_f, k, True)[:2]
            else:
                pass
            for o in otra:
                h_o, r_o = fuentes[o]
                if len(r_o) > k:
                    arranques_base[(k, "otra_fuente")] = reducir_dar_zarrouk(h_o, r_o, k, False)[:2]
                else:
                    pass
        for j, (h0, r0) in enumerate(arranques_globales(k, espesores_mc, rho_mc, z_busqueda, n_exploratorios,
                                                         semilla=k)):
            arranques_base[(k, f"exploratorio_{j}")] = (h0, r0)
    # Divisiones del mejor modelo de k−1 capas (se construyen al evaluar,
    # cuando ya existe el de k−1): introducen una capa nueva donde el modelo
    # anterior tiene más espacio, con contraste de ambos signos.
    n_division = 8
    divisiones = [f"division_{j}" for j in range(n_division)] if modo != "profundidad" else []
    variantes = sorted({v for _, v in arranques_base} | set(divisiones))

    cache: dict[tuple, dict[str, Any]] = {}

    def base_de(k: int, variante: str):
        if variante.startswith("division_"):
            previos = [r for r in cache.values() if r.get("k") == k - 1 and np.isfinite(r.get("chi2", np.inf))]
            if not previos:
                return None
            else:
                pass
            mejor_prev = min(previos, key=lambda r: r["chi2"])
            opciones = _arranques_por_division(mejor_prev, z_fondo)
            j = int(variante.split("_")[1])
            return opciones[j] if j < len(opciones) else None
        else:
            return arranques_base.get((k, variante))

    def evaluar(k: int, variante: str, escala: float, gamma: float) -> dict[str, Any]:
        clave = (k, variante, round(escala, 4), round(gamma, 4))
        if clave in cache:
            return cache[clave]
        else:
            pass
        if verificar_cancelacion is not None:
            verificar_cancelacion()
        else:
            pass
        base = base_de(k, variante)
        if base is None:
            res = {"chi2": np.inf, "k": k}
        else:
            h0 = np.asarray(base[0], float) * escala
            r0 = np.asarray(base[1], float)
            lr = np.log(r0)
            r0 = np.exp(lr.mean() + gamma * (lr - lr.mean()))
            if restriccion is not None:
                h0 = restriccion.proyectar(h0)
            else:
                pass
            try:
                res = ajustar_bloques_lm(geom, rhoa, error_rel, h0, r0, lim_h, lim_r, max_nfev=50,
                                         restriccion=restriccion)
            except Exception:
                res = {"chi2": np.inf}
            res["k"] = k
        res.update({"arranque": variante, "escala": escala, "gamma": gamma})
        res[clave_ic] = criterio_de(res["chi2"], k) if np.isfinite(res["chi2"]) else np.inf
        cache[clave] = res
        return res

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla))
    # Cobertura mínima: todos los arranques base de cada k se evalúan tal
    # cual (escala 1, γ 1); después Optuna afina con n_trials pruebas más.
    for k in ks:
        for v in sorted(v for (kk, v) in arranques_base if kk == k):
            estudio.enqueue_trial({"k": k, "variante": v, "escala": 1.0, "gamma": 1.0})
        if k > ks[0]:
            for v in divisiones:
                estudio.enqueue_trial({"k": k, "variante": v, "escala": 1.0, "gamma": 1.0})
        else:
            pass

    def objetivo(trial):
        k = trial.suggest_int("k", ks[0], ks[-1])
        variante = trial.suggest_categorical("variante", variantes)
        escala = trial.suggest_float("escala", 0.6, 1.6, log=True)
        gamma = trial.suggest_float("gamma", 0.5, 1.5)
        return float(evaluar(k, variante, escala, gamma)[clave_ic])

    total = len(estudio.get_trials()) + max(int(n_trials), 5)
    estudio.optimize(objetivo, n_trials=total)
    ajustes = [r for r in cache.values() if np.isfinite(r.get("chi2", np.inf))]
    if not ajustes:
        raise InversionSEVError("Ninguna inversión por bloques convergió.")
    else:
        pass
    # Refinamiento: las 5 mejores pruebas de cada k se llevan a convergencia
    # completa (cada prueba de Optuna se limita a 50 evaluaciones).
    refinados = []
    for k in sorted({r["k"] for r in ajustes}):
        for r in sorted((a for a in ajustes if a["k"] == k), key=lambda a: a["chi2"])[:5]:
            try:
                f = ajustar_bloques_lm(geom, rhoa, error_rel, r["espesores"], r["rho"], lim_h, lim_r,
                                       max_nfev=200, restriccion=restriccion)
            except Exception:
                continue
            if np.isfinite(f["chi2"]) and f["chi2"] <= r["chi2"]:
                f.update({"k": k, "arranque": r["arranque"] + " (refinado)", "escala": r["escala"],
                          "gamma": r["gamma"]})
                f[clave_ic] = criterio_de(f["chi2"], k)
                refinados.append(f)
            else:
                pass
    ajustes = ajustes + refinados

    # Mejor por k, k por criterio con parsimonia (ΔIC < 2 → el más simple).
    # Desempate entre modelos equivalentes: el más parecido a la fuente
    # elegida en el paso 3 (Zohdy o multicapa), con cada celda pesada por
    # la sensibilidad de los datos a ella.
    if modo != "profundidad":
        h_ref, r_ref = (np.asarray(v, float) for v in fuentes[fuente])
    else:
        h_ref, r_ref = np.asarray(espesores_mc, float), np.asarray(rho_mc, float)
    try:
        J_mc = fw.jacobiano_log(geom, h_ref, r_ref, incluir_espesores=False) / error_rel[:, None]
        sens_mc = np.sqrt(np.sum(J_mc ** 2, axis=0))
    except Exception:
        sens_mc = None
    candidatos = []
    filas = []
    for k in sorted({r["k"] for r in ajustes}):
        de_k = [r for r in ajustes if r["k"] == k]
        mejor_k = _elegir_equivalente(de_k, N, h_ref, r_ref, z_busqueda[1], sensibilidad=sens_mc)
        mejor_k = dict(mejor_k)
        mejor_k["k"] = k
        mejor_k["BIC"] = criterio_de(mejor_k["chi2"], k) if clave_ic == "BIC" else mejor_k.get("BIC", np.nan)
        p = 2 * k - 1
        mejor_k["BIC"] = mejor_k["chi2"] * N + p * np.log(N)
        mejor_k["AIC"] = mejor_k["chi2"] * N + 2 * p + 2 * p * (p + 1) / max(N - p - 1, 1)
        candidatos.append(mejor_k)
        filas.append({"k": k, "parametros": p, "chi2": mejor_k["chi2"], "rms": mejor_k["rms"],
                      "BIC": mejor_k["BIC"], "AIC": mejor_k["AIC"],
                      "arranque": f"{mejor_k['arranque']} (escala {mejor_k['escala']:.2f}, γ {mejor_k['gamma']:.2f})",
                      "pruebas": len(de_k),
                      "interfaces_m": ", ".join(f"{v:.3g}" for v in np.cumsum(mejor_k["espesores"])),
                      "resistividades": ", ".join(f"{v:.3g}" for v in mejor_k["rho"])})
    minimo = min(c[clave_ic] for c in candidatos)
    elegido = min((c for c in candidatos if c[clave_ic] <= minimo + 2.0), key=lambda c: c["k"])
    nota_k = f"k por {clave_ic} con parsimonia (ΔIC < 2 → el más simple)"
    if k_dz is not None:
        # Se conserva la estructura de la fuente: el mayor k ≤ k_dz cuyo
        # criterio no está en desventaja fuerte (ΔIC ≤ 6) frente al mejor.
        # Además, las capas añadidas deben mejorar el ajuste como lo exige el
        # AIC (Δφ ≥ 4 por capa: 2 por parámetro) o reducir χ² a la mitad; si no,
        # son escalones de la suavidad del multicapa, no capas que los datos
        # vean.
        phi_base = elegido["chi2"] * N
        aceptables = [c for c in candidatos if c["k"] <= k_dz and c[clave_ic] <= minimo + MARGEN_ESTRUCTURAL
                      and (c["k"] <= elegido["k"] or phi_base - c["chi2"] * N >= 4.0 * (c["k"] - elegido["k"])
                           or c["chi2"] <= 0.5 * elegido["chi2"])]
        if aceptables:
            estructural = max(aceptables, key=lambda c: c["k"])
            if estructural["k"] > elegido["k"]:
                rechazo = ("" if estructural["k"] == k_dz else
                           f"; más capas (hasta {k_dz}) no mejoran el ajuste lo suficiente")
                nota_k = (f"k = {estructural['k']} por la estructura de {fuente} ({k_dz} capas distintas por Dar "
                          f"Zarrouk): el {clave_ic} prefería {elegido['k']}, pero la diferencia "
                          f"({estructural[clave_ic] - minimo:.1f}) no es evidencia fuerte en contra "
                          f"(≤ {MARGEN_ESTRUCTURAL:g}){rechazo}")
                elegido = estructural
            else:
                nota_k = (f"k = {elegido['k']} por {clave_ic} con parsimonia, compatible con la estructura de "
                          f"{fuente} ({k_dz} capas distintas por Dar Zarrouk)")
        else:
            pass
    else:
        pass
    # Pulido con pyGIMLi (solo si cumple el espesor mínimo y mejora Δφ > 1).
    try:
        pulido = invertir_bloques(geom, rhoa, error_rel, elegido["espesores"], elegido["rho"], lam=1.0,
                                  limites_espesor=lim_h, limites_rho=lim_r)
        factible = restriccion is None or restriccion.cumple(pulido["espesores"])
        if factible and (elegido["chi2"] - pulido["chi2"]) * N > 1.0:
            for c in ("espesores", "rho", "respuesta", "chi2", "rms"):
                elegido[c] = pulido[c]
            elegido["arranque"] = f"{elegido['arranque']} + pulido pyGIMLi"
        else:
            pass
    except Exception as exc:
        logger.debug("Pulido falló: %s", exc)
    if not k_fijo and "estructura de" not in nota_k.split(":")[0]:
        elegido = _fusionar_similares(geom, rhoa, error_rel, elegido, clave_ic, lim_h, lim_r, candidatos,
                                      restriccion=restriccion)
    else:
        pass
    tabla = pd.DataFrame(filas)
    tabla["seleccionado"] = tabla["k"] == elegido["k"]
    pruebas = pd.DataFrame([{"k": r["k"], "arranque": r["arranque"], "escala": r["escala"], "gamma": r["gamma"],
                             "chi2": r["chi2"], clave_ic: r[clave_ic]} for r in cache.values()])
    restr = (f"espesor mínimo: 1.ª capa ≥ {restriccion.h_min_1:.2f} m y cada capa ≥ "
             f"{100 * restriccion.fraccion:.0f} % de la profundidad de su techo" if restriccion else "sin espesor mínimo")
    proceso.append(
        f"4. Inversión de bloques optimizada con Optuna ({len(estudio.trials)} pruebas: arranques base de cada k "
        f"(Dar Zarrouk, otra fuente, división del modelo de k−1 capas, {n_exploratorios} exploratorios) + "
        f"{max(int(n_trials), 5)} de búsqueda): hiperparámetros k "
        f"({ks[0]}–{ks[-1]}), variante del arranque, escala de profundidades (0.6–1.6) y contraste γ (0.5–1.5); "
        f"cada prueba por Levenberg–Marquardt con {restr}; {nota_k}; desempate entre equivalentes: mínima "
        f"estructura. Elegido: {elegido['k']} capas, "
        f"χ² = {elegido['chi2']:.2f}, arranque {elegido['arranque']}."
    )
    return elegido, tabla, pruebas, proceso


def seleccionar_simplificado(
    geom, rhoa, error_rel, espesores_mc, rho_mc, k_max: int = 6, criterio: str = "BIC",
    verificar_cancelacion: Optional[Callable[[], None]] = None, z_max_datos: Optional[float] = None,
    n_arranques: int = 24, z_min_datos: Optional[float] = None, k_fijo: int = 0,
    modelo_zohdy: Optional[tuple[np.ndarray, np.ndarray]] = None, z_viable: Optional[float] = None,
    restriccion: Optional[RestriccionEspesor] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Modelo simplificado: Dar Zarrouk + búsqueda multiarranque + BIC.

    Para cada k se ajustan (Levenberg–Marquardt rápido) los arranques de
    Dar Zarrouk, los de división del modelo de k−1 capas y
    ``n_arranques`` arranques globales (:func:`arranques_globales`); el
    mejor se pule con la inversión de bloques de pyGIMLi.

    Returns
    -------
    (mejor, tabla)
        ``mejor`` con espesores, ρ, respuesta, χ², k y el arranque
        usado; ``tabla`` con una fila por k evaluado.
    """
    N = len(rhoa)
    k_lim = int(min(k_max, len(rho_mc), (N - 1) // 2))
    if k_lim < 2:
        raise InversionSEVError(f"Con {N} datos no se puede ajustar un modelo de 2 capas o más.")
    else:
        pass

    techos_mc = np.cumsum(espesores_mc)
    # Ninguna capa puede ser más gruesa que lo que los datos ven: una
    # interfaz por debajo de ~2 veces la mayor profundidad mediana de
    # investigación no está controlada por ninguna medición.
    z_fondo = 2.0 * float(z_max_datos) if z_max_datos else float(techos_mc[-1])
    if z_viable:
        # Con la profundidad viable de Zohdy, ninguna capa puede ser más
        # gruesa ni ninguna interfaz más profunda que ella.
        z_fondo = float(z_viable)
    else:
        pass
    lim_h = (max(float(espesores_mc[0]) / 20.0, 1e-3), z_fondo)
    lim_r = (max(float(np.min(rhoa)) / 20.0, RHO_MIN), min(float(np.max(rhoa)) * 20.0, RHO_MAX))
    # Rango donde se buscan interfaces: de la mitad de la menor profundidad
    # de investigación a 1.5 veces la mayor.
    z_sup = 1.5 * float(z_max_datos) if z_max_datos else float(techos_mc[-1]) / 2.0
    z_sup = float(z_viable) if z_viable else z_sup
    z_inf = 0.5 * float(z_min_datos) if z_min_datos else float(espesores_mc[0])
    z_busqueda = (max(z_inf, lim_h[0] * 2.0), max(z_sup, 2.0 * z_inf))

    # Sensibilidad de los datos a cada celda del multicapa (para el
    # desempate entre modelos equivalentes).
    try:
        J_mc = fw.jacobiano_log(geom, espesores_mc, rho_mc, incluir_espesores=False) / np.asarray(error_rel, float)[:, None]
        sens_mc = np.sqrt(np.sum(J_mc ** 2, axis=0))
    except Exception:
        sens_mc = None

    candidatos = []
    filas = []
    previo = None
    for k in range(2, k_lim + 1):
        if verificar_cancelacion is not None:
            verificar_cancelacion()
        else:
            pass
        mejor_k = None
        arranques = []
        for conservar in (False, True):
            h0, r0, _tramos = reducir_dar_zarrouk(espesores_mc, rho_mc, k, conservar_espesor=conservar)
            arranques.append((h0, r0, "espesor conservado" if conservar else "Dar Zarrouk clásico"))
        if modelo_zohdy is not None and len(modelo_zohdy[1]) > k:
            # Zohdy reducido a k capas (Dar Zarrouk): espesores y ρ se liberan
            # después en la inversión de bloques.
            for conservar in (False, True):
                h0, r0, _tramos = reducir_dar_zarrouk(modelo_zohdy[0], modelo_zohdy[1], k, conservar_espesor=conservar)
                arranques.append((h0, r0, "Zohdy reducido (Dar Zarrouk)"))
        else:
            pass
        if previo is not None:
            for h0, r0 in _arranques_por_division(previo, z_fondo):
                arranques.append((h0, r0, "división del modelo de k−1 capas"))
        else:
            pass
        n_k = int(min(n_arranques, 8 * (k - 1)))
        for h0, r0 in arranques_globales(k, espesores_mc, rho_mc, z_busqueda, n_k, semilla=k):
            arranques.append((h0, r0, "búsqueda global (multiarranque)"))
        # Dos etapas: (1) todos los arranques con pocas iteraciones, para
        # descartar los que van hacia mínimos malos; (2) los mejores se
        # llevan a convergencia partiendo de donde quedaron.
        previos = []
        for h0, r0, etiqueta in arranques:
            try:
                res = ajustar_bloques_lm(geom, rhoa, error_rel, h0, r0, lim_h, lim_r, max_nfev=8,
                                         restriccion=restriccion)
            except Exception as exc:
                logger.debug("Bloques k=%d falló: %s", k, exc)
                continue
            if np.isfinite(res["chi2"]):
                res["arranque"] = etiqueta
                res["inicial_h"], res["inicial_rho"] = h0, r0
                previos.append(res)
            else:
                pass
        previos.sort(key=lambda r: r["chi2"])
        ajustes = []
        for res in previos[:6]:
            try:
                fino = ajustar_bloques_lm(geom, rhoa, error_rel, res["espesores"], res["rho"], lim_h, lim_r,
                                          restriccion=restriccion)
            except Exception as exc:
                logger.debug("Bloques k=%d (etapa 2) falló: %s", k, exc)
                continue
            if np.isfinite(fino["chi2"]):
                fino["arranque"] = res["arranque"]
                fino["inicial_h"], fino["inicial_rho"] = res["inicial_h"], res["inicial_rho"]
                ajustes.append(fino)
            else:
                pass
        if not ajustes:
            continue
        else:
            pass
        mejor_k = _elegir_equivalente(ajustes, N, espesores_mc, rho_mc, z_busqueda[1], sensibilidad=sens_mc)
        # Pulido con pyGIMLi: solo se acepta si mejora el ajuste de forma
        # apreciable (Δφ > 1); si no, se desplazaría por el valle de
        # equivalencia y se perdería la elección de mínima estructura.
        try:
            pulido = invertir_bloques(geom, rhoa, error_rel, mejor_k["espesores"], mejor_k["rho"],
                                      lam=1.0, limites_espesor=lim_h, limites_rho=lim_r)
            factible = restriccion is None or restriccion.cumple(pulido["espesores"])
            if factible and (mejor_k["chi2"] - pulido["chi2"]) * N > 1.0:
                for clave_p in ("espesores", "rho", "respuesta", "chi2", "rms"):
                    mejor_k[clave_p] = pulido[clave_p]
            else:
                pass
        except Exception as exc:
            logger.debug("Pulido k=%d falló: %s", k, exc)
        previo = mejor_k
        p = 2 * k - 1
        phi = mejor_k["chi2"] * N
        mejor_k["k"] = k
        mejor_k["BIC"] = phi + p * np.log(N)
        mejor_k["AIC"] = phi + 2 * p + (2 * p * (p + 1) / max(N - p - 1, 1))
        candidatos.append(mejor_k)
        filas.append({"k": k, "parametros": p, "chi2": mejor_k["chi2"], "rms": mejor_k["rms"],
                      "BIC": mejor_k["BIC"], "AIC": mejor_k["AIC"], "arranque": mejor_k["arranque"],
                      "interfaces_m": ", ".join(f"{v:.3g}" for v in np.cumsum(mejor_k["espesores"])),
                      "resistividades": ", ".join(f"{v:.3g}" for v in mejor_k["rho"])})

    if not candidatos:
        raise InversionSEVError("Ninguna inversión por capas convergió.")
    else:
        pass

    clave = "AIC" if criterio == "AIC" else "BIC"
    minimo = min(c[clave] for c in candidatos)
    fijos = [c for c in candidatos if c["k"] == int(k_fijo)]
    if fijos:
        # Número de capas fijado por el usuario: sin criterio ni fusión.
        elegido = fijos[0]
    else:
        # Parsimonia: el más simple dentro de 2 unidades del mínimo.
        elegido = min((c for c in candidatos if c[clave] <= minimo + 2.0), key=lambda c: c["k"])
        elegido = _fusionar_similares(geom, rhoa, error_rel, elegido, clave, lim_h, lim_r, candidatos,
                                      restriccion=restriccion)
    tabla = pd.DataFrame(filas)
    tabla["seleccionado"] = tabla["k"] == elegido["k"]
    return elegido, tabla


def _fusionar_similares(geom, rhoa, error_rel, modelo, clave, lim_h, lim_r, candidatos=(),
                        razon: float = 1.25, restriccion: Optional[RestriccionEspesor] = None):
    """Fusiona capas contiguas casi iguales si el criterio no empeora.

    El criterio de información puede conservar dos capas contiguas de
    resistividad parecida (una capa partida en dos) cuando la partición
    mejora el ajuste marginalmente. Aquí se prueba fusionarlas (Dar
    Zarrouk) y reinvertir. Para conservar una partición entre capas con
    menos de 25 % de contraste se exige evidencia **positiva** a su
    favor (ΔBIC > 6 en la escala de Kass y Raftery); si no la hay, se
    queda el modelo con una capa menos.
    """
    N = len(rhoa)
    actual = modelo
    while len(actual["rho"]) > 2:
        rho = np.asarray(actual["rho"], float)
        h = np.asarray(actual["espesores"], float)
        cocientes = np.abs(np.log(rho[1:] / rho[:-1]))
        j = int(np.argmin(cocientes))
        if cocientes[j] > np.log(razon):
            break
        else:
            pass
        if j + 1 == len(rho) - 1:
            nuevos_h = h[:j]
            nuevos_rho = np.concatenate([rho[:j], [float(np.sqrt(rho[j] * rho[j + 1]))]])
        else:
            dz = parametros_dar_zarrouk(h[j:j + 2], rho[j:j + 2])
            nuevos_h = np.concatenate([h[:j], [dz["H"]], h[j + 2:]])
            nuevos_rho = np.concatenate([rho[:j], [dz["rho_m"]], rho[j + 2:]])
        try:
            if restriccion is not None:
                res = ajustar_bloques_lm(geom, rhoa, error_rel, nuevos_h, nuevos_rho, lim_h, lim_r,
                                         restriccion=restriccion)
                res["respuesta"] = fw.respuesta(geom, res["espesores"], res["rho"])
            else:
                res = invertir_bloques(geom, rhoa, error_rel, nuevos_h, nuevos_rho,
                                       limites_espesor=lim_h, limites_rho=lim_r)
        except Exception:
            break
        k = len(res["rho"])
        p = 2 * k - 1
        phi = res["chi2"] * N
        res["BIC"] = phi + p * np.log(N)
        res["AIC"] = phi + 2 * p + (2 * p * (p + 1) / max(N - p - 1, 1))
        res["k"] = k
        res["arranque"] = f"{actual['arranque']} + fusión de capas similares"
        # El modelo de k−1 capas ya evaluado en la selección puede ser mejor
        # que la reinversión de la fusión (otro mínimo local).
        for previo in candidatos:
            if previo["k"] == k and previo[clave] < res[clave]:
                res = dict(previo)
                res["arranque"] = f"{previo['arranque']} (fusión de capas similares)"
            else:
                pass
        if res[clave] <= actual[clave] + 6.0:
            actual = res
        else:
            break
    return actual


# ======================================================================
# IP (Seigel linealizado)
# ======================================================================


def invertir_ip(
    geom, espesores, resistividades, ip_obs, error_ip, suave: bool = True,
) -> dict[str, Any]:
    """Cargabilidad por capa a partir de la cargabilidad aparente.

    Usa la relación de Seigel linealizada ma ≈ J·m, con
    J = ∂ln ρa/∂ln ρ del modelo de resistividad ya invertido. El problema
    es lineal, se resuelve con mínimos cuadrados acotados (m ≥ 0) y
    regularización de suavidad (modelo multicapa) o de amortiguamiento
    (modelo por capas). λ se elige por discrepancia. El ajuste final se
    verifica con la relación de Seigel **no lineal**.
    """
    from scipy.optimize import lsq_linear

    ip_obs = np.asarray(ip_obs, float)
    error_ip = np.maximum(np.asarray(error_ip, float), 1e-3)
    J = fw.jacobiano_log(geom, espesores, resistividades, incluir_espesores=False)
    n = J.shape[1]
    # Regularización: suavidad (multicapa) más un amortiguamiento débil hacia
    # la mediana de la cargabilidad aparente. Sin ese término, las capas
    # profundas y gruesas —casi insensibles— pueden tomar valores extremos
    # alternados que el dato no exige.
    m_ref = float(np.median(np.clip(ip_obs, 0, None)))
    if suave:
        L = np.vstack([np.diff(np.eye(n), axis=0), 0.1 * np.eye(n)])
        b_reg = np.concatenate([np.zeros(n - 1), 0.1 * np.full(n, m_ref)])
    else:
        L = np.eye(n)
        b_reg = np.full(n, m_ref)
    W = 1.0 / error_ip
    A_dat = J * W[:, None]
    b_dat = ip_obs * W

    resultados = []
    for lam in np.logspace(-3, 3, 19):
        A = np.vstack([A_dat, np.sqrt(lam) * L])
        b = np.concatenate([b_dat, np.sqrt(lam) * b_reg])
        sol = lsq_linear(A, b, bounds=(0.0, 900.0), method="bvls" if n <= 40 else "trf")
        calc = J @ sol.x
        chi2 = float(np.mean(((calc - ip_obs) * W) ** 2))
        resultados.append((lam, sol.x, calc, chi2))
    # Discrepancia: el λ más grande (modelo más suave) cuyo χ² llega a 1; si
    # el error de IP está subestimado y ningún λ llega, se acepta hasta 1.2
    # veces el mínimo alcanzable en vez de quedarse con el λ más pequeño,
    # que produce oscilaciones sin significado.
    objetivo_chi2 = max(1.0, 1.2 * min(r[3] for r in resultados))
    ajustan = [r for r in resultados if r[3] <= objetivo_chi2]
    lam, m, calc_lin, chi2 = max(ajustan, key=lambda r: r[0])

    calc_seigel = fw.cargabilidad_aparente(geom, espesores, resistividades, m)
    if not suave:
        # Modelo de pocas capas: refinamiento con la relación de Seigel **no
        # lineal**. Con pocas incógnitas el problema está bien planteado y el
        # amortiguamiento de la solución lineal (que acerca todas las capas a
        # la mediana) ya no hace falta; solo se conserva un término muy débil
        # hacia la solución lineal para las capas que el dato no ve.
        from scipy.optimize import least_squares

        peso_debil = 1e-2

        def residuos_seigel(mm):
            calc = fw.cargabilidad_aparente(geom, espesores, resistividades, mm)
            return np.concatenate([(calc - ip_obs) * W, peso_debil * (mm - m) / max(m_ref, 1.0)])

        try:
            sol_nl = least_squares(residuos_seigel, np.clip(m, 0.0, 899.0), bounds=(0.0, 900.0),
                                   method="trf", max_nfev=60)
            calc_nl = fw.cargabilidad_aparente(geom, espesores, resistividades, sol_nl.x)
            if np.mean(((calc_nl - ip_obs) * W) ** 2) < np.mean(((calc_seigel - ip_obs) * W) ** 2):
                m, calc_seigel = sol_nl.x, calc_nl
            else:
                pass
        except Exception as exc:
            logger.debug("Refinamiento no lineal de IP falló: %s", exc)
    else:
        pass
    return {
        "m": m, "respuesta_lineal": calc_lin, "respuesta": calc_seigel,
        "chi2": float(np.mean(((calc_seigel - ip_obs) * W) ** 2)),
        "chi2_lineal": chi2, "lam": float(lam),
        "rms_abs": float(np.sqrt(np.mean((calc_seigel - ip_obs) ** 2))),
    }


# ======================================================================
# Orquestador de un sondeo
# ======================================================================


@dataclass
class ConfigInversion:
    """Parámetros de la inversión de un sondeo.

    Attributes
    ----------
    metodo_lambda : {'optuna', 'discrepancia', 'manual'}
    regularizacion : {'suave', 'robusta', 'nitida'}
        Suave (por omisión): Occam clásico, L2. Robusta: Occam con los datos
        en norma L1 (reponderación iterativa de pyGIMLi), menos sensible a
        lecturas aberrantes. Nítida: soporte de gradiente mínimo (bloques).
    lam : float
        λ del modo manual.
    n_trials : int
        Pruebas de Optuna.
    robusto : bool
        Reponderación robusta de datos (IRLS) en la inversión suave.
    k_max : int
        Máximo de capas del modelo simplificado.
    modo_bloques : {'dar_zarrouk', 'profundidad'}
        Origen del modelo de bloques inicial: reducción de Dar Zarrouk de
        Zohdy y del multicapa (se usa la que mejor ajusta) o división
        log-uniforme de la profundidad de Edwards en ``k_profundidad``
        capas.
    k_profundidad : int
        Capas del modo por profundidad.
    n_trials_bloques : int
        Pruebas de Optuna en la inversión de bloques.
    espesor_min_fraccion : float
        Espesor mínimo de cada capa del simplificado como fracción de la
        profundidad de su techo (la primera capa: al menos la profundidad
        de investigación de la apertura más corta). 0 = sin restricción.
    zohdy : {'edwards', 'rango', 'factor_edwards'}
        Profundidad de atribución de Zohdy, z = AB/n. ``'edwards'``: n en
        n_E ± ``zohdy_ventana`` (n_E = AB/z_e de Edwards); ``'rango'``: n
        entre ``zohdy_n_min`` y ``zohdy_n_max``. En ambos el paso es
        ``zohdy_paso`` y cada n se evalúa con Zohdy completo (resistividades
        corregidas). ``'factor_edwards'``: versión previa.
    malla : {'zohdy', 'logaritmica'}
        Malla del multicapa. ``'zohdy'``: interfaces de Zohdy subdivididas
        hasta la profundidad viable f·max(z_e). ``'logaritmica'``: malla
        regular en log z hasta 2.5·max(z_e).
    k_fijo : int
        Número de capas del simplificado fijado a mano (0 = lo elige el
        criterio de información).
    n_arranques : int
        Arranques globales por número de capas en la búsqueda del
        simplificado (0 = solo Dar Zarrouk y divisiones).
    criterio : {'BIC', 'AIC'}
    calcular_doi : bool
    invertir_ip : bool
    motor : {'numpy', 'pygimli'}
        Motor de modelado directo (ver :mod:`forward_sev`).
    """

    metodo_lambda: str = "optuna"
    regularizacion: str = "suave"
    lam: float = 20.0
    n_trials: int = 10
    robusto: bool = False
    k_max: int = 8
    n_arranques: int = 24
    k_fijo: int = 0
    malla: str = "zohdy"
    modo_bloques: str = "dar_zarrouk"
    k_profundidad: int = 5
    n_trials_bloques: int = 40
    k_desde_estructura: bool = True
    espesor_min_fraccion: float = 0.25
    zohdy: str = "edwards"
    zohdy_ventana: float = 1.0
    zohdy_n_min: float = 5.0
    zohdy_n_max: float = 8.0
    zohdy_paso: float = 0.125
    criterio: str = "BIC"
    calcular_doi: bool = True
    invertir_ip: bool = True
    motor: str = "numpy"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ResultadoInversion:
    """Tablas de resultado de un sondeo invertido."""

    multicapa: pd.DataFrame
    simplificado: pd.DataFrame
    ajuste: pd.DataFrame
    seleccion: pd.DataFrame
    lambdas: pd.DataFrame
    metricas: dict[str, Any] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)
    zohdy: Optional[pd.DataFrame] = None
    pruebas_bloques: Optional[pd.DataFrame] = None
    proceso: Optional[pd.DataFrame] = None


def invertir_sondeo(
    procesado: pd.DataFrame, config: Optional[ConfigInversion] = None,
    verificar_cancelacion: Optional[Callable[[], None]] = None,
) -> ResultadoInversion:
    """Ejecuta el flujo completo de inversión de un sondeo preprocesado."""
    import procesamiento_sev as pr

    cfg = config or ConfigInversion()
    fw.fijar_motor(cfg.motor)
    advertencias: list[str] = []
    proceso: list[str] = []
    inicio = time.time()

    datos = procesado[procesado["usar"]].reset_index(drop=True)
    if len(datos) < 5:
        raise InversionSEVError(f"Solo hay {len(datos)} datos utilizables (mínimo 5).")
    else:
        pass
    geom = fw.GeometriaSondeo.desde_tabla(datos)
    rhoa = datos["rhoa_proc"].to_numpy(float)
    error = datos["error_rel"].to_numpy(float)

    # --- 1. Zohdy sobre la curva monovaluada ---
    curva = pr.curva_para_zohdy(procesado)
    curva = curva[np.concatenate([[True], np.diff(curva["z_mediana"].to_numpy(float)) > 0])]
    geom_z = fw.GeometriaSondeo.desde_tabla(curva)
    ze_curva = curva["z_mediana"].to_numpy(float)
    longitud_ab = longitud_ab_equivalente(curva)
    modo_zohdy = {"ab_n": "rango"}.get(cfg.zohdy, cfg.zohdy)
    if modo_zohdy in ("edwards", "rango") and np.all(np.diff(longitud_ab) > 0):
        # n de Edwards: la profundidad mediana de investigación expresada como
        # AB/n (≈ 5.3 en Schlumberger, ≈ 5.8 en Wenner).
        n_edwards = float(np.median(longitud_ab / ze_curva))
        if modo_zohdy == "edwards":
            n_min = max(n_edwards - cfg.zohdy_ventana, 1.0)
            n_max = n_edwards + cfg.zohdy_ventana
        else:
            n_min, n_max = cfg.zohdy_n_min, cfg.zohdy_n_max
        zoh = zh.invertir_zohdy_ab(geom_z, curva["rhoa_proc"].to_numpy(float), longitud_ab,
                                   n_min=n_min, n_max=n_max, paso=cfg.zohdy_paso)
        # Factor equivalente sobre la profundidad de Edwards (para la malla).
        zoh.factor_profundidad = float(np.median(longitud_ab / zoh.n_ab / ze_curva))
        proceso.append(
            f"1. Zohdy: profundidad de cada capa z = AB/n; n de Edwards = {n_edwards:.2f}; búsqueda de n en "
            f"{n_min:.3g}–{n_max:.3g} con paso {cfg.zohdy_paso:g} "
            f"({'Edwards ± ' + format(cfg.zohdy_ventana, 'g') if modo_zohdy == 'edwards' else 'rango fijo'}); "
            f"cada n se evalúa con las resistividades ya corregidas. Mejor: AB/{zoh.n_ab:g}, RMS {zoh.rms:.2f} %, "
            f"{len(zoh.resistividades)} capas."
        )
        if min(abs(zoh.n_ab - n_min), abs(zoh.n_ab - n_max)) < 1e-9:
            advertencias.append(
                f"Zohdy: la mejor profundidad quedó en el límite de la búsqueda (AB/{zoh.n_ab:g}); el óptimo puede "
                f"estar fuera de AB/{n_min:.3g}–AB/{n_max:.3g}. Considere ampliar el rango."
            )
        else:
            pass
    else:
        zoh = zh.invertir_zohdy(geom_z, curva["rhoa_proc"].to_numpy(float), ze_curva)
        proceso.append(f"1. Zohdy (factor sobre Edwards, versión previa): factor {zoh.factor_profundidad:.2f}, "
                       f"RMS {zoh.rms:.2f} %.")
    # Malla del multicapa: capas finas equiespaciadas en log z, desde la
    # mitad de la menor profundidad de investigación hasta 2.5 veces la
    # mayor (espacio para que la DOI muestre dónde dejan de ver los datos).
    # El modelo de Zohdy, muestreado en esa malla, es el modelo inicial.
    z_inv = datos["z_mediana"].to_numpy(float)
    n_capas = int(np.clip(round(1.6 * len(datos)), 30, 48))
    if cfg.malla == "zohdy":
        espesores, z_viable = malla_desde_zohdy(zoh.espesores, zoh.factor_profundidad, float(z_inv.max()),
                                                n_capas, z_e_min=float(z_inv.min()))
    else:
        espesores = capas_logaritmicas(max(0.5 * float(z_inv.min()), 0.05), 2.5 * float(z_inv.max()), n_capas)
        z_viable = 1.5 * float(z_inv.max())
    techos_malla = np.concatenate([[0.0], np.cumsum(espesores)])
    centros = techos_malla + np.concatenate([espesores, [espesores[-1]]]) / 2
    rho_inicial = fw.muestrear_modelo(zoh.espesores, zoh.resistividades, centros)

    # --- 2. Occam con pyGIMLi desde Zohdy ---
    robusto = bool(cfg.robusto or cfg.regularizacion == "robusta")
    mejor, lambdas = buscar_lambda(
        geom, rhoa, error, espesores, rho_inicial,
        metodo=cfg.metodo_lambda, n_trials=cfg.n_trials, lam_manual=cfg.lam,
        robusto=robusto, verificar_cancelacion=verificar_cancelacion,
        nitida=cfg.regularizacion == "nitida",
    )
    rho_mc = mejor["rho"]
    nombre_reg = {"suave": "suave (Occam, L2)", "robusta": "robusta (datos en norma L1 por IRLS)",
                  "nitida": "nítida (gradiente mínimo)"}.get(cfg.regularizacion, cfg.regularizacion)
    nombre_lam = {"optuna": f"Optuna ({cfg.n_trials} pruebas; λ entre 0.1 y 1000)",
                  "discrepancia": "discrepancia (mayor λ con χ² ≤ 1)", "manual": "manual"}.get(cfg.metodo_lambda, "")
    if cfg.regularizacion == "nitida" and cfg.metodo_lambda != "manual":
        nombre_lam = "continuación de 100 a 0.3"
    else:
        pass
    proceso.append(
        f"2. Multicapa con pyGIMLi: {len(rho_mc)} capas de espesor fijo hasta {z_viable:.1f} m "
        f"(malla {'sobre las interfaces de Zohdy' if cfg.malla == 'zohdy' else 'logarítmica'}), modelo inicial = "
        f"Zohdy; regularización {nombre_reg}; λ por {nombre_lam} → λ = {mejor['lam']:.3g}, χ² = {mejor['chi2']:.2f}."
    )
    if mejor["chi2"] > 3.0:
        advertencias.append(
            f"El modelo multicapa no alcanza el nivel de ruido (χ² = {mejor['chi2']:.1f}): "
            "el error asignado puede ser optimista o hay efectos no 1D."
        )
    else:
        pass

    # --- 3. DOI ---
    doi, indice_doi = float(np.sum(espesores)), np.zeros(len(rho_mc))
    if cfg.calcular_doi:
        try:
            doi, indice_doi = profundidad_investigacion(
                geom, rhoa, error, espesores, lam=max(mejor["lam"], 1.0),
                z_min_valida=float(np.min(datos["z_mediana"])),
                z_escala=float(np.median(datos["z_mediana"])),
            )
            # Ninguna medición tiene su profundidad mediana por debajo de
            # max(z_e); más allá de 1.5 veces esa profundidad la DOI sería
            # una extrapolación del índice, no información de los datos.
            doi = min(doi, 1.5 * float(z_inv.max()), z_viable if cfg.malla == "zohdy" else np.inf)
        except Exception as exc:
            advertencias.append(f"No se pudo estimar la DOI ({exc}).")
    else:
        pass

    # --- 4. Bloques: Dar Zarrouk (o profundidad) + inversión optimizada con Optuna ---
    restriccion = (RestriccionEspesor(h_min_1=float(np.min(datos["z_mediana"])),
                                      fraccion=float(cfg.espesor_min_fraccion))
                   if cfg.espesor_min_fraccion > 0 else None)
    simp, seleccion, pruebas_bloques, pasos = invertir_bloques_optuna(
        geom, rhoa, error, {"Zohdy": (zoh.espesores, zoh.resistividades), "multicapa": (espesores, rho_mc)},
        espesores, rho_mc, k_max=cfg.k_max, k_fijo=int(cfg.k_fijo if cfg.modo_bloques != "profundidad"
                                                         else cfg.k_profundidad),
        criterio=cfg.criterio, n_trials=cfg.n_trials_bloques, restriccion=restriccion,
        z_max_datos=float(np.max(datos["z_mediana"])), z_min_datos=float(np.min(datos["z_mediana"])),
        z_viable=z_viable if cfg.malla == "zohdy" else None, modo=cfg.modo_bloques,
        z_e=datos["z_mediana"].to_numpy(float), verificar_cancelacion=verificar_cancelacion,
        k_desde_estructura=bool(cfg.k_desde_estructura),
    )
    proceso.extend(pasos)
    cov = covarianza_bloques(geom, simp["espesores"], simp["rho"], error)

    # --- 5. IP ---
    ip = datos["ip_proc"].to_numpy(float)
    hay_ip = bool(cfg.invertir_ip and np.any(ip > 0))
    ip_mc = ip_simp = None
    if hay_ip:
        err_ip = datos["error_ip"].to_numpy(float)
        ip_mc = invertir_ip(geom, espesores, rho_mc, ip, err_ip, suave=True)
        ip_simp = invertir_ip(geom, simp["espesores"], simp["rho"], ip, err_ip, suave=False)
    else:
        pass

    # --- Tablas ---
    techos = np.concatenate([[0.0], np.cumsum(espesores)])
    multicapa = pd.DataFrame({
        "capa": np.arange(1, len(rho_mc) + 1),
        "z_techo": techos,
        "z_base": np.concatenate([np.cumsum(espesores), [np.inf]]),
        "espesor": np.concatenate([espesores, [np.inf]]),
        "rho": rho_mc,
        "rho_zohdy": rho_inicial,
        "indice_doi": indice_doi,
        "bajo_doi": techos >= doi,
        "m_ip": ip_mc["m"] if ip_mc else 0.0,
    })
    multicapa["S"] = np.where(np.isfinite(multicapa["espesor"]), multicapa["espesor"] / multicapa["rho"], np.nan)
    multicapa["T"] = np.where(np.isfinite(multicapa["espesor"]), multicapa["espesor"] * multicapa["rho"], np.nan)
    import parametros_sev as ps

    multicapa = ps.acumulados(multicapa)

    k = len(simp["rho"])
    techos_s = np.concatenate([[0.0], np.cumsum(simp["espesores"])])
    sig_h = np.concatenate([cov["sigma_h"], [np.nan]])
    sig_r = cov["sigma_rho"]
    simplificado = pd.DataFrame({
        "capa": np.arange(1, k + 1),
        "z_techo": techos_s,
        "z_base": np.concatenate([np.cumsum(simp["espesores"]), [np.inf]]),
        "espesor": np.concatenate([simp["espesores"], [np.inf]]),
        "rho": simp["rho"],
        "rho_min": simp["rho"] * np.exp(-np.minimum(sig_r, np.log(100))),
        "rho_max": simp["rho"] * np.exp(np.minimum(sig_r, np.log(100))),
        "espesor_min": np.concatenate([simp["espesores"] * np.exp(-np.minimum(cov["sigma_h"], np.log(100))), [np.nan]]),
        "espesor_max": np.concatenate([simp["espesores"] * np.exp(np.minimum(cov["sigma_h"], np.log(100))), [np.nan]]),
        "sigma_ln_rho": sig_r,
        "sigma_ln_h": sig_h,
        "m_ip": ip_simp["m"] if ip_simp else 0.0,
    })
    simplificado["S"] = simplificado["espesor"] / simplificado["rho"]
    simplificado["T"] = simplificado["espesor"] * simplificado["rho"]
    simplificado.loc[~np.isfinite(simplificado["espesor"]), ["S", "T"]] = np.nan
    resolucion = [
        _resolucion(cov["sigma_h"][i], cov["sigma_rho"][i], cov["sigma_S"][i], cov["sigma_T"][i])
        for i in range(k - 1)
    ]
    resolucion.append("semiespacio" if cov["sigma_rho"][-1] <= np.log(1.25) else "semiespacio poco resuelto")
    simplificado["resolucion"] = resolucion
    simplificado["bajo_doi"] = simplificado["z_techo"] >= doi
    simplificado = ps.acumulados(simplificado)

    ajuste = datos[["AB2", "MN2", "a", "z_mediana", "rhoa_obs", "rhoa_proc", "error_rel",
                    "ip_obs", "ip_proc", "error_ip"]].copy()
    ajuste["rhoa_multicapa"] = mejor["respuesta"]
    ajuste["rhoa_simplificado"] = simp["respuesta"]
    ajuste["rhoa_zohdy"] = fw.respuesta(geom, zoh.espesores, zoh.resistividades)
    ajuste["error_zohdy_pct"] = 100 * (ajuste["rhoa_zohdy"] / ajuste["rhoa_proc"] - 1)
    # Modelo de Zohdy tal como sale del método (una capa por dato, con sus
    # propias profundidades), no remuestreado en la malla del multicapa.
    bases_z = np.cumsum(zoh.espesores)
    zohdy_tabla = pd.DataFrame({
        "capa": np.arange(1, len(zoh.resistividades) + 1),
        "z_techo": np.concatenate([[0.0], bases_z]),
        "z_base": np.concatenate([bases_z, [np.inf]]),
        "espesor": np.concatenate([zoh.espesores, [np.inf]]),
        "rho": zoh.resistividades,
    })
    ajuste["error_multicapa_pct"] = 100 * (ajuste["rhoa_multicapa"] / ajuste["rhoa_proc"] - 1)
    ajuste["error_simplificado_pct"] = 100 * (ajuste["rhoa_simplificado"] / ajuste["rhoa_proc"] - 1)
    ajuste["ip_multicapa"] = ip_mc["respuesta"] if ip_mc else 0.0
    ajuste["ip_simplificado"] = ip_simp["respuesta"] if ip_simp else 0.0

    dz_mc = parametros_dar_zarrouk(espesores[techos[:-1] < doi], rho_mc[:-1][techos[:-1] < doi])
    dz_s = parametros_dar_zarrouk(simp["espesores"], simp["rho"])
    metricas = {
        "zohdy_rms": zoh.rms,
        "zohdy_factor": zoh.factor_profundidad,
        "zohdy_n_capas": int(len(zoh.resistividades)),
        "zohdy_n_ab": float(zoh.n_ab) if zoh.n_ab else np.nan,
        "profundidad_viable": float(z_viable),
        "malla": cfg.malla,
        "multicapa_chi2": mejor["chi2"],
        "multicapa_rms": mejor["rms"],
        "multicapa_lambda": mejor["lam"],
        "multicapa_n_capas": int(len(rho_mc)),
        "doi": doi,
        "simplificado_chi2": simp["chi2"],
        "simplificado_rms": simp["rms"],
        "simplificado_n_capas": int(k),
        "simplificado_arranque": simp["arranque"],
        "criterio": cfg.criterio,
        "metodo_lambda": cfg.metodo_lambda,
        "tiene_ip": hay_ip,
        "ip_multicapa_rms": ip_mc["rms_abs"] if ip_mc else np.nan,
        "ip_simplificado_rms": ip_simp["rms_abs"] if ip_simp else np.nan,
        "S_multicapa": dz_mc["S"],
        "T_multicapa": dz_mc["T"],
        "S_simplificado": dz_s["S"],
        "T_simplificado": dz_s["T"],
        "duracion_s": time.time() - inicio,
        "motor": cfg.motor,
    }
    return ResultadoInversion(
        multicapa=multicapa, simplificado=simplificado, ajuste=ajuste,
        seleccion=seleccion, lambdas=lambdas, metricas=metricas, advertencias=advertencias,
        zohdy=zohdy_tabla, pruebas_bloques=pruebas_bloques,
        proceso=pd.DataFrame({"paso": [p.split(".")[0] for p in proceso], "descripcion": proceso}),
    )
