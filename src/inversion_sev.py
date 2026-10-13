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


def ajustar_bloques_lm(
    geom, rhoa, error_rel, espesores0, resistividades0,
    limites_espesor: tuple[float, float], limites_rho: tuple[float, float], max_nfev: int = 50,
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
                        sensibilidad: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Entre los ajustes de k capas que explican los datos igual de bien
    (φ dentro de ``tolerancia_phi`` = 0.25 del mejor, ≈ medio σ), elige el más parecido al
    multicapa.

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
    elegido = min(equivalentes, key=lambda a: a["distancia_multicapa"])
    elegido["n_equivalentes"] = len(equivalentes)
    return elegido


def seleccionar_simplificado(
    geom, rhoa, error_rel, espesores_mc, rho_mc, k_max: int = 6, criterio: str = "BIC",
    verificar_cancelacion: Optional[Callable[[], None]] = None, z_max_datos: Optional[float] = None,
    n_arranques: int = 24, z_min_datos: Optional[float] = None, k_fijo: int = 0,
    modelo_zohdy: Optional[tuple[np.ndarray, np.ndarray]] = None, z_viable: Optional[float] = None,
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
                res = ajustar_bloques_lm(geom, rhoa, error_rel, h0, r0, lim_h, lim_r, max_nfev=8)
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
                fino = ajustar_bloques_lm(geom, rhoa, error_rel, res["espesores"], res["rho"], lim_h, lim_r)
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
            if (mejor_k["chi2"] - pulido["chi2"]) * N > 1.0:
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
        elegido = _fusionar_similares(geom, rhoa, error_rel, elegido, clave, lim_h, lim_r, candidatos)
    tabla = pd.DataFrame(filas)
    tabla["seleccionado"] = tabla["k"] == elegido["k"]
    return elegido, tabla


def _fusionar_similares(geom, rhoa, error_rel, modelo, clave, lim_h, lim_r, candidatos=(),
                        razon: float = 1.25):
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
    regularizacion : {'nitida', 'suave'}
        Nítida (soporte de gradiente mínimo, por defecto): multicapa por
        bloques con interfaces definidas. Suave: Occam clásico (L2).
    lam : float
        λ del modo manual.
    n_trials : int
        Pruebas de Optuna.
    robusto : bool
        Reponderación robusta de datos (IRLS) en la inversión suave.
    k_max : int
        Máximo de capas del modelo simplificado.
    zohdy : {'ab_n', 'factor_edwards'}
        Profundidad de atribución de Zohdy: z = AB/n con n buscado entre
        ``zohdy_n_min`` y ``zohdy_n_max`` (paso ``zohdy_paso``), evaluando
        cada n con las resistividades ya corregidas; o el factor sobre la
        profundidad de Edwards elegido antes de corregir (versión previa).
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
    regularizacion: str = "nitida"
    lam: float = 20.0
    n_trials: int = 10
    robusto: bool = False
    k_max: int = 8
    n_arranques: int = 24
    k_fijo: int = 0
    malla: str = "zohdy"
    zohdy: str = "ab_n"
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


def invertir_sondeo(
    procesado: pd.DataFrame, config: Optional[ConfigInversion] = None,
    verificar_cancelacion: Optional[Callable[[], None]] = None,
) -> ResultadoInversion:
    """Ejecuta el flujo completo de inversión de un sondeo preprocesado."""
    import procesamiento_sev as pr

    cfg = config or ConfigInversion()
    fw.fijar_motor(cfg.motor)
    advertencias: list[str] = []
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
    if cfg.zohdy == "ab_n" and np.all(np.diff(longitud_ab) > 0):
        zoh = zh.invertir_zohdy_ab(geom_z, curva["rhoa_proc"].to_numpy(float), longitud_ab,
                                   n_min=cfg.zohdy_n_min, n_max=cfg.zohdy_n_max, paso=cfg.zohdy_paso)
        # Factor equivalente sobre la profundidad de Edwards (para la malla).
        zoh.factor_profundidad = float(np.median(longitud_ab / zoh.n_ab / ze_curva))
        if min(abs(zoh.n_ab - cfg.zohdy_n_min), abs(zoh.n_ab - cfg.zohdy_n_max)) < 1e-9:
            advertencias.append(
                f"Zohdy: la mejor profundidad quedó en el límite del rango (AB/{zoh.n_ab:g}); el óptimo puede "
                f"estar fuera de AB/{cfg.zohdy_n_min:g}–AB/{cfg.zohdy_n_max:g}. Considere ampliar el rango."
            )
        else:
            pass
    else:
        zoh = zh.invertir_zohdy(geom_z, curva["rhoa_proc"].to_numpy(float), ze_curva)
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
    mejor, lambdas = buscar_lambda(
        geom, rhoa, error, espesores, rho_inicial,
        metodo=cfg.metodo_lambda, n_trials=cfg.n_trials, lam_manual=cfg.lam,
        robusto=cfg.robusto, verificar_cancelacion=verificar_cancelacion,
        nitida=cfg.regularizacion == "nitida",
    )
    rho_mc = mejor["rho"]
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

    # --- 4. Simplificado: Dar Zarrouk + bloques + BIC ---
    simp, seleccion = seleccionar_simplificado(
        geom, rhoa, error, espesores, rho_mc, k_max=cfg.k_max, criterio=cfg.criterio,
        verificar_cancelacion=verificar_cancelacion, z_max_datos=float(np.max(datos["z_mediana"])),
        n_arranques=cfg.n_arranques, z_min_datos=float(np.min(datos["z_mediana"])),
        k_fijo=int(cfg.k_fijo), modelo_zohdy=(zoh.espesores, zoh.resistividades),
        z_viable=z_viable if cfg.malla == "zohdy" else None,
    )
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
        zohdy=zohdy_tabla,
    )
