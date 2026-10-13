"""
parametros_sev.py
==================================================================
Parámetros derivados del modelo invertido: Dar Zarrouk, resistividad
promedio, corrosividad del suelo y unidades geoeléctricas de campaña.

Parámetros de Dar Zarrouk (Maillet, 1947)
--------------------------------------------
Para el paquete de capas entre la superficie y la profundidad H:

=====================  =====================  =========================
Símbolo                Expresión              Significado
=====================  =====================  =========================
S  (siemens)           Σ hᵢ / ρᵢ              Conductancia longitudinal
T  (Ω·m²)              Σ hᵢ · ρᵢ              Resistencia transversal
                                              unitaria
ρL (Ω·m)               H / S                  Resistividad longitudinal
                                              (media armónica)
ρT (Ω·m)               T / H                  Resistividad transversal
                                              (media aritmética)
λ                      √(ρT / ρL)             Coeficiente de
                                              pseudoanisotropía (≥ 1)
ρm (Ω·m)               √(ρL · ρT)             Resistividad media
                                              (media geométrica)
=====================  =====================  =========================

S y T son exactamente las cantidades que el principio de equivalencia
conserva, así que son **más robustas que h y ρ por separado**. Como
control de calidad, deben coincidir entre el modelo multicapa y el
simplificado: la reducción de Dar Zarrouk las conserva por
construcción, y si la inversión final las cambió mucho, algo no cuadra.

Corrosividad
---------------
Se clasifica con la resistividad **longitudinal (armónica)** de la capa
de suelo entre 0 y la profundidad de interés (por defecto 3 m, la de una
tubería enterrada). Se usa la armónica y no la aritmética porque la
corriente de corrosión busca el camino más conductor: una capa delgada
de baja resistividad domina el riesgo, y la media armónica es la que lo
refleja. Los umbrales por defecto son los de la escala de uso común en
protección catódica (NACE / Roberge, en Ω·cm: 500, 1000, 2000,
10 000), y son configurables.

Unidades geoeléctricas
-------------------------
Las capas del modelo simplificado de todos los sondeos se agrupan con
una **mezcla gaussiana** (log ρ, profundidad y, si existe, IP) cuyo
número de componentes se elige por BIC. La misma etiqueta significa lo
mismo en todos los sondeos, lo que permite correlacionarlos en los
perfiles.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("sev.parametros")

#: Umbrales de corrosividad en Ω·m (equivalen a 500, 1000, 2000 y 10 000 Ω·cm).
UMBRALES_CORROSIVIDAD = (5.0, 10.0, 20.0, 100.0)
CLASES_CORROSIVIDAD = ("Muy alta", "Alta", "Moderada", "Baja", "Muy baja")


def truncar(modelo: pd.DataFrame, profundidad: float) -> tuple[np.ndarray, np.ndarray]:
    """Espesores y resistividades del paquete entre 0 y ``profundidad``.

    La capa que cruza la profundidad se recorta, y el semiespacio aporta
    lo que falte hasta llegar a ella.
    """
    techos = modelo["z_techo"].to_numpy(float)
    bases = modelo["z_base"].to_numpy(float)
    rho = modelo["rho"].to_numpy(float)
    h = np.clip(np.minimum(bases, profundidad) - techos, 0.0, None)
    activo = h > 0
    return h[activo], rho[activo]


def dar_zarrouk(modelo: pd.DataFrame, profundidad: float) -> dict[str, float]:
    """S, T, ρL, ρT, λ y ρm hasta ``profundidad``."""
    h, rho = truncar(modelo, profundidad)
    if len(h) == 0:
        return {k: np.nan for k in ("H", "S", "T", "rho_L", "rho_T", "lambda", "rho_m")}
    else:
        pass
    H = float(h.sum())
    S = float(np.sum(h / rho))
    T = float(np.sum(h * rho))
    rho_l = H / S
    rho_t = T / H
    return {
        "H": H, "S": S, "T": T, "rho_L": rho_l, "rho_T": rho_t,
        "lambda": float(np.sqrt(rho_t / rho_l)), "rho_m": float(np.sqrt(rho_l * rho_t)),
    }


def acumulados(modelo: pd.DataFrame) -> pd.DataFrame:
    """Agrega a un modelo de capas los parámetros de Dar Zarrouk acumulados
    desde la superficie hasta la base de cada capa.

    Columnas nuevas (NaN en el semiespacio):

    - ``S_acum`` (S) y ``T_acum`` (Ω·m²): conductancia longitudinal y
      resistencia transversal de 0 a z_base.
    - ``rho_L_prom`` = z_base / S_acum: resistividad promedio longitudinal
      (media armónica pesada por espesor; la que ve una corriente
      horizontal, dominada por las capas conductoras).
    - ``rho_T_prom`` = T_acum / z_base: resistividad promedio transversal
      (media aritmética pesada por espesor; la que ve una corriente
      vertical, dominada por las capas resistivas).
    - ``rho_prom`` = √(ρL·ρT): resistividad media del paquete.
    - ``anisotropia`` λ = √(ρT/ρL) del paquete (1 = homogéneo).

    Son independientes de cómo se discretice el modelo: el multicapa y el
    simplificado deben dar curvas parecidas donde los datos resuelven.
    Es idempotente (recalcula si las columnas ya existen).
    """
    m = modelo.copy()
    esp = m["espesor"].to_numpy(float)
    rho = m["rho"].to_numpy(float)
    finito = np.isfinite(esp)
    S = np.where(finito, esp / rho, 0.0).cumsum()
    T = np.where(finito, esp * rho, 0.0).cumsum()
    H = np.where(finito, esp, 0.0).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = H / S
        rt = T / H
    m["S_acum"] = np.where(finito, S, np.nan)
    m["T_acum"] = np.where(finito, T, np.nan)
    m["rho_L_prom"] = np.where(finito, rl, np.nan)
    m["rho_T_prom"] = np.where(finito, rt, np.nan)
    m["rho_prom"] = np.where(finito, np.sqrt(rl * rt), np.nan)
    m["anisotropia"] = np.where(finito, np.sqrt(rt / rl), np.nan)
    # Contraste de cada capa contra el promedio (ρm) del paquete que tiene
    # encima: > 2 = la resistividad sube con fuerza en esa capa; < 0.5 = baja.
    prom_arriba = np.concatenate([[np.nan], np.sqrt(rl * rt)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        contraste = rho / prom_arriba
    m["contraste_vs_arriba"] = contraste
    m["cambio"] = np.where(contraste >= UMBRAL_CAMBIO, "aumento fuerte",
                           np.where(contraste <= 1.0 / UMBRAL_CAMBIO, "disminución fuerte", ""))
    return m


#: Contraste (ρ capa / ρ promedio de lo de arriba) a partir del cual se
#: marca un cambio fuerte de resistividad.
UMBRAL_CAMBIO = 2.0
#: Anisotropía máxima de un grupo de celdas para considerarlo una sola capa
#: (λ = 1.02 equivale a dos celdas de igual espesor con ρ que difieren ×1.5).
LAMBDA_EQUIVALENTE = 1.02


def agrupar_equivalentes(modelo: pd.DataFrame, lambda_max: float = LAMBDA_EQUIVALENTE) -> pd.DataFrame:
    """Agrupa celdas contiguas que pueden representarse como una sola capa.

    Se acumulan T y S celda a celda; mientras la anisotropía del grupo
    λ = √(ρT/ρL) = √(T·S)/H no pase de ``lambda_max``, las celdas son
    eléctricamente casi iguales y el grupo equivale a una capa con
    ρ = √(T/S) (Dar Zarrouk) sin cambiar la respuesta. El semiespacio
    queda como último grupo.

    Comparar la T de cada celda por separado no basta: en el multicapa las
    celdas profundas son más gruesas y su T es mayor aunque su ρ sea la
    misma. Lo que decide si se pueden unir es λ del grupo.
    """
    esp = modelo["espesor"].to_numpy(float)
    rho = modelo["rho"].to_numpy(float)
    techos = modelo["z_techo"].to_numpy(float)
    grupos = []
    i = 0
    n = len(rho)
    while i < n:
        if not np.isfinite(esp[i]):
            grupos.append((i, i + 1))
            break
        else:
            pass
        j = i + 1
        while j < n and np.isfinite(esp[j]):
            h = esp[i:j + 1]
            r = rho[i:j + 1]
            lam = np.sqrt(np.sum(h * r) * np.sum(h / r)) / np.sum(h)
            if lam > lambda_max:
                break
            else:
                pass
            j += 1
        grupos.append((i, j))
        i = j
    filas = []
    for k, (a, b) in enumerate(grupos, start=1):
        h = esp[a:b]
        r = rho[a:b]
        if np.all(np.isfinite(h)):
            H, S, T = float(h.sum()), float(np.sum(h / r)), float(np.sum(h * r))
            filas.append({"capa_equivalente": k, "celdas": f"{a + 1}–{b}" if b - a > 1 else f"{a + 1}",
                          "n_celdas": b - a, "z_techo": float(techos[a]), "z_base": float(techos[a] + H),
                          "espesor": H, "S": S, "T": T, "rho_equivalente": float(np.sqrt(T / S)),
                          "rho_L": H / S, "rho_T": T / H, "anisotropia": float(np.sqrt(T * S) / H)})
        else:
            filas.append({"capa_equivalente": k, "celdas": f"{a + 1}–{n}" if n - a > 1 else f"{a + 1}",
                          "n_celdas": n - a, "z_techo": float(techos[a]), "z_base": np.inf, "espesor": np.inf,
                          "S": np.nan, "T": np.nan, "rho_equivalente": float(rho[a]),
                          "rho_L": np.nan, "rho_T": np.nan, "anisotropia": np.nan})
    return pd.DataFrame(filas)


def curva_acumulada(modelo: pd.DataFrame, z: np.ndarray, columna: str = "T") -> np.ndarray:
    """T(z) = ∫₀ᶻ ρ dz' o S(z) = ∫₀ᶻ dz'/ρ evaluadas en las profundidades
    ``z`` (lineales dentro de cada capa)."""
    techos = modelo["z_techo"].to_numpy(float)
    bases = modelo["z_base"].to_numpy(float)
    rho = modelo["rho"].to_numpy(float)
    z = np.asarray(z, float)[:, None]
    dentro = np.clip(np.minimum(z, bases[None, :]) - techos[None, :], 0.0, None)
    peso = rho if columna == "T" else 1.0 / rho
    return (dentro * peso[None, :]).sum(axis=1)


def clase_corrosividad(rho: float, umbrales: Sequence[float] = UMBRALES_CORROSIVIDAD) -> str:
    """Clase de corrosividad de una resistividad (Ω·m)."""
    if not np.isfinite(rho):
        return "—"
    else:
        pass
    for umbral, clase in zip(umbrales, CLASES_CORROSIVIDAD):
        if rho < umbral:
            return clase
        else:
            pass
    return CLASES_CORROSIVIDAD[-1]


def parametros_sondeo(
    multicapa: pd.DataFrame,
    simplificado: pd.DataFrame,
    profundidad: Optional[float] = None,
    doi: Optional[float] = None,
    prof_corrosion: float = 3.0,
    umbrales_corrosion: Sequence[float] = UMBRALES_CORROSIVIDAD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tabla de parámetros de un sondeo.

    Parameters
    ----------
    profundidad : float, optional
        Profundidad de cálculo de Dar Zarrouk. Por defecto la DOI (o la
        base de la última capa finita del simplificado si no hay DOI).
    prof_corrosion : float
        Profundidad del suelo evaluado para corrosividad (m).

    Returns
    -------
    (tabla, resumen)
        ``tabla``: una fila por modelo (multicapa / simplificado) con los
        parámetros. ``resumen``: valores escalares para el catálogo.
    """
    if profundidad is None or not np.isfinite(profundidad) or profundidad <= 0:
        candidatos = [v for v in (doi,) if v is not None and np.isfinite(v) and v > 0]
        if candidatos:
            profundidad = float(candidatos[0])
        else:
            finitas = simplificado["z_base"].replace(np.inf, np.nan).dropna()
            profundidad = float(finitas.max()) if len(finitas) else 10.0
    else:
        pass

    filas = []
    for nombre, modelo in (("Multicapa", multicapa), ("Simplificado", simplificado)):
        dz = dar_zarrouk(modelo, profundidad)
        suelo = dar_zarrouk(modelo, prof_corrosion)
        filas.append({
            "modelo": nombre, "profundidad_m": profundidad, **dz,
            "rho_suelo_L": suelo["rho_L"],
            "corrosividad": clase_corrosividad(suelo["rho_L"], umbrales_corrosion),
        })
    tabla = pd.DataFrame(filas)

    mc = tabla.iloc[0]
    sp = tabla.iloc[1]
    resumen = {
        "profundidad": profundidad,
        "S": float(mc["S"]), "T": float(mc["T"]),
        "rho_L": float(mc["rho_L"]), "rho_T": float(mc["rho_T"]),
        "lambda": float(mc["lambda"]), "rho_m": float(mc["rho_m"]),
        "rho_suelo": float(mc["rho_suelo_L"]),
        "corrosividad": str(mc["corrosividad"]),
        "diferencia_S_pct": float(100 * (sp["S"] / mc["S"] - 1)) if mc["S"] else np.nan,
        "diferencia_T_pct": float(100 * (sp["T"] / mc["T"] - 1)) if mc["T"] else np.nan,
        "prof_corrosion": prof_corrosion,
    }
    return tabla, resumen


def corrosividad_por_capa(modelo: pd.DataFrame, umbrales=UMBRALES_CORROSIVIDAD) -> pd.Series:
    """Clase de corrosividad de cada capa de un modelo."""
    return modelo["rho"].apply(lambda r: clase_corrosividad(r, umbrales))


def unidades_geoelectricas(
    capas: pd.DataFrame, n_max: int = 6, semilla: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa las capas de todos los sondeos en unidades geoeléctricas.

    Parameters
    ----------
    capas : DataFrame
        Capas de los modelos simplificados de la campaña, con columnas
        ``Sondeo, capa, z_techo, z_base, rho, m_ip``.

    Returns
    -------
    (capas_con_unidad, resumen)
        ``unidad`` numerada de la más conductora (U1) a la más
        resistiva, y un resumen estadístico por unidad.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    tabla = capas.copy().reset_index(drop=True)
    if len(tabla) < 3:
        tabla["unidad"] = "U1"
        return tabla, pd.DataFrame()
    else:
        pass

    base_finita = np.where(np.isfinite(tabla["z_base"]), tabla["z_base"], tabla["z_techo"] * 1.5 + 1.0)
    profundidad_media = np.sqrt(np.maximum(tabla["z_techo"], 0.1) * np.maximum(base_finita, 0.2))
    atributos = [np.log10(tabla["rho"].to_numpy(float)), 0.5 * np.log10(profundidad_media)]
    usa_ip = bool((tabla["m_ip"] > 0).any())
    if usa_ip:
        atributos.append(np.log10(1.0 + tabla["m_ip"].to_numpy(float)))
    else:
        pass
    X = StandardScaler().fit_transform(np.column_stack(atributos))
    # La profundidad pesa la mitad: agrupa por propiedad eléctrica, no por
    # cota, pero desempata capas parecidas a profundidades muy distintas.
    X[:, 1] *= 0.5

    mejor, mejor_bic = None, np.inf
    for n in range(1, int(min(n_max, len(tabla) - 1)) + 1):
        modelo = GaussianMixture(n_components=n, covariance_type="full", random_state=semilla,
                                 n_init=3, reg_covar=1e-3).fit(X)
        bic = modelo.bic(X)
        if bic < mejor_bic:
            mejor, mejor_bic = modelo, bic
        else:
            pass

    etiquetas = mejor.predict(X)
    orden = (pd.Series(np.log10(tabla["rho"].to_numpy(float))).groupby(etiquetas).mean()
             .sort_values().index.tolist())
    renombre = {original: f"U{i + 1}" for i, original in enumerate(orden)}
    tabla["unidad"] = [renombre[e] for e in etiquetas]
    tabla["prob_unidad"] = mejor.predict_proba(X).max(axis=1)

    resumen = tabla.groupby("unidad").agg(
        n_capas=("rho", "size"),
        rho_mediana=("rho", "median"),
        rho_p10=("rho", lambda v: float(np.percentile(v, 10))),
        rho_p90=("rho", lambda v: float(np.percentile(v, 90))),
        z_techo_mediana=("z_techo", "median"),
        m_ip_mediana=("m_ip", "median"),
        n_sondeos=("Sondeo", "nunique"),
    ).reset_index()
    return tabla, resumen
