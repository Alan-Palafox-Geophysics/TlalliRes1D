"""
procesamiento_sev.py
==================================================================
Preparación, control de calidad y preprocesamiento de un SEV.

Cadena de un sondeo
----------------------
1. **Recálculo y validación** (:func:`preparar_crudo`)
   Se detecta el arreglo (por la columna K, ver :mod:`geometria_sev`),
   se recalculan K, R = V/I y ρa = K·R, y se marcan las discrepancias
   con lo que traía el archivo. Si ρa del archivo y K·V/I difieren en
   una potencia de diez casi exacta (mV frente a V, mA frente a A), se
   corrige la escala y se avisa en vez de propagar el error.
2. **Corrección de empalmes** (:func:`corregir_empalmes`)
   En Schlumberger, al ampliar MN aparecen saltos entre tramos de la
   curva. La parte no física (heterogeneidad junto a M y N) se corrige
   con un factor multiplicativo por tramo, calculado con la media
   geométrica de las razones en los AB/2 repetidos. El primer tramo es
   la referencia.
3. **Promedio de direcciones** (:func:`promediar_direcciones`)
   Media **geométrica** (en log ρa, porque la resistividad es
   log-normal) de las direcciones medidas (NS, EW…), interpolando en
   log-log cuando no comparten las mismas aperturas. La dispersión entre
   direcciones se conserva como estimación del error de cada punto, y
   el cociente ρmáx/ρmín como índice de anisotropía azimutal.
4. **Preprocesamiento** (:func:`preprocesar`)
   Detección de datos anómalos sobre la curva promedio y su
   tratamiento, con un modelo de error explícito para la inversión.

Detección de anómalos (modo automático)
------------------------------------------
Se usa un criterio **guiado por la física**: la curva se ajusta con una
inversión suave robusta (pyGIMLi, IRLS que imita la norma L1) y se mide
cuánto se aparta cada dato de la mejor curva que **un medio 1D puede
producir**. Un dato que ningún modelo estratificado reproduce es ruido
o efecto lateral, y es justo lo que debe atenuarse. Para declararlo
anómalo se exigen dos condiciones a la vez:

- significación estadística: |z| > umbral, con z escalado por la MAD
  (desviación absoluta mediana) de los residuales;
- relevancia práctica: |residual| > umbral de error relativo.

Se añade la restricción física de pendiente: en Schlumberger y Wenner la
rama ascendente de la curva log-log no puede superar 45° (pendiente 1);
una pendiente mayor solo puede venir de error de medición.

Tratamientos disponibles
---------------------------
``suavizar`` (por defecto)
    El dato anómalo se sustituye por la curva ajustada y su error se
    eleva al tamaño del residual, para que pese en proporción a la
    confianza que merece.
``ponderar``
    El dato se conserva tal cual y solo se eleva su error.
``eliminar``
    El dato no entra en la inversión.

Si ningún dato supera los umbrales, la curva preprocesada es idéntica a
la observada.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

import geometria_sev as geo

logger = logging.getLogger("sev.procesamiento")


class ProcesamientoSEVError(Exception):
    """El sondeo no tiene datos suficientes para procesarse."""


#: Mínimo de mediciones válidas para considerar un sondeo procesable.
MIN_DATOS = 5


# ======================================================================
# 1. Recálculo y validación
# ======================================================================


def _potencia_diez(razon: float) -> int:
    """Exponente entero si ``razon`` es casi una potencia de diez (≠ 1)."""
    if not np.isfinite(razon) or razon <= 0:
        return 0
    else:
        pass
    exponente = np.log10(razon)
    entero = int(np.round(exponente))
    if entero != 0 and abs(exponente - entero) < 0.02:
        return entero
    else:
        return 0


def preparar_crudo(
    df: pd.DataFrame, arreglo_forzado: Optional[str] = None,
) -> tuple[pd.DataFrame, geo.Deteccion, dict[str, Any], list[str]]:
    """Geometría, recálculo de ρa y banderas de calidad de un sondeo.

    Returns
    -------
    (tabla, deteccion, metricas, advertencias)
        ``tabla`` es la entrada con las columnas añadidas
        ``xA..xN, am, an, bm, bn, K_geom, z_mediana, R_calc, rhoa,
        qc_k, qc_rhoa, valido``.
    """
    advertencias: list[str] = []
    tabla = df.copy().sort_values(["Direccion", "AB2", "MN2"], kind="stable").reset_index(drop=True)

    deteccion = geo.detectar_arreglo(tabla, forzado=arreglo_forzado)
    if not deteccion.confirmada:
        advertencias.append(deteccion.mensaje)
    else:
        pass
    tabla = pd.concat([tabla, geo.geometria(tabla, deteccion.arreglo, deteccion.convencion)], axis=1)

    V = tabla["V"].to_numpy(float)
    I = tabla["I"].to_numpy(float)
    R_col = tabla["R"].to_numpy(float)
    K_col = tabla["K"].to_numpy(float)
    rho_col = tabla["ResApp"].to_numpy(float)
    K = tabla["K_geom"].to_numpy(float)

    hay_vi = np.isfinite(V) & np.isfinite(I) & (I != 0)
    R = np.where(hay_vi, V / np.where(hay_vi, I, 1.0), R_col)

    # Escala de unidades: si ρa del archivo y K·R difieren en una potencia
    # de diez casi exacta, el problema es de unidades (mV/V, mA/A).
    ambos = np.isfinite(R) & np.isfinite(rho_col) & (R > 0) & (rho_col > 0)
    if ambos.sum() >= 3:
        exponente = _potencia_diez(float(np.median(rho_col[ambos] / (K[ambos] * R[ambos]))))
        if exponente != 0:
            R = R * 10.0 ** exponente
            advertencias.append(
                f"V/I y ResApp difieren en un factor 10^{exponente}: se asume una "
                "diferencia de unidades (mV/V o mA/A) y se corrige R."
            )
        else:
            pass
    else:
        pass

    # ρa preferente: K de la geometría por R medido. Si no hay R, se reescala
    # el ρa del archivo con la razón K_geom/K_archivo (corrige un K mal
    # anotado), y si tampoco hay K se usa tal cual.
    rho_desde_k = np.where(np.isfinite(K_col) & (K_col > 0), rho_col * K / np.where(K_col > 0, K_col, 1.0), rho_col)
    rhoa = np.where(np.isfinite(R) & (R > 0), K * R, rho_desde_k)

    tabla["R_calc"] = R
    tabla["rhoa"] = rhoa
    with np.errstate(divide="ignore", invalid="ignore"):
        tabla["qc_k"] = np.abs(np.log(K_col / K))
        tabla["qc_rhoa"] = np.abs(np.log(rho_col / rhoa))
    tabla["valido"] = np.isfinite(rhoa) & (rhoa > 0) & np.isfinite(K)
    if hay_vi.any():
        tabla.loc[hay_vi & ((V <= 0) | (I <= 0)), "valido"] = False
    else:
        pass

    n_invalidos = int((~tabla["valido"]).sum())
    if n_invalidos:
        advertencias.append(f"{n_invalidos} medición(es) con ρa, V o I no positivos se descartan.")
    else:
        pass
    n_k = int((tabla["qc_k"] > geo.TOLERANCIA_K).sum())
    if n_k and deteccion.confirmada:
        advertencias.append(
            f"{n_k} medición(es) con K del archivo distinto del geométrico (>3 %): "
            "se usa el geométrico."
        )
    else:
        pass

    metricas = {
        "arreglo": deteccion.arreglo,
        "convencion": deteccion.convencion,
        "error_k_mediano": deteccion.error_k,
        "n_registros": int(len(tabla)),
        "n_validos": int(tabla["valido"].sum()),
        "n_direcciones": int(tabla["Direccion"].nunique()),
        "direcciones": ",".join(sorted(tabla["Direccion"].unique())),
        "tiene_ip": bool(tabla["tiene_ip"].any()),
        "ab2_min": float(np.nanmin(tabla["AB2"])),
        "ab2_max": float(np.nanmax(tabla["AB2"])),
    }
    return tabla, deteccion, metricas, advertencias


# ======================================================================
# 2. Corrección de empalmes (Schlumberger)
# ======================================================================


def _ruido_lecturas(tabla: pd.DataFrame) -> float:
    """Ruido relativo de una lectura individual (una dirección).

    Con dos o más direcciones se estima comparando las direcciones en la
    misma apertura (AB/2, MN/2): es la medida más directa del ruido de
    campo y no depende de ningún suavizado. Con una sola dirección se usa
    la dispersión robusta de la curva respecto de un LOWESS en log-log
    (algo pesimista si hay saltos de empalme, lo que vuelve más prudente
    la corrección).
    """
    validos = tabla[tabla["valido"]]
    if validos["Direccion"].nunique() >= 2:
        clave = [np.round(validos["AB2"], 6), np.round(validos["MN2"].fillna(0), 6)]
        grupos = np.log(validos["rhoa"]).groupby(clave)
        s = grupos.std(ddof=1).to_numpy(float)
        n = grupos.size().to_numpy(float)
        valor = sigma_agrupada(s, n)
        if valor > 0:
            return float(max(valor, 0.005))
        else:
            pass
    else:
        pass
    x, y = _curva_unica(validos["AB2"].to_numpy(float), validos["rhoa"].to_numpy(float))
    if len(x) < 6:
        return 0.05
    else:
        pass
    r = np.log(y) - lowess(np.log(x), np.log(y), fraccion=0.3)
    return float(max(1.4826 * np.median(np.abs(r - np.median(r))), 0.005))


def _residuales_fisicos(grupo: pd.DataFrame) -> pd.Series:
    """ln(ρa / ρa_modelo) de cada lectura respecto de un ajuste físico
    robusto (pyGIMLi) que usa la geometría **exacta** de cada lectura.

    Es lo que permite separar el salto de empalme real (estático) del
    salto físico: con MN grande respecto de AB, la resistividad aparente
    cambia legítimamente al ampliar MN, y el modelo directo con la
    geometría exacta ya lo predice.
    """
    import forward_sev as fw
    import inversion_sev as inv

    datos = grupo.sort_values("AB2")
    geom = fw.GeometriaSondeo.desde_tabla(datos)
    rho = datos["rhoa"].to_numpy(float)
    ajuste = inv.ajuste_fisico_robusto(geom, rho, np.full(len(rho), 0.05),
                                       datos["z_mediana"].to_numpy(float))
    return pd.Series(np.log(rho / ajuste), index=datos.index)


def corregir_empalmes(
    tabla: pd.DataFrame, umbral: float = 0.02, activar: bool = True, arreglo: str = "SCH",
    significancia: float = 3.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Corrige los saltos **no físicos** entre tramos de MN (Schlumberger).

    Al ampliar MN, la curva puede saltar por dos motivos: uno físico
    (con MN no despreciable frente a AB la medida cambia de verdad) y
    otro no físico (heterogeneidad junto a M y N, desplazamiento
    estático). Corregir el salto completo —como se hace a mano— borra
    también la parte física. Aquí se ajusta un modelo 1D robusto con la
    geometría exacta de cada lectura y el factor de empalme se mide
    sobre los **residuales** en los AB/2 repetidos: lo que el modelo no
    explica es el desplazamiento estático.

    Además:

    - las diferencias de los distintos traslapes deben ser coherentes
      entre sí (si no, hay una lectura aberrante en el traslape);
    - el factor solo se aplica si es estadísticamente significativo:
      mayor que ``significancia``·σ·√(2/n), con σ el ruido por lectura
      (estimado entre direcciones) y n los traslapes válidos.

    Returns
    -------
    (tabla, informe)
        ``tabla`` con ``factor_empalme`` y ``rhoa_corr``; ``informe``
        con una entrada por empalme.
    """
    salida = tabla.copy()
    salida["factor_empalme"] = 1.0
    informe: list[dict[str, Any]] = []
    if not activar or arreglo != "SCH":
        salida["rhoa_corr"] = salida["rhoa"]
        return salida, informe
    else:
        pass

    sigma = _ruido_lecturas(salida)
    for direccion, grupo in salida[salida["valido"]].groupby("Direccion"):
        tramos = sorted(grupo["MN2"].dropna().unique())
        if len(tramos) < 2:
            continue
        else:
            pass
        try:
            residuo = _residuales_fisicos(grupo)
        except Exception as exc:
            logger.warning("Empalmes de %s: ajuste físico falló (%s).", direccion, exc)
            continue
        acumulado = 1.0
        for anterior, siguiente in zip(tramos[:-1], tramos[1:]):
            filas = []
            for mn2 in (anterior, siguiente):
                tramo = grupo[grupo["MN2"] == mn2]
                filas.append(pd.Series(residuo.loc[tramo.index].to_numpy(), index=np.round(tramo["AB2"], 6))
                             .groupby(level=0).mean())
            comunes = np.intersect1d(filas[0].index, filas[1].index)
            if len(comunes) == 0:
                informe.append({"Direccion": direccion, "MN2_ref": anterior, "MN2": siguiente,
                                "n_traslape": 0, "factor": 1.0, "umbral_significancia": np.nan,
                                "coherente": True, "aplicado": False})
                continue
            else:
                pass
            diferencia = filas[0].loc[comunes].to_numpy() - filas[1].loc[comunes].to_numpy()
            factor_local = float(np.exp(np.median(diferencia)))
            # Coherencia: las diferencias de los traslapes deben coincidir
            # dentro del ruido; si no, alguna lectura del traslape es
            # aberrante y el factor no es fiable. Con un solo traslape no hay
            # con qué comparar y se exige una significancia mayor.
            coherente = len(diferencia) < 2 or float(np.ptp(diferencia)) <= max(3.0 * np.sqrt(2.0) * sigma, 0.1)
            k_sig = significancia if len(diferencia) >= 2 else significancia + 1.0
            limite = max(umbral, k_sig * sigma * np.sqrt(2.0 / len(comunes)))
            aplicar = coherente and abs(np.log(factor_local)) > limite
            if aplicar:
                acumulado = acumulado * factor_local
            else:
                pass
            mascara = (salida["Direccion"] == direccion) & (salida["MN2"] == siguiente)
            salida.loc[mascara, "factor_empalme"] = acumulado
            informe.append({"Direccion": direccion, "MN2_ref": anterior, "MN2": siguiente,
                            "n_traslape": int(len(comunes)), "factor": factor_local,
                            "umbral_significancia": float(np.exp(limite)), "coherente": bool(coherente),
                            "aplicado": bool(aplicar)})
    salida["rhoa_corr"] = salida["rhoa"] * salida["factor_empalme"]
    return salida, informe


# ======================================================================
# 3. Promedio de direcciones
# ======================================================================


def _curva_unica(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curva monovaluada en log-log: media geométrica de los x repetidos."""
    serie = pd.Series(np.log(y), index=np.round(x, 6)).groupby(level=0).mean().sort_index()
    return serie.index.to_numpy(float), np.exp(serie.to_numpy(float))


def _interpolar_loglog(x_ref, x, y) -> np.ndarray:
    """Interpola en log-log dentro del rango medido (NaN fuera)."""
    lx = np.log(x)
    valores = np.interp(np.log(x_ref), lx, np.log(y), left=np.nan, right=np.nan)
    return np.exp(valores)


def promediar_direcciones(tabla: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Curva observada final del sondeo: media geométrica de direcciones.

    La geometría de la curva promedio es la de la dirección con más
    datos válidos (dirección de referencia): así cada punto conserva su
    AB/2 y MN/2 reales y el modelado directo es exacto. Las demás
    direcciones se interpolan en log-log sobre esas aperturas, sin
    extrapolar.

    Returns
    -------
    (promedio, metricas, advertencias)
        ``promedio`` con ``rhoa_obs`` (media geométrica), ``sigma_dir``
        (desviación de log ρa entre direcciones), ``n_dir``,
        ``anisotropia`` (ρmáx/ρmín), ``ip_obs`` y una columna
        ``rhoa_<DIR>`` por dirección.
    """
    advertencias: list[str] = []
    validos = tabla[tabla["valido"]].copy()
    if len(validos) < MIN_DATOS:
        raise ProcesamientoSEVError(
            f"El sondeo solo tiene {len(validos)} mediciones válidas (mínimo {MIN_DATOS})."
        )
    else:
        pass

    columna_rho = "rhoa_corr" if "rhoa_corr" in validos.columns else "rhoa"
    conteo = validos.groupby("Direccion").size().sort_values(ascending=False)
    referencia = str(conteo.index[0])
    base = validos[validos["Direccion"] == referencia].sort_values(["AB2", "MN2"]).reset_index(drop=True)

    columnas_geo = ["a", "AB2", "MN2", "xA", "xB", "xM", "xN", "am", "an", "bm", "bn",
                    "K_geom", "z_mediana"]
    promedio = base[columnas_geo].copy()
    promedio.insert(0, "Sondeo", base["Sondeo"].iloc[0])

    log_rho = []
    ips = []
    for direccion, grupo in validos.groupby("Direccion"):
        if direccion == referencia:
            valores = base[columna_rho].to_numpy(float)
            ip = base["IP"].to_numpy(float)
        else:
            # Interpolación por tramo de MN cuando el tramo existe en ambas
            # direcciones; si no, sobre la curva única de la dirección.
            valores = np.full(len(base), np.nan)
            ip = np.full(len(base), np.nan)
            for mn2, sub_base in base.groupby("MN2"):
                sub = grupo[np.isclose(grupo["MN2"], mn2)]
                if len(sub) < 2:
                    sub = grupo
                else:
                    pass
                x_u, y_u = _curva_unica(sub["AB2"].to_numpy(float), sub[columna_rho].to_numpy(float))
                if len(x_u) < 2:
                    continue
                else:
                    pass
                indices = sub_base.index.to_numpy()
                valores[indices] = _interpolar_loglog(sub_base["AB2"].to_numpy(float), x_u, y_u)
                ip_serie = sub.groupby(np.round(sub["AB2"], 6))["IP"].mean().sort_index()
                ip[indices] = np.interp(np.log(sub_base["AB2"].to_numpy(float)),
                                        np.log(ip_serie.index.to_numpy(float)), ip_serie.to_numpy(float),
                                        left=np.nan, right=np.nan)
        promedio[f"rhoa_{direccion}"] = valores
        log_rho.append(np.log(valores))
        ips.append(ip)

    matriz = np.vstack(log_rho)
    n_dir = np.sum(np.isfinite(matriz), axis=0)
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        promedio["rhoa_obs"] = np.exp(np.nanmean(matriz, axis=0))
        promedio["sigma_dir"] = np.where(n_dir >= 2, np.nanstd(matriz, axis=0, ddof=1), np.nan)
        promedio["anisotropia"] = np.where(
            n_dir >= 2, np.exp(np.nanmax(matriz, axis=0) - np.nanmin(matriz, axis=0)), np.nan)
    promedio["n_dir"] = n_dir
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        matriz_ip = np.vstack(ips)
        promedio["ip_obs"] = np.nanmean(matriz_ip, axis=0)
        promedio["sigma_ip_dir"] = np.where(
            np.sum(np.isfinite(matriz_ip), axis=0) >= 2, np.nanstd(matriz_ip, axis=0, ddof=1), np.nan)
    promedio["ip_obs"] = promedio["ip_obs"].fillna(0.0)
    promedio["direccion_ref"] = referencia

    n_direcciones = validos["Direccion"].nunique()
    anis_mediana = float(np.nanmedian(promedio["anisotropia"])) if n_direcciones > 1 else 1.0
    if n_direcciones > 1 and anis_mediana > 1.3:
        advertencias.append(
            f"Las direcciones difieren de forma sistemática (ρmáx/ρmín mediano = "
            f"{anis_mediana:.2f}). Puede haber anisotropía azimutal o una estructura "
            "lateral (2D/3D): el modelo 1D promedio debe interpretarse con cautela."
        )
    else:
        pass

    metricas = {
        "direccion_referencia": referencia,
        "n_direcciones": int(n_direcciones),
        "anisotropia_mediana": anis_mediana,
        "anisotropia_max": float(np.nanmax(promedio["anisotropia"])) if n_direcciones > 1 else 1.0,
        "dispersion_mediana": float(np.nanmedian(promedio["sigma_dir"])) if n_direcciones > 1 else 0.0,
        "n_puntos": int(len(promedio)),
    }
    return promedio, metricas, advertencias


# ======================================================================
# 4. Suavizadores y detección de anómalos
# ======================================================================


def lowess(x: np.ndarray, y: np.ndarray, fraccion: float = 0.4, iteraciones: int = 3,
           grado: int = 1) -> np.ndarray:
    """Regresión local robusta (LOWESS, Cleveland 1979), lineal o cuadrática.

    Pesos tricúbicos en x y re-ponderación bicuadrada de los residuales
    para que un dato aberrante no arrastre la curva.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    k = int(min(n, max(3 + grado, np.ceil(fraccion * n))))
    robustez = np.ones(n)
    ajuste = y.copy()
    for _ in range(iteraciones):
        for i in range(n):
            distancia = np.abs(x - x[i])
            h = np.sort(distancia)[k - 1] + 1e-12
            w = np.clip(1 - (distancia / h) ** 3, 0, None) ** 3 * robustez
            if w.sum() <= 0:
                ajuste[i] = y[i]
                continue
            else:
                pass
            X = np.column_stack([(x - x[i]) ** p for p in range(grado + 1)])
            WX = X * w[:, None]
            try:
                coef = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)[0]
                ajuste[i] = coef[0]
            except np.linalg.LinAlgError:
                ajuste[i] = y[i]
        residuo = y - ajuste
        s = np.median(np.abs(residuo)) + 1e-12
        robustez = np.clip(1 - (residuo / (6 * s)) ** 2, 0, None) ** 2
    return ajuste


def mediana_movil(y: np.ndarray, ventana: int = 5) -> np.ndarray:
    """Mediana móvil centrada (bordes con ventana recortada)."""
    y = np.asarray(y, float)
    medio = ventana // 2
    return np.array([np.median(y[max(0, i - medio): i + medio + 1]) for i in range(len(y))])


def savgol(y: np.ndarray, ventana: int = 7, orden: int = 2) -> np.ndarray:
    """Savitzky-Golay con ventana ajustada al número de datos."""
    from scipy.signal import savgol_filter

    n = len(y)
    v = min(ventana if ventana % 2 else ventana + 1, n if n % 2 else n - 1)
    if v <= orden:
        return np.asarray(y, float)
    else:
        return savgol_filter(np.asarray(y, float), v, orden)


def sigma_agrupada(sigma: np.ndarray, n_dir: np.ndarray) -> float:
    """Ruido por dirección estimado con todas las aperturas del sondeo.

    Cada apertura aporta una varianza muestral s² con ν = n_dir − 1
    grados de libertad, y s²·ν/σ² ~ χ²_ν. La mediana de s² es robusta
    frente a aperturas con efectos laterales, pero está sesgada: para
    dos direcciones (ν = 1) vale 0.455·σ². Se corrige dividiendo por la
    mediana de χ²_ν/ν; sin esa corrección el ruido sale ~1.5 veces menor
    y la inversión ajustaría ruido.
    """
    from scipy.stats import chi2

    s = np.asarray(sigma, float)
    n = np.asarray(n_dir, float)
    validas = np.isfinite(s) & (n >= 2)
    if not validas.any():
        return 0.0
    else:
        pass
    nu = max(float(np.median(n[validas])) - 1.0, 1.0)
    return float(np.sqrt(np.median(s[validas] ** 2) / (chi2.median(nu) / nu)))


def ruido_entre_direcciones(promedio: pd.DataFrame) -> Optional[float]:
    """Ruido por lectura estimado de las diferencias entre direcciones.

    Para cada apertura se calcula la desviación de cada dirección respecto
    de la media de todas; a esas desviaciones se les resta la mediana por
    (dirección, tramo de MN), que elimina diferencias **sistemáticas**
    —anisotropía, desplazamientos estáticos de un tramo— que no son ruido
    aleatorio y que inflarían la estimación. El ruido es 1.4826·MAD de lo
    que queda, corregido por los grados de libertad (√(n/(n−1))).

    Returns
    -------
    float or None
        ``None`` si hay menos de dos direcciones.
    """
    columnas = [c for c in promedio.columns if c.startswith("rhoa_") and c not in
                ("rhoa_obs", "rhoa_ref", "rhoa_proc")]
    if len(columnas) < 2:
        return None
    else:
        pass
    matriz = np.log(promedio[columnas].to_numpy(float))
    completas = np.all(np.isfinite(matriz), axis=1)
    if completas.sum() < 4:
        return None
    else:
        pass
    m = matriz[completas]
    desv = m - m.mean(axis=1, keepdims=True)
    tramos = promedio.loc[completas, "MN2"].fillna(0).to_numpy(float)
    # Solo los tramos con suficientes lecturas cuentan como grupo propio (en
    # Wenner o dipolares cada lectura tiene su MN y restar su mediana
    # anularía la desviación); el resto forma un único grupo.
    valores, conteos = np.unique(tramos, return_counts=True)
    pequenos = valores[conteos < 4]
    tramos = np.where(np.isin(tramos, pequenos), -1.0, tramos)
    for t in np.unique(tramos):
        filas = tramos == t
        desv[filas] -= np.median(desv[filas], axis=0, keepdims=True)
    n = m.shape[1]
    mad = np.median(np.abs(desv - np.median(desv)))
    return float(max(1.4826 * mad * np.sqrt(n / (n - 1.0)), 0.001))


def ruido_por_curvatura(promedio: pd.DataFrame, columna: str = "rhoa_obs") -> float:
    """Ruido de una curva de una sola dirección (estimador de diferencias).

    Con una sola dirección no hay con qué comparar cada lectura. Se usa el
    estimador de diferencias de Rice (1984) en su versión robusta: en
    log-log, la segunda diferencia de tres aperturas consecutivas cancela
    la parte suave de la curva y deja sobre todo el ruido, con varianza
    6σ². No depende de ningún modelo ni suavizado, así que no confunde la
    estructura real de la curva con ruido.
    """
    x, y = _curva_unica(promedio["AB2"].to_numpy(float), promedio[columna].to_numpy(float))
    if len(x) < 5:
        return 0.0
    else:
        pass
    ly = np.log(y)
    dd = ly[2:] - 2 * ly[1:-1] + ly[:-2]
    return float(1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(6.0))


def escala_robusta(residuos: np.ndarray, piso: float) -> float:
    """Desviación robusta (1.4826·MAD) con piso, para escalar z.

    El piso evita que una curva casi perfecta (MAD ≈ 0) convierta
    diferencias insignificantes en z enormes.
    """
    mad = 1.4826 * np.median(np.abs(residuos - np.median(residuos)))
    return float(max(mad, piso))


@dataclass
class ConfigPreproceso:
    """Parámetros del preprocesamiento de un sondeo.

    Attributes
    ----------
    modo : {'Auto', 'Manual'}
        Auto: ajuste físico robusto y umbrales por defecto.
    metodo : {'fisico', 'lowess', 'mediana', 'savgol'}
        Curva de referencia contra la que se miden los residuales
        (Manual).
    umbral_rel : float
        Error relativo a partir del cual un dato se trata (0.10 = 10 %).
    umbral_z : float
        z robusto mínimo para declarar un dato anómalo.
    umbral_disp : float
        Dispersión entre direcciones (desv. de log ρa) a partir de la cual
        un punto se trata aunque su promedio no sea anómalo.
    piso_error : float
        Error relativo mínimo asignado a cada dato en la inversión. El
        error efectivo es el mayor entre este piso y el ruido que se mide
        en los datos (entre direcciones, o por diferencias en una sola).
    tratamiento : {'suavizar', 'ponderar', 'eliminar'}
    ventana : int
        Ventana de los filtros de mediana y Savitzky-Golay.
    corregir_empalmes : bool
    umbral_empalme : float
    verificar_pendiente : bool
        Restricción de 45° en la rama ascendente (SCH/WE).
    ip_umbral_abs : float
        Desviación absoluta de IP (mV/V) para tratar un dato de IP.
    """

    modo: str = "Auto"
    metodo: str = "fisico"
    umbral_rel: float = 0.10
    umbral_z: float = 3.0
    umbral_disp: float = 0.20
    piso_error: float = 0.001
    tratamiento: str = "suavizar"
    ventana: int = 5
    corregir_empalmes: bool = True
    umbral_empalme: float = 0.02
    verificar_pendiente: bool = True
    ip_umbral_abs: float = 2.0
    arreglo_forzado: Optional[str] = None

    def efectiva(self) -> "ConfigPreproceso":
        """En modo Auto, los parámetros por defecto (física + umbrales)."""
        if self.modo == "Auto":
            return ConfigPreproceso(
                modo="Auto", corregir_empalmes=self.corregir_empalmes,
                arreglo_forzado=self.arreglo_forzado, piso_error=self.piso_error,
                tratamiento=self.tratamiento,
            )
        else:
            return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _curva_referencia(
    promedio: pd.DataFrame, metodo: str, cfg: ConfigPreproceso,
) -> np.ndarray:
    """Curva suave contra la que se miden los residuales (en Ω·m)."""
    x = np.log(promedio["AB2"].to_numpy(float))
    y = np.log(promedio["rhoa_obs"].to_numpy(float))
    if metodo == "fisico":
        import forward_sev as fw
        import inversion_sev as inv

        geom = fw.GeometriaSondeo.desde_tabla(promedio)
        n_dir = promedio["n_dir"].to_numpy(float) if "n_dir" in promedio else np.ones(len(promedio))
        agrupada = ruido_entre_direcciones(promedio)
        if agrupada is None:
            agrupada = sigma_agrupada(promedio["sigma_dir"].to_numpy(float), n_dir)
        else:
            pass
        error = np.maximum(cfg.piso_error, agrupada / np.sqrt(np.maximum(n_dir, 1.0)))
        try:
            return inv.ajuste_fisico_robusto(geom, promedio["rhoa_obs"].to_numpy(float), error,
                                             promedio["z_mediana"].to_numpy(float))
        except Exception as exc:  # pyGIMLi puede fallar con datos degenerados
            logger.warning("Ajuste físico robusto falló (%s); se usa LOWESS.", exc)
            metodo = "lowess"
    else:
        pass

    orden = np.argsort(x, kind="stable")
    ajuste = np.empty_like(y)
    if metodo == "mediana":
        ajuste[orden] = mediana_movil(y[orden], cfg.ventana)
    else:
        if metodo == "savgol":
            ajuste[orden] = savgol(y[orden], cfg.ventana)
        else:
            ajuste[orden] = lowess(x[orden], y[orden], fraccion=0.4)
    return np.exp(ajuste)


def _violaciones_pendiente(x: np.ndarray, y: np.ndarray, residuos: np.ndarray,
                           limite: float = 1.15) -> np.ndarray:
    """Puntos responsables de una rama ascendente > 45° en log-log."""
    marca = np.zeros(len(x), dtype=bool)
    orden = np.argsort(x, kind="stable")
    lx = np.log(x[orden])
    ly = np.log(y[orden])
    for i in range(len(orden) - 1):
        dx = lx[i + 1] - lx[i]
        if dx <= 1e-6:
            continue
        else:
            pass
        if (ly[i + 1] - ly[i]) / dx > limite:
            # Se culpa al punto que más se aparta de la curva de referencia.
            culpable = orden[i] if abs(residuos[orden[i]]) > abs(residuos[orden[i + 1]]) else orden[i + 1]
            marca[culpable] = True
        else:
            pass
    return marca


def preprocesar(
    promedio: pd.DataFrame, config: Optional[ConfigPreproceso] = None, arreglo: str = "SCH",
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Detecta y trata datos anómalos de la curva promedio.

    Returns
    -------
    (procesado, metricas, advertencias)
        ``procesado`` con ``rhoa_ref`` (curva de referencia),
        ``residuo`` (log), ``z``, ``anomalo``, ``motivo``,
        ``rhoa_proc``, ``error_rel``, ``usar`` y las equivalentes de IP
        (``ip_ref``, ``ip_anomalo``, ``ip_proc``, ``error_ip``).
    """
    cfg = (config or ConfigPreproceso()).efectiva()
    advertencias: list[str] = []
    tabla = promedio.copy().reset_index(drop=True)
    if len(tabla) < MIN_DATOS:
        raise ProcesamientoSEVError(f"Solo hay {len(tabla)} puntos en la curva promedio.")
    else:
        pass

    rho = tabla["rhoa_obs"].to_numpy(float)
    referencia = _curva_referencia(tabla, cfg.metodo, cfg)
    residuo = np.log(rho / referencia)
    escala = escala_robusta(residuo, piso=cfg.piso_error / 2)
    z = residuo / escala
    sigma = np.nan_to_num(tabla["sigma_dir"].to_numpy(float), nan=0.0)

    # Error de cada dato: error estándar de la media de sus direcciones,
    # con el ruido por dirección estimado de forma AGRUPADA sobre todo el
    # sondeo. La desviación de solo dos lecturas es una estimación muy
    # inestable punto a punto (puede salir casi cero por azar) y usarla
    # como peso individual daría pesos arbitrarios.
    n_dir = np.maximum(tabla["n_dir"].to_numpy(float), 1.0) if "n_dir" in tabla else np.ones(len(tabla))
    ruido_dir = ruido_entre_direcciones(tabla)
    if ruido_dir is None:
        ruido_dir = ruido_por_curvatura(tabla)
        n_dir = np.ones(len(tabla))
    else:
        pass
    # Error efectivo: el ruido medido en los propios datos, nunca menor que
    # el piso. Con datos de campo el ruido medido manda (2–10 % típico); con
    # datos muy limpios el piso evita pesos infinitos.
    error_base = np.maximum(cfg.piso_error, ruido_dir / np.sqrt(n_dir))

    anomalo_estadistico = (np.abs(z) > cfg.umbral_z) & (np.abs(residuo) > np.log1p(cfg.umbral_rel))
    disperso = (sigma > cfg.umbral_disp) & (sigma > 2.5 * ruido_dir)
    pendiente = np.zeros(len(tabla), dtype=bool)
    if cfg.verificar_pendiente and arreglo in ("SCH", "WE"):
        pendiente = _violaciones_pendiente(tabla["AB2"].to_numpy(float), rho, residuo)
        pendiente = pendiente & (np.abs(residuo) > cfg.piso_error)
    else:
        pass

    # Dispersión entre direcciones: si una dirección se aparta de la
    # referencia física y la otra no, la discrepante se descarta y el
    # promedio se rehace con las coherentes. Es filtrar la dispersión con
    # criterio, en vez de promediar un dato bueno con uno malo.
    columnas_dir = [c for c in tabla.columns if c.startswith("rhoa_") and c not in
                    ("rhoa_obs", "rhoa_ref", "rhoa_proc")]
    rho_coherente = rho.copy()
    descartadas: list[str] = [""] * len(tabla)
    tolerancia = max(np.log1p(cfg.umbral_rel), 2.0 * escala)
    for i in np.where(disperso | anomalo_estadistico)[0]:
        valores = {c[5:]: tabla.at[i, c] for c in columnas_dir if np.isfinite(tabla.at[i, c])}
        if len(valores) < 2:
            continue
        else:
            pass
        buenas = {d: v for d, v in valores.items() if abs(np.log(v / referencia[i])) <= tolerancia}
        if 0 < len(buenas) < len(valores):
            rho_coherente[i] = float(np.exp(np.mean(np.log(list(buenas.values())))))
            descartadas[i] = ",".join(sorted(set(valores) - set(buenas)))
        else:
            pass
    rescatado = np.array([bool(d) for d in descartadas])

    anomalo = anomalo_estadistico | disperso | pendiente
    motivos = []
    for i in range(len(tabla)):
        partes = []
        if anomalo_estadistico[i]:
            partes.append(f"residual {100 * (np.exp(abs(residuo[i])) - 1):.0f} % (z={z[i]:.1f})")
        else:
            pass
        if disperso[i]:
            partes.append(f"dispersión entre direcciones {100 * sigma[i]:.0f} %")
        else:
            pass
        if pendiente[i]:
            partes.append("pendiente > 45°")
        else:
            pass
        if rescatado[i]:
            partes.append(f"se descarta la dirección {descartadas[i]}")
        else:
            pass
        motivos.append("; ".join(partes))

    rhoa_proc = rho.copy()
    error_rel = error_base.copy()
    usar = np.ones(len(tabla), dtype=bool)
    tratamiento = cfg.tratamiento
    # Valor de sustitución: el promedio de las direcciones coherentes si se
    # pudo rescatar; si no, la curva de referencia.
    sustituto = np.where(rescatado, rho_coherente, referencia)
    if tratamiento == "suavizar":
        rhoa_proc = np.where(anomalo, sustituto, rho)
        error_rel = np.where(anomalo & ~rescatado, np.maximum(error_base, np.abs(residuo)),
                             np.where(rescatado, np.maximum(cfg.piso_error, ruido_dir), error_base))
    else:
        if tratamiento == "ponderar":
            error_rel = np.where(anomalo, np.maximum(error_base, 3 * np.abs(residuo)), error_base)
        else:
            usar = ~anomalo

    tabla["rhoa_ref"] = referencia
    tabla["residuo"] = residuo
    tabla["z"] = z
    tabla["anomalo"] = anomalo
    tabla["motivo"] = motivos
    tabla["rhoa_proc"] = rhoa_proc
    tabla["error_rel"] = error_rel
    tabla["usar"] = usar

    # --- IP: suavizado robusto en escala lineal ---
    ip = tabla["ip_obs"].to_numpy(float)
    tiene_ip = bool(np.any(np.abs(ip) > 0))
    if tiene_ip:
        x = np.log(tabla["AB2"].to_numpy(float))
        orden = np.argsort(x, kind="stable")
        ip_ref = np.empty_like(ip)
        ip_ref[orden] = lowess(x[orden], ip[orden], fraccion=0.35, grado=2)
        r_ip = ip - ip_ref
        z_ip = r_ip / escala_robusta(r_ip, piso=max(0.25, cfg.ip_umbral_abs / 4))
        ip_anomalo = ((np.abs(z_ip) > cfg.umbral_z) & (np.abs(r_ip) > cfg.ip_umbral_abs)) | (ip < -cfg.ip_umbral_abs)
        tabla["ip_ref"] = ip_ref
        tabla["ip_anomalo"] = ip_anomalo
        # Mismo modelo de error que la resistividad: ruido por dirección
        # agrupado (corregido por sesgo) dividido entre √n, con piso de
        # 0.1 mV/V y 1 % relativo (con datos reales manda el ruido medido).
        if "sigma_ip_dir" in tabla:
            ruido_ip = sigma_agrupada(tabla["sigma_ip_dir"].to_numpy(float), n_dir)
        else:
            ruido_ip = 0.0
        # La dispersión de la curva promedio alrededor de su suavizado es la
        # medida más directa del ruido efectivo de cada dato de IP (incluye
        # todo lo que el promedio no eliminó).
        dispersion_ip = 1.4826 * float(np.median(np.abs(r_ip - np.median(r_ip))))
        error_ip = np.maximum.reduce([np.full(len(ip), 0.1), 0.01 * np.abs(ip),
                                      np.full(len(ip), ruido_ip) / np.sqrt(n_dir),
                                      np.full(len(ip), dispersion_ip)])
        if tratamiento == "suavizar":
            tabla["ip_proc"] = np.where(ip_anomalo, np.clip(ip_ref, 0, None), ip)
            tabla["error_ip"] = np.where(ip_anomalo, np.maximum(error_ip, np.abs(r_ip)), error_ip)
        else:
            tabla["ip_proc"] = np.clip(ip, 0, None)
            tabla["error_ip"] = np.where(ip_anomalo, np.maximum(error_ip, 3 * np.abs(r_ip)), error_ip)
    else:
        tabla["ip_ref"] = 0.0
        tabla["ip_anomalo"] = False
        tabla["ip_proc"] = 0.0
        tabla["error_ip"] = 1.0

    n_anomalos = int(anomalo.sum())
    requirio = n_anomalos > 0 or int(tabla["ip_anomalo"].sum()) > 0
    if not requirio:
        advertencias.append("La curva no requirió preprocesamiento: la preprocesada es la observada.")
    else:
        pass
    if n_anomalos > len(tabla) * 0.3:
        advertencias.append(
            f"{n_anomalos} de {len(tabla)} puntos superan los umbrales: más de un 30 %. "
            "Revise el umbral o la calidad del sondeo antes de interpretar."
        )
    else:
        pass

    metricas = {
        "modo": cfg.modo,
        "metodo": cfg.metodo,
        "tratamiento": tratamiento,
        "n_puntos": int(len(tabla)),
        "n_anomalos": n_anomalos,
        "n_ip_anomalos": int(tabla["ip_anomalo"].sum()),
        "requirio_preproceso": bool(requirio),
        "residuo_rms": float(np.sqrt(np.mean(residuo ** 2))),
        "error_mediano": float(np.median(error_rel)),
        "ruido_por_direccion": ruido_dir,
        "n_direcciones_descartadas": int(rescatado.sum()),
        "umbral_rel": cfg.umbral_rel,
        "tiene_ip": tiene_ip,
    }
    return tabla, metricas, advertencias


def curva_para_zohdy(procesado: pd.DataFrame) -> pd.DataFrame:
    """Curva monovaluada (un punto por AB/2) para el método de Zohdy.

    En los AB/2 repetidos por cambio de MN se promedia en log y se
    conserva la geometría del tramo de MN mayor, que es la que mejor
    señal tiene a esa apertura.
    """
    datos = procesado[procesado["usar"]].copy()
    datos["_clave"] = np.round(datos["AB2"], 6)
    datos["_log"] = np.log(datos["rhoa_proc"])
    filas = []
    for _, grupo in datos.groupby("_clave", sort=True):
        fila = grupo.sort_values("MN2").iloc[-1].copy()
        fila["rhoa_proc"] = float(np.exp(grupo["_log"].mean()))
        fila["ip_proc"] = float(grupo["ip_proc"].mean())
        filas.append(fila)
    curva = pd.DataFrame(filas).drop(columns=["_clave", "_log"]).reset_index(drop=True)
    return curva.sort_values("z_mediana", kind="stable").reset_index(drop=True)
