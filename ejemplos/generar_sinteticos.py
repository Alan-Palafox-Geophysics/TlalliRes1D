"""
generar_sinteticos.py
==================================================================
Genera SEV sintéticos (Schlumberger y Wenner, dos direcciones, IP) con
ruido conocido para verificar la aplicación de punta a punta.

Modelo verdadero (7 capas, 100 m de interés)::

    0–2 m     conductor   10 Ω·m   25 mV/V
    2–10 m    medio       60 Ω·m    8 mV/V   (el enunciado daba 5–10; 2–5 se asume medio)
    10–20 m   conductor   10 Ω·m   25 mV/V
    20–40 m   resistivo  300 Ω·m    3 mV/V
    40–70 m   conductor   10 Ω·m   25 mV/V
    70–90 m   resistivo  300 Ω·m    3 mV/V
    90–∞      medio       60 Ω·m    8 mV/V

Por defecto las curvas son **exactas** (sin ruido), con AB/2 hasta
500 m. Opcionalmente (``--ruido 0.10 --aberrantes --desplazamiento``):

- ruido multiplicativo log-normal en cada lectura de cada dirección;
- valores aberrantes deliberados (×2.5, ×0.4, ×1.8) en lecturas
  conocidas, listados en ``ruido_inyectado.csv``;
- desplazamiento estático de +15 % en el tramo de MN mayor de la
  dirección EW de los Schlumberger.

Seis sondeos a lo largo de un perfil E-W (S1…S6, 100 m de separación,
Schlumberger y Wenner alternados) permiten además ver pseudosecciones y
la inversión lateralmente restringida.

Uso::

    python ejemplos/generar_sinteticos.py                     # sin ruido
    python ejemplos/generar_sinteticos.py --ruido 0.10 --aberrantes --desplazamiento
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import forward_sev as fw  # noqa: E402

#: Modelos disponibles: espesores (m), ρ (Ω·m) y m (mV/V).
MODELOS = {
    # A: 7 capas con contrastes fuertes (10/300 Ω·m).
    "A": (np.array([2.0, 8.0, 10.0, 20.0, 30.0, 20.0]),
          np.array([10.0, 60.0, 10.0, 300.0, 10.0, 300.0, 60.0]),
          np.array([25.0, 8.0, 25.0, 3.0, 25.0, 3.0, 8.0])),
    # B: 6 capas con contrastes moderados (0-5-15-30-50-80 m).
    "B": (np.array([5.0, 10.0, 15.0, 20.0, 30.0]),
          np.array([50.0, 100.0, 20.0, 60.0, 10.0, 75.0]),
          np.array([10.0, 6.0, 20.0, 8.0, 25.0, 7.0])),
}
ESPESORES, RHO, IP = MODELOS["A"]
RUIDO = 0.0
CORRIENTE_MA = 200.0
#: AB/2 máximo (m). Con 500 m la profundidad mediana de investigación
#: llega a ~190 m (Schlumberger) y ~170 m (Wenner): lo necesario para
#: resolver un modelo cuya última interfaz está a 90 m.
AB2_MAX = 500.0


def _geometria_sch():
    """AB/2 de 1 m a AB2_MAX (~9 aperturas por década) con tres tramos de
    MN y traslapes."""
    ab2 = np.round(np.logspace(0, np.log10(AB2_MAX), int(round(9 * np.log10(AB2_MAX))) + 1), 2)
    filas = []
    for mn2, (lo, hi) in ((0.5, (1.0, 16.0)), (5.0, (10.0, 160.0)), (25.0, (100.0, AB2_MAX))):
        for a in ab2[(ab2 >= lo) & (ab2 <= hi)]:
            filas.append((np.nan, a, mn2))
    return filas


def _geometria_wen():
    """a de 1 m a AB2_MAX/1.5 (~9 aperturas por década)."""
    a_max = AB2_MAX / 1.5
    n = int(round(9 * np.log10(a_max))) + 1
    return [(a, 1.5 * a, a / 2.0) for a in np.round(np.logspace(0, np.log10(a_max), n), 2)]


def generar(semilla: int = 2026, ruido: float | None = None, aberrantes: bool = False,
            desplazamiento: bool = False, modelo: str = "A") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera los seis SEV.

    Parameters
    ----------
    ruido : float, optional
        Ruido multiplicativo (σ de ln ρa). Por defecto :data:`RUIDO` (0:
        curvas exactas).
    aberrantes : bool
        Inyecta tres lecturas aberrantes por sondeo (dirección NS).
    desplazamiento : bool
        +15 % en el tramo de MN mayor de la dirección EW (Schlumberger).
    modelo : str
        Clave de :data:`MODELOS`.
    """
    espesores, rho_modelo, ip_modelo = MODELOS[modelo]
    rng = np.random.default_rng(semilla)
    sigma = RUIDO if ruido is None else float(ruido)
    filas, inyectados = [], []
    for i in range(6):
        nombre = f"S{i + 1}"
        es_sch = i % 2 == 0
        geometria = _geometria_sch() if es_sch else _geometria_wen()
        a = np.array([g[0] for g in geometria], float)
        ab2 = np.array([g[1] for g in geometria], float)
        mn2 = np.array([g[2] for g in geometria], float)
        if es_sch:
            A, B, M, N = -ab2, ab2, -mn2, mn2
        else:
            A, B, M, N = -1.5 * a, 1.5 * a, -0.5 * a, 0.5 * a
        geom = fw.GeometriaSondeo(np.abs(M - A), np.abs(N - A), np.abs(M - B), np.abs(N - B))
        rho_verdadera = fw.respuesta(geom, espesores, rho_modelo)
        ip_verdadera = fw.cargabilidad_aparente(geom, espesores, rho_modelo, ip_modelo)
        K = geom.k
        for direccion in ("NS", "EW"):
            rho = rho_verdadera * np.exp(rng.normal(0.0, sigma, len(ab2))) if sigma > 0 else rho_verdadera.copy()
            if sigma > 0:
                ip = ip_verdadera * np.exp(rng.normal(0.0, sigma, len(ab2))) + rng.normal(0, 0.3, len(ab2))
            else:
                ip = ip_verdadera.copy()
            if desplazamiento and es_sch and direccion == "EW":
                rho = np.where(mn2 == mn2.max(), rho * 1.15, rho)
            else:
                pass
            if aberrantes and direccion == "NS":
                # Aberrantes deliberados en posiciones fijas.
                for indice, factor in ((6, 2.5), (14, 0.4), (19, 1.8)):
                    rho[indice] *= factor
                    inyectados.append({"Sondeo": nombre, "Direccion": direccion, "AB2": ab2[indice],
                                       "MN2": mn2[indice], "factor": factor})
            else:
                pass
            V = rho * CORRIENTE_MA / K
            for j in range(len(ab2)):
                filas.append({
                    "Sondeo": nombre, "ID": j + 1, "a": a[j], "AB/2": ab2[j], "MN/2": mn2[j],
                    "K": round(K[j], 4), "I": CORRIENTE_MA, "V": round(V[j], 6),
                    "R": round(V[j] / CORRIENTE_MA, 8), "ResApp": round(rho[j], 4),
                    "Direccion": direccion, "IP": round(max(ip[j], 0.0), 3),
                })
    datos = pd.DataFrame(filas)
    coordenadas = pd.DataFrame({"Sondeo": [f"S{i + 1}" for i in range(6)],
                                "X": 500000.0 + 100.0 * np.arange(6), "Y": 2150000.0 + 3.0 * np.arange(6),
                                "Z": 2240.0 - 1.5 * np.arange(6)})
    return datos, coordenadas, pd.DataFrame(inyectados, columns=["Sondeo", "Direccion", "AB2", "MN2", "factor"])


def modelo_verdadero(modelo: str = "A") -> pd.DataFrame:
    espesores, rho_modelo, ip_modelo = MODELOS[modelo]
    return pd.DataFrame({"espesor": list(espesores) + [np.inf], "rho": rho_modelo, "m_ip": ip_modelo})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEV sintéticos Schlumberger/Wenner.")
    parser.add_argument("--ruido", type=float, default=0.0, help="σ del ruido relativo (0 = exacto)")
    parser.add_argument("--aberrantes", action="store_true", help="inyectar lecturas aberrantes")
    parser.add_argument("--desplazamiento", action="store_true", help="salto de +15 %% en un tramo de MN")
    parser.add_argument("--modelo", default="A", choices=sorted(MODELOS), help="modelo verdadero")
    parser.add_argument("--carpeta", default=None, help="carpeta de salida (por defecto ejemplos/)")
    argumentos = parser.parse_args()
    carpeta = Path(argumentos.carpeta) if argumentos.carpeta else Path(__file__).resolve().parent
    carpeta.mkdir(parents=True, exist_ok=True)
    datos, coords, aberrantes = generar(ruido=argumentos.ruido, aberrantes=argumentos.aberrantes,
                                        desplazamiento=argumentos.desplazamiento, modelo=argumentos.modelo)
    datos.to_csv(carpeta / "sev_sinteticos.txt", sep="\t", index=False, na_rep="")
    coords.to_csv(carpeta / "coordenadas_sinteticos.txt", sep="\t", index=False)
    aberrantes.to_csv(carpeta / "ruido_inyectado.csv", index=False)
    modelo_verdadero(argumentos.modelo).to_csv(carpeta / "modelo_verdadero.txt", sep="\t", index=False, na_rep="inf")
    print(f"{len(datos)} lecturas de {datos['Sondeo'].nunique()} sondeos escritas en {carpeta}")
