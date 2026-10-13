"""
lectura_sev.py
==================================================================
Lectura y estandarización de archivos de Sondeo Eléctrico Vertical
(SEV) con o sin Polarización Inducida (IP).

Formato de entrada
---------------------
Texto plano (``.txt``, ``.csv``, ``.dat``) con las columnas::

    Sondeo, ID, a, AB/2, MN/2, K, I, V, R, ResApp, Direccion, IP

- El separador se detecta solo (tabulador, coma, punto y coma o
  espacios) y se admite coma decimal.
- Los nombres de columna se reconocen sin importar mayúsculas, acentos,
  espacios ni sinónimos frecuentes (``Rhoa``, ``AB2``, ``Dirección``,
  ``Cargabilidad``…).
- ``IP`` es opcional: si falta, vale cero (y la línea se trata como
  solo resistividad).
- ``Direccion`` es opcional: si falta, todo el sondeo es una única
  dirección ``"U"``.
- ``K``, ``R`` y ``ResApp`` son redundantes entre sí y con la
  geometría; se conservan tal cual vienen y se **recalculan** en
  :mod:`procesamiento_sev`, donde la discrepancia sirve de control de
  calidad.
- Sin encabezado, se asume el orden de columnas de arriba.

Un archivo puede contener **varios sondeos** (columna ``Sondeo``); cada
uno se procesa como una unidad independiente del proyecto, igual que
una línea en la versión 2D.

También se lee aquí el archivo opcional de **coordenadas** de los
sondeos (``Sondeo, X, Y, Z``), necesario para el mapa y los perfiles.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("sev.lectura")


class LecturaSEVError(Exception):
    """El archivo no se pudo interpretar como datos de SEV."""


#: Columnas estándar, en el orden del formato de entrada.
COLUMNAS_ESTANDAR = (
    "Sondeo", "ID", "a", "AB2", "MN2", "K", "I", "V", "R", "ResApp",
    "Direccion", "IP",
)

#: Sinónimos aceptados (ya normalizados: minúsculas, sin acentos ni
#: separadores) para cada columna estándar.
SINONIMOS: dict[str, tuple[str, ...]] = {
    "Sondeo": ("sondeo", "sev", "sounding", "estacion", "punto", "nombre", "station", "ves"),
    "ID": ("id", "no", "num", "numero", "n", "lectura", "idx", "index"),
    "a": ("a", "aespaciamiento", "espaciamiento", "dipolo", "aa"),
    "AB2": ("ab2", "abmedios", "ab", "semiab", "l"),
    "MN2": ("mn2", "mnmedios", "mn", "semimn", "b"),
    "K": ("k", "kgeom", "factorgeometrico", "factork", "geometricfactor"),
    "I": ("i", "ima", "corriente", "current", "intensidad"),
    "V": ("v", "vmv", "voltaje", "voltage", "dv", "potencial", "deltav", "vp"),
    "R": ("r", "rohm", "resistencia", "resistance", "vi"),
    "ResApp": ("resapp", "rhoa", "resap", "resistividadaparente", "rho", "ra",
               "pa", "rhoap", "appres", "rapp", "resistividad"),
    "Direccion": ("direccion", "dir", "azimut", "azimuth", "orientacion", "rumbo",
                  "direction"),
    "IP": ("ip", "cargabilidad", "m", "ma", "mvv", "chargeability", "msec", "ms"),
}

SINONIMOS_COORDENADAS: dict[str, tuple[str, ...]] = {
    "Sondeo": SINONIMOS["Sondeo"],
    "X": ("x", "este", "easting", "e", "utme", "lon", "longitud", "xutm"),
    "Y": ("y", "norte", "northing", "utmn", "lat", "latitud", "yutm"),
    "Z": ("z", "cota", "elevacion", "elev", "altitud", "h", "msnm", "zutm"),
}


def _normalizar(texto: Any) -> str:
    """Nombre de columna normalizado para comparar sinónimos."""
    crudo = unicodedata.normalize("NFKD", str(texto))
    sin_acentos = "".join(c for c in crudo if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", sin_acentos.lower())


def _mapear_columnas(columnas: list[str], sinonimos: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Asocia cada columna del archivo con su nombre estándar.

    Primero las coincidencias exactas y después las de sinónimo, para
    que una columna llamada literalmente ``R`` no se la quede
    ``ResApp`` por un sinónimo más general.
    """
    asignadas: dict[str, str] = {}
    usadas: set[str] = set()
    normalizadas = {c: _normalizar(c) for c in columnas}

    for estandar in sinonimos:
        objetivo = _normalizar(estandar)
        for original, norm in normalizadas.items():
            if original in usadas:
                continue
            else:
                pass
            if norm == objetivo:
                asignadas[original] = estandar
                usadas.add(original)
                break
            else:
                pass

    for estandar, lista in sinonimos.items():
        if estandar in asignadas.values():
            continue
        else:
            pass
        for original, norm in normalizadas.items():
            if original in usadas:
                continue
            else:
                pass
            if norm in lista:
                asignadas[original] = estandar
                usadas.add(original)
                break
            else:
                pass
    return asignadas


def _texto_desde(fuente: Union[str, Path, bytes]) -> str:
    """Contenido de texto de una ruta o de bytes, probando codificaciones."""
    if isinstance(fuente, (bytes, bytearray)):
        crudo = bytes(fuente)
    else:
        crudo = Path(fuente).read_bytes()
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return crudo.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return crudo.decode("utf-8", errors="replace")


def _es_numero(texto: str) -> bool:
    try:
        float(str(texto).replace(",", "."))
        return True
    except ValueError:
        return False


def _leer_tabla(texto: str) -> pd.DataFrame:
    """Lee una tabla de texto con separador y decimal desconocidos.

    Se descartan líneas vacías y comentarios (``#``, ``//``). Si la
    primera línea útil es numérica, no hay encabezado.
    """
    lineas = [
        l for l in texto.splitlines()
        if l.strip() and not l.strip().startswith(("#", "//"))
    ]
    if not lineas:
        raise LecturaSEVError("El archivo está vacío.")
    else:
        pass

    muestra = "\n".join(lineas[:20])
    if "\t" in muestra:
        separador = "\t"
    else:
        if ";" in muestra:
            separador = ";"
        else:
            # Coma como separador solo si no parece coma decimal: con coma
            # decimal y espacios como separador, cada línea tiene muchos
            # tokens con espacios.
            if "," in muestra and len(re.split(r"\s+", lineas[-1].strip())) <= 2:
                separador = ","
            else:
                if "," in muestra and lineas[0].count(",") >= 3:
                    separador = ","
                else:
                    separador = r"\s+"

    decimal = "."
    if separador in ("\t", ";", r"\s+"):
        cuerpo = "\n".join(lineas[1:20])
        if re.search(r"\d,\d", cuerpo) and not re.search(r"\d\.\d", cuerpo):
            decimal = ","
        else:
            pass
    else:
        pass

    primera = re.split(separador if separador != r"\s+" else r"\s+", lineas[0].strip())
    hay_encabezado = sum(_es_numero(t) for t in primera) < max(1, len(primera) // 2)

    tabla = pd.read_csv(
        io.StringIO("\n".join(lineas)),
        sep=separador, engine="python", decimal=decimal,
        header=0 if hay_encabezado else None, skipinitialspace=True,
    )
    tabla.columns = [str(c).strip() for c in tabla.columns]
    if not hay_encabezado:
        nombres = list(COLUMNAS_ESTANDAR[: tabla.shape[1]])
        if tabla.shape[1] > len(COLUMNAS_ESTANDAR):
            nombres += [f"extra_{i}" for i in range(tabla.shape[1] - len(COLUMNAS_ESTANDAR))]
        else:
            pass
        tabla.columns = nombres
    else:
        pass
    return tabla


def _a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte a número aceptando coma decimal residual."""
    if serie.dtype == object:
        serie = serie.astype(str).str.replace(",", ".", regex=False).str.strip()
    else:
        pass
    return pd.to_numeric(serie, errors="coerce")


def leer_sev(
    fuente: Union[str, Path, bytes],
    nombre_sondeo_defecto: str = "SEV-1",
) -> pd.DataFrame:
    """Lee un archivo de SEV/IP y lo devuelve en el esquema estándar.

    Parameters
    ----------
    fuente : str, Path or bytes
        Ruta del archivo o su contenido.
    nombre_sondeo_defecto : str
        Nombre para los registros cuando el archivo no trae columna
        ``Sondeo``.

    Returns
    -------
    pandas.DataFrame
        Columnas :data:`COLUMNAS_ESTANDAR` (numéricas salvo ``Sondeo`` y
        ``Direccion``) más ``tiene_ip`` (bool por fila).

    Raises
    ------
    LecturaSEVError
        Si faltan las columnas mínimas (una distancia y una medida).
    """
    tabla = _leer_tabla(_texto_desde(fuente))
    mapeo = _mapear_columnas(list(tabla.columns), SINONIMOS)
    tabla = tabla.rename(columns=mapeo)

    df = pd.DataFrame(index=tabla.index)
    for columna in COLUMNAS_ESTANDAR:
        if columna in tabla.columns:
            df[columna] = tabla[columna]
        else:
            df[columna] = np.nan

    df["Sondeo"] = (
        df["Sondeo"].where(df["Sondeo"].notna(), nombre_sondeo_defecto).astype(str).str.strip()
    )
    df["Direccion"] = (
        df["Direccion"].where(df["Direccion"].notna(), "U").astype(str).str.strip().str.upper()
    )
    df.loc[df["Direccion"].isin(["", "NAN", "NONE"]), "Direccion"] = "U"

    for columna in ("ID", "a", "AB2", "MN2", "K", "I", "V", "R", "ResApp", "IP"):
        df[columna] = _a_numerico(df[columna])

    sin_distancia = df[["a", "AB2"]].isna().all(axis=1)
    if sin_distancia.all():
        raise LecturaSEVError(
            "No se encontró ninguna columna de distancia (AB/2 o a). Columnas leídas: "
            f"{list(tabla.columns)}"
        )
    else:
        pass

    tiene_medida = df["ResApp"].notna() | (df["V"].notna() & df["I"].notna()) | df["R"].notna()
    if not tiene_medida.any():
        raise LecturaSEVError(
            "No hay medidas: hace falta ResApp, o bien V e I, o R. Columnas leídas: "
            f"{list(tabla.columns)}"
        )
    else:
        pass

    df["tiene_ip"] = df["IP"].notna() & (df["IP"].abs() > 0)
    df["IP"] = df["IP"].fillna(0.0)

    # ID consecutivo por sondeo y dirección cuando falta.
    if df["ID"].isna().any():
        df["ID"] = df.groupby(["Sondeo", "Direccion"]).cumcount() + 1
    else:
        pass
    df["ID"] = df["ID"].astype(int)

    descartadas = int((sin_distancia | ~tiene_medida).sum())
    if descartadas:
        logger.warning("leer_sev: %d fila(s) sin distancia o sin medida se descartan.", descartadas)
    else:
        pass
    df = df.loc[~sin_distancia & tiene_medida].reset_index(drop=True)
    logger.info(
        "leer_sev: %d registros, %d sondeo(s), direcciones=%s",
        len(df), df["Sondeo"].nunique(), sorted(df["Direccion"].unique()),
    )
    return df


def leer_coordenadas(fuente: Union[str, Path, bytes]) -> pd.DataFrame:
    """Lee el archivo de coordenadas ``Sondeo, X, Y, Z``.

    ``Z`` es opcional (cero si falta): sin cota, el perfil se dibuja en
    profundidad en vez de en elevación.
    """
    tabla = _leer_tabla(_texto_desde(fuente))
    if not any(_normalizar(c) in SINONIMOS_COORDENADAS["X"] + ("x",) for c in tabla.columns):
        # Sin encabezado: orden fijo.
        nombres = ["Sondeo", "X", "Y", "Z"][: tabla.shape[1]]
        tabla.columns = nombres + [f"extra_{i}" for i in range(tabla.shape[1] - len(nombres))]
    else:
        tabla = tabla.rename(columns=_mapear_columnas(list(tabla.columns), SINONIMOS_COORDENADAS))

    faltan = [c for c in ("Sondeo", "X", "Y") if c not in tabla.columns]
    if faltan:
        raise LecturaSEVError(f"Al archivo de coordenadas le faltan las columnas {faltan}.")
    else:
        pass

    df = pd.DataFrame({
        "Sondeo": tabla["Sondeo"].astype(str).str.strip(),
        "X": _a_numerico(tabla["X"]),
        "Y": _a_numerico(tabla["Y"]),
        "Z": _a_numerico(tabla["Z"]) if "Z" in tabla.columns else 0.0,
    })
    df["Z"] = df["Z"].fillna(0.0)
    return df.dropna(subset=["X", "Y"]).drop_duplicates("Sondeo", keep="last").reset_index(drop=True)


def leer_modelo_referencia(fuente: Union[str, Path, bytes]) -> pd.DataFrame:
    """Lee un modelo de capas conocido (pozo, sintético) para comparar.

    Admite columnas ``espesor, rho[, m_ip]`` (el último espesor vacío o
    ``inf`` es el semiespacio) o ``z_techo, rho[, m_ip]``. Devuelve el
    esquema de los modelos de la aplicación: ``capa, z_techo, z_base,
    espesor, rho, m_ip``.
    """
    texto = _texto_desde(fuente).replace("inf", "nan").replace("Inf", "nan")
    tabla = _leer_tabla(texto)
    norm = {_normalizar(c): c for c in tabla.columns}
    rho_col = next((norm[k] for k in ("rho", "resistividad", "res", "ohmm") if k in norm), None)
    if rho_col is None:
        raise LecturaSEVError("El modelo de referencia necesita una columna 'rho'.")
    else:
        pass
    rho = _a_numerico(tabla[rho_col]).to_numpy(float)
    ip_col = next((norm[k] for k in ("mip", "m", "ip", "cargabilidad") if k in norm), None)
    m_ip = _a_numerico(tabla[ip_col]).fillna(0).to_numpy(float) if ip_col else np.zeros(len(rho))
    if "ztecho" in norm or "techo" in norm or "z" in norm:
        z = _a_numerico(tabla[norm.get("ztecho", norm.get("techo", norm.get("z")))]).to_numpy(float)
        bases = np.concatenate([z[1:], [np.inf]])
        techos = z
    else:
        h_col = next((norm[k] for k in ("espesor", "h", "thk", "espesorm") if k in norm), None)
        if h_col is None:
            raise LecturaSEVError("El modelo de referencia necesita 'espesor' o 'z_techo'.")
        else:
            pass
        h = _a_numerico(tabla[h_col]).to_numpy(float)
        h = np.where(np.isfinite(h), h, np.inf)
        h[-1] = np.inf
        bases = np.cumsum(h)
        techos = np.concatenate([[0.0], bases[:-1]])
    return pd.DataFrame({"capa": np.arange(1, len(rho) + 1), "z_techo": techos, "z_base": bases,
                         "espesor": bases - techos, "rho": rho, "m_ip": m_ip})
