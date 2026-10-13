"""
almacenamiento_sev.py
==================================================================
Capa de persistencia de proyectos SEV/IP.

Misma arquitectura que TlalliRes2D: **Parquet particionado por sondeo
+ catálogo SQLite + consultas DuckDB**.

- **Parquet** (``<etapa>/SEV=<id>/<tabla>.parquet``): fuente de verdad
  de los datos de cada sondeo.
- **Catálogo SQLite**: qué sondeos existen, su arreglo, coordenadas,
  estado por etapa y métricas. La interfaz navega sin abrir Parquet.
- **DuckDB**: consultas multi-sondeo (mapa, unidades, perfiles).

Etapas y tablas
------------------
=================  ===============================================
Etapa              Tablas
=================  ===============================================
``01_crudo``       ``lectura``, ``empalmes``, ``modelo_referencia``
``02_promedio``    ``promedio``
``03_procesado``   ``procesado``
``04_inversion``   ``multicapa``, ``simplificado``, ``ajuste``,
                   ``seleccion``, ``lambdas``
``05_parametros``  ``parametros``, ``unidades``, ``resumen_unidades``
``06_perfiles``    ``perfil_sondeos``, ``perfil_lci``,
                   ``perfil_ajuste``, ``perfil_ruta``
=================  ===============================================

Las tablas de campaña (unidades geoeléctricas) se guardan bajo el
sondeo reservado ``CAMPANA``; las de perfiles, bajo el nombre del
perfil.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import pandas as pd

logger = logging.getLogger("sev.almacenamiento")


class AlmacenamientoError(Exception):
    """Error base de la capa de persistencia."""


class ProyectoNoEncontradoError(AlmacenamientoError, FileNotFoundError):
    """La ruta indicada no contiene un proyecto SEV válido."""


class EtapaDesconocidaError(AlmacenamientoError, ValueError):
    """Se solicitó una etapa/tabla que no existe en el esquema."""


ETAPAS: dict[str, str] = {
    "crudo": "01_crudo",
    "promedio": "02_promedio",
    "procesado": "03_procesado",
    "inversion": "04_inversion",
    "parametros": "05_parametros",
    "perfiles": "06_perfiles",
}

TABLAS_POR_ETAPA: dict[str, tuple[str, ...]] = {
    "crudo": ("lectura", "empalmes", "modelo_referencia"),
    "promedio": ("promedio",),
    "procesado": ("procesado",),
    "inversion": ("multicapa", "simplificado", "ajuste", "seleccion", "lambdas", "zohdy", "pruebas_bloques", "proceso"),
    "parametros": ("parametros", "unidades", "resumen_unidades"),
    "perfiles": ("perfil_sondeos", "perfil_lci", "perfil_ajuste", "perfil_ruta"),
}

#: Identificador reservado para las tablas de campaña.
CAMPANA = "CAMPANA"

NOMBRE_CATALOGO = "_catalogo.sqlite"
NOMBRE_CONFIG = "_proyecto.json"
CARPETA_ENTRADAS = "_entradas"


def _validar_etapa(etapa: str) -> None:
    if etapa not in ETAPAS:
        raise EtapaDesconocidaError(f"Etapa '{etapa}' desconocida. Opciones: {list(ETAPAS)}")
    else:
        pass


def _validar_tabla(etapa: str, tabla: str) -> None:
    _validar_etapa(etapa)
    if tabla not in TABLAS_POR_ETAPA[etapa]:
        raise EtapaDesconocidaError(
            f"Tabla '{tabla}' no pertenece a la etapa '{etapa}'. Válidas: {TABLAS_POR_ETAPA[etapa]}"
        )
    else:
        pass


def sanear_id(identificador: Any) -> str:
    """Nombre seguro para carpeta de partición."""
    texto = str(identificador).strip()
    for caracter in '/\\:*?"<>| \t\n=':
        texto = texto.replace(caracter, "_")
    return texto if texto else "SIN_NOMBRE"


@dataclass
class Proyecto:
    """Un proyecto SEV en disco."""

    ruta: Path
    nombre: str
    cliente: Optional[str] = None
    anio: Optional[str] = None

    @property
    def ruta_catalogo(self) -> Path:
        return self.ruta / NOMBRE_CATALOGO

    def ruta_etapa(self, etapa: str) -> Path:
        _validar_etapa(etapa)
        return self.ruta / ETAPAS[etapa]

    def ruta_tabla(self, etapa: str, tabla: str, sondeo: Any) -> Path:
        _validar_tabla(etapa, tabla)
        return self.ruta_etapa(etapa) / f"SEV={sanear_id(sondeo)}" / f"{tabla}.parquet"

    def as_dict(self) -> dict[str, Any]:
        return {"nombre": self.nombre, "cliente": self.cliente, "anio": self.anio,
                "ruta": str(self.ruta), "tipo": "SEV"}


def crear_proyecto(
    ruta_base: Union[str, Path], nombre: str, cliente: Optional[str] = None,
    anio: Optional[str] = None, sobrescribir: bool = False,
) -> Proyecto:
    """Crea la estructura de carpetas y el catálogo de un proyecto."""
    ruta = Path(ruta_base) / sanear_id(nombre)
    if ruta.exists() and sobrescribir:
        shutil.rmtree(ruta)
    else:
        pass
    ruta.mkdir(parents=True, exist_ok=True)
    for carpeta in ETAPAS.values():
        (ruta / carpeta).mkdir(exist_ok=True)
    proyecto = Proyecto(ruta=ruta, nombre=nombre, cliente=cliente, anio=anio)
    (ruta / NOMBRE_CONFIG).write_text(
        json.dumps(proyecto.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    _inicializar_catalogo(proyecto)
    logger.info("Proyecto creado: %s", ruta)
    return proyecto


def abrir_proyecto(ruta: Union[str, Path]) -> Proyecto:
    """Abre un proyecto existente."""
    ruta = Path(ruta)
    config = ruta / NOMBRE_CONFIG
    if not config.exists():
        raise ProyectoNoEncontradoError(f"'{ruta}' no es un proyecto válido (falta {NOMBRE_CONFIG}).")
    else:
        pass
    datos = json.loads(config.read_text(encoding="utf-8"))
    if datos.get("tipo", "SEV") != "SEV":
        raise ProyectoNoEncontradoError("El archivo es un proyecto de otra aplicación (no SEV).")
    else:
        pass
    proyecto = Proyecto(ruta=ruta, nombre=datos.get("nombre", ruta.name),
                        cliente=datos.get("cliente"), anio=datos.get("anio"))
    _inicializar_catalogo(proyecto)
    return proyecto


# ======================================================================
# Catálogo
# ======================================================================

_SQL_CREAR = """
CREATE TABLE IF NOT EXISTS sondeos (
    sondeo         TEXT PRIMARY KEY,
    archivo_origen TEXT,
    arreglo        TEXT,
    convencion     TEXT,
    n_registros    INTEGER,
    n_direcciones  INTEGER,
    ab2_max        REAL,
    tiene_ip       INTEGER,
    x              REAL,
    y              REAL,
    z              REAL,
    creado_en      TEXT
);
CREATE TABLE IF NOT EXISTS etapas (
    sondeo TEXT NOT NULL, etapa TEXT NOT NULL, tabla TEXT NOT NULL,
    n_filas INTEGER, ruta_relativa TEXT, actualizado_en TEXT,
    PRIMARY KEY (sondeo, etapa, tabla)
);
CREATE TABLE IF NOT EXISTS metricas (
    sondeo TEXT NOT NULL, etapa TEXT NOT NULL, clave TEXT NOT NULL,
    valor_num REAL, valor_texto TEXT, actualizado_en TEXT,
    PRIMARY KEY (sondeo, etapa, clave)
);
"""


@contextmanager
def _conectar(proyecto: Proyecto):
    """Conexión al catálogo que **confirma y cierra** al salir.

    ``with sqlite3.connect(...)`` solo confirma la transacción: la
    conexión queda abierta hasta que el recolector la libera, lo que en
    una sesión larga acumula descriptores y avisos de recursos.
    """
    conexion = sqlite3.connect(proyecto.ruta_catalogo)
    try:
        yield conexion
        conexion.commit()
    finally:
        conexion.close()


def _inicializar_catalogo(proyecto: Proyecto) -> None:
    with _conectar(proyecto) as conexion:
        conexion.executescript(_SQL_CREAR)


def _ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def registrar_sondeo(
    proyecto: Proyecto, sondeo: Any, archivo_origen: Optional[str] = None,
    arreglo: Optional[str] = None, convencion: Optional[str] = None,
    n_registros: Optional[int] = None, n_direcciones: Optional[int] = None,
    ab2_max: Optional[float] = None, tiene_ip: bool = False,
) -> None:
    """Registra o actualiza un sondeo (sin tocar sus coordenadas)."""
    with _conectar(proyecto) as conexion:
        conexion.execute(
            """
            INSERT INTO sondeos (sondeo, archivo_origen, arreglo, convencion, n_registros,
                                 n_direcciones, ab2_max, tiene_ip, creado_en)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sondeo) DO UPDATE SET
                archivo_origen=excluded.archivo_origen, arreglo=excluded.arreglo,
                convencion=excluded.convencion, n_registros=excluded.n_registros,
                n_direcciones=excluded.n_direcciones, ab2_max=excluded.ab2_max,
                tiene_ip=excluded.tiene_ip
            """,
            (str(sondeo), archivo_origen, arreglo, convencion, n_registros, n_direcciones,
             ab2_max, int(bool(tiene_ip)), _ahora()),
        )


def fijar_coordenadas(proyecto: Proyecto, coordenadas: pd.DataFrame) -> int:
    """Asigna X, Y, Z a los sondeos registrados. Devuelve cuántos cambió."""
    cambiados = 0
    with _conectar(proyecto) as conexion:
        for _, fila in coordenadas.iterrows():
            cursor = conexion.execute(
                "UPDATE sondeos SET x=?, y=?, z=? WHERE sondeo=?",
                (float(fila["X"]), float(fila["Y"]), float(fila.get("Z", 0.0) or 0.0),
                 str(fila["Sondeo"])),
            )
            cambiados += cursor.rowcount
    return cambiados


def coordenadas(proyecto: Proyecto) -> pd.DataFrame:
    """Sondeos con coordenadas (``Sondeo, X, Y, Z``)."""
    with _conectar(proyecto) as conexion:
        df = pd.read_sql_query(
            "SELECT sondeo AS Sondeo, x AS X, y AS Y, z AS Z FROM sondeos "
            "WHERE x IS NOT NULL AND y IS NOT NULL ORDER BY sondeo", conexion)
    df["Z"] = df["Z"].fillna(0.0)
    return df


def registrar_metricas(proyecto: Proyecto, sondeo: Any, etapa: str, metricas: dict[str, Any]) -> None:
    """Guarda métricas escalares (numéricas o de texto) en el catálogo."""
    _validar_etapa(etapa)
    ahora = _ahora()
    filas = []
    for clave, valor in metricas.items():
        if isinstance(valor, bool):
            filas.append((str(sondeo), etapa, clave, None, str(valor), ahora))
        else:
            if isinstance(valor, (int, float)):
                filas.append((str(sondeo), etapa, clave, float(valor), None, ahora))
            else:
                if hasattr(valor, "item"):
                    filas.append((str(sondeo), etapa, clave, float(valor.item()), None, ahora))
                else:
                    try:
                        texto = valor if isinstance(valor, str) else json.dumps(valor, default=str)
                        filas.append((str(sondeo), etapa, clave, None, texto, ahora))
                    except (TypeError, ValueError):
                        logger.debug("Métrica no serializable: %s", clave)
    if not filas:
        return None
    else:
        pass
    with _conectar(proyecto) as conexion:
        conexion.executemany(
            """
            INSERT INTO metricas (sondeo, etapa, clave, valor_num, valor_texto, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sondeo, etapa, clave) DO UPDATE SET
                valor_num=excluded.valor_num, valor_texto=excluded.valor_texto,
                actualizado_en=excluded.actualizado_en
            """, filas)
    return None


def guardar_tabla(
    proyecto: Proyecto, df: pd.DataFrame, etapa: str, tabla: str, sondeo: Any,
    metricas: Optional[dict[str, Any]] = None,
) -> Path:
    """Guarda un DataFrame en la partición del sondeo y lo anota."""
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    a_guardar = df.copy()
    for columna in a_guardar.columns:
        if a_guardar[columna].dtype == object:
            a_guardar[columna] = a_guardar[columna].astype(str)
        else:
            pass
    a_guardar.to_parquet(ruta, index=False, compression="snappy")
    with _conectar(proyecto) as conexion:
        conexion.execute(
            """
            INSERT INTO etapas (sondeo, etapa, tabla, n_filas, ruta_relativa, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sondeo, etapa, tabla) DO UPDATE SET
                n_filas=excluded.n_filas, ruta_relativa=excluded.ruta_relativa,
                actualizado_en=excluded.actualizado_en
            """,
            (str(sondeo), etapa, tabla, len(a_guardar), str(ruta.relative_to(proyecto.ruta)), _ahora()),
        )
    if metricas:
        registrar_metricas(proyecto, sondeo, etapa, metricas)
    else:
        pass
    return ruta


def cargar_tabla(
    proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any, columnas: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """Carga una tabla de un sondeo (``None`` si no existe)."""
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    if not ruta.exists():
        return None
    else:
        return pd.read_parquet(ruta, columns=columnas)


def existe_tabla(proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any) -> bool:
    return proyecto.ruta_tabla(etapa, tabla, sondeo).exists()


def eliminar_tabla(proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any) -> None:
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    if ruta.exists():
        ruta.unlink()
    else:
        pass
    with _conectar(proyecto) as conexion:
        conexion.execute("DELETE FROM etapas WHERE sondeo=? AND etapa=? AND tabla=?",
                         (str(sondeo), etapa, tabla))


def eliminar_sondeo(proyecto: Proyecto, sondeo: Any) -> None:
    """Elimina todos los datos de un sondeo."""
    for etapa in ETAPAS:
        carpeta = proyecto.ruta_etapa(etapa) / f"SEV={sanear_id(sondeo)}"
        if carpeta.exists():
            shutil.rmtree(carpeta)
        else:
            pass
    with _conectar(proyecto) as conexion:
        for tabla in ("sondeos", "etapas", "metricas"):
            conexion.execute(f"DELETE FROM {tabla} WHERE sondeo = ?", (str(sondeo),))


def listar_sondeos(proyecto: Proyecto) -> pd.DataFrame:
    """Sondeos registrados (desde el catálogo)."""
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query("SELECT * FROM sondeos ORDER BY sondeo", conexion)


def estado_pipeline(proyecto: Proyecto, sondeo: Optional[Any] = None) -> pd.DataFrame:
    consulta = "SELECT sondeo, etapa, tabla, n_filas, actualizado_en FROM etapas"
    parametros: tuple = ()
    if sondeo is not None:
        consulta += " WHERE sondeo = ?"
        parametros = (str(sondeo),)
    else:
        pass
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query(consulta + " ORDER BY sondeo, etapa, tabla", conexion, params=parametros)


def obtener_metricas(proyecto: Proyecto, sondeo: Any, etapa: Optional[str] = None) -> pd.DataFrame:
    consulta = "SELECT etapa, clave, valor_num, valor_texto FROM metricas WHERE sondeo = ?"
    parametros: list = [str(sondeo)]
    if etapa is not None:
        consulta += " AND etapa = ?"
        parametros.append(etapa)
    else:
        pass
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query(consulta, conexion, params=parametros)


def metricas_campana(proyecto: Proyecto, etapa: str) -> pd.DataFrame:
    """Métricas de una etapa de todos los sondeos, en formato ancho."""
    with _conectar(proyecto) as conexion:
        df = pd.read_sql_query(
            "SELECT sondeo, clave, valor_num, valor_texto FROM metricas WHERE etapa = ?",
            conexion, params=(etapa,))
    if len(df) == 0:
        return pd.DataFrame()
    else:
        pass
    df["valor"] = df["valor_num"].where(df["valor_num"].notna(), df["valor_texto"])
    return df.pivot(index="sondeo", columns="clave", values="valor").reset_index()


# ======================================================================
# Consultas multi-sondeo (DuckDB)
# ======================================================================


def consultar_multisondeo(proyecto: Proyecto, etapa: str, tabla: str) -> pd.DataFrame:
    """Tabla de todos los sondeos (columna ``SEV`` desde la partición)."""
    _validar_tabla(etapa, tabla)
    patron = str(proyecto.ruta_etapa(etapa) / "SEV=*" / f"{tabla}.parquet")
    try:
        with duckdb.connect() as conexion:
            df = conexion.execute(
                f"SELECT * FROM read_parquet('{patron}', hive_partitioning=true, union_by_name=true)"
            ).fetch_df()
    except (duckdb.IOException, duckdb.InvalidInputException):
        return pd.DataFrame()
    if "SEV" in df.columns:
        df["SEV"] = df["SEV"].astype(str)
        df = df[df["SEV"] != CAMPANA]
    else:
        pass
    return df.reset_index(drop=True)


def tamano_proyecto(proyecto: Proyecto) -> dict[str, Any]:
    total = sum(f.stat().st_size for f in proyecto.ruta.rglob("*") if f.is_file())
    return {"total_mb": round(total / 1024 / 1024, 3)}


def rango_color_campana(proyecto: Proyecto, percentil_bajo: float = 1.5,
                        percentil_alto: float = 98.5) -> Optional[list[float]]:
    """Rango log10 de resistividad de todos los modelos multicapa."""
    df = consultar_multisondeo(proyecto, "inversion", "multicapa")
    if len(df) == 0:
        return None
    else:
        pass
    import numpy as np

    valores = np.log10(df["rho"].to_numpy(float))
    valores = valores[np.isfinite(valores)]
    if len(valores) < 2:
        return None
    else:
        pass
    bajo, alto = np.percentile(valores, [percentil_bajo, percentil_alto])
    if alto <= bajo:
        alto = bajo + 1.0
    else:
        pass
    return [float(bajo), float(alto)]
