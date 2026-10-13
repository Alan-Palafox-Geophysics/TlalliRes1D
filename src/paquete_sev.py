"""
paquete_sev.py
==================================================================
Empaquetado y restauración de proyectos en formato **``.isev``**, y
gestión de los espacios de trabajo temporales del servidor.

Modelo de operación (aplicación alojada, sin custodia de datos)
------------------------------------------------------------------
La aplicación se ejecuta en un servidor remoto, y un navegador **no
puede** dar acceso al disco del usuario: solo permite subir archivos y
descargarlos. Por eso el proyecto no puede "vivir" en una ruta local
mientras la app lo edita.

El diseño que sí cumple el objetivo de no custodiar información:

1. El usuario **sube** sus datos crudos, o un ``.isev`` de una sesión
   anterior.
2. La aplicación trabaja en un **espacio temporal propio de esa
   sesión**, con el mismo autoguardado por etapas de siempre.
3. El usuario **descarga** el ``.isev`` actualizado y los resultados.
4. Al cerrar la sesión —o al caducar por inactividad— el espacio
   temporal **se elimina**.

Así el archivo ``.isev`` de la PC del usuario es la fuente de verdad
del proyecto, y el servidor no conserva nada tras la sesión.

Sobre el formato
-------------------
Un ``.isev`` es un ZIP con la estructura completa del proyecto
(Parquet por etapa, catálogo SQLite, archivos de entrada) más un
``manifiesto.json`` con versión de formato, fecha, licencia que lo
generó y líneas incluidas. El manifiesto permite rechazar archivos de
versiones incompatibles cuando el formato evolucione, en vez de fallar
con un error críptico a mitad de la restauración.

La extensión propia es identidad de producto, no protección: renombrar
a ``.zip`` permite abrirlo con cualquier descompresor. Conviene
saberlo y no confiar en ello como medida de seguridad.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("sev.paquete")

#: Extensión de los archivos de proyecto.
EXTENSION_PROYECTO = ".isev"

#: Versión del formato del paquete. Se incrementa cuando un cambio
#: rompe la compatibilidad de lectura.
VERSION_FORMATO = 1

NOMBRE_MANIFIESTO = "_manifiesto.json"

#: Prefijo de los espacios de trabajo temporales.
PREFIJO_ESPACIO = "sev_sesion_"

#: Horas tras las que un espacio temporal sin actividad se elimina.
HORAS_CADUCIDAD_ESPACIO = 8.0


# ======================================================================
# Excepciones
# ======================================================================


class PaqueteError(Exception):
    """Error base del empaquetado de proyectos."""


class PaqueteIncompatibleError(PaqueteError):
    """El archivo procede de una versión de formato no soportada."""


class PaqueteInvalidoError(PaqueteError):
    """El archivo no es un paquete de proyecto válido."""


# ======================================================================
# Espacios de trabajo temporales
# ======================================================================


def raiz_espacios() -> Path:
    """Carpeta que contiene todos los espacios de trabajo temporales."""
    raiz = Path(tempfile.gettempdir()) / "sev_espacios"
    raiz.mkdir(parents=True, exist_ok=True)
    return raiz


def crear_espacio(id_sesion: str) -> Path:
    """Crea (o recupera) el espacio de trabajo de una sesión.

    Es idempotente: si la sesión ya tenía espacio, devuelve el mismo,
    de modo que recargar la página no pierda el trabajo en curso.
    """
    espacio = raiz_espacios() / f"{PREFIJO_ESPACIO}{id_sesion}"
    espacio.mkdir(parents=True, exist_ok=True)
    marcar_actividad(espacio)
    return espacio


def marcar_actividad(espacio: Union[str, Path]) -> None:
    """Actualiza la marca de actividad del espacio.

    La limpieza automática se basa en esta marca, no en la fecha de los
    archivos: un proyecto puede pasar horas sin escribir nada mientras
    el usuario revisa resultados, y no debería considerarse abandonado
    por eso.
    """
    espacio = Path(espacio)
    if not espacio.exists():
        return None
    else:
        pass
    (espacio / ".actividad").write_text(str(time.time()), encoding="utf-8")
    return None


def eliminar_espacio(espacio: Union[str, Path]) -> None:
    """Elimina un espacio de trabajo y todo su contenido."""
    espacio = Path(espacio)
    if espacio.exists():
        shutil.rmtree(espacio, ignore_errors=True)
        logger.info("Espacio de trabajo eliminado: %s", espacio.name)
    else:
        pass


def limpiar_espacios_caducados(horas: float = HORAS_CADUCIDAD_ESPACIO) -> int:
    """Elimina los espacios sin actividad reciente.

    Es necesaria porque una sesión puede terminar **sin** pasar por el
    cierre ordenado: el usuario cierra la pestaña, se corta la
    conexión, o se reinicia el proceso. Sin esta limpieza, esos
    espacios quedarían en disco indefinidamente y el servidor acabaría
    llenándose.

    Returns
    -------
    int
        Número de espacios eliminados.
    """
    limite = time.time() - horas * 3600.0
    eliminados = 0

    for espacio in raiz_espacios().glob(f"{PREFIJO_ESPACIO}*"):
        if not espacio.is_dir():
            continue
        else:
            pass

        marca = espacio / ".actividad"
        if marca.exists():
            try:
                ultima = float(marca.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                ultima = espacio.stat().st_mtime
        else:
            ultima = espacio.stat().st_mtime

        if ultima < limite:
            shutil.rmtree(espacio, ignore_errors=True)
            eliminados += 1
        else:
            pass

    if eliminados:
        logger.info("Limpieza: %d espacio(s) de trabajo caducado(s) eliminado(s).", eliminados)
    else:
        pass
    return eliminados


def uso_disco_espacios() -> dict[str, Any]:
    """Uso de disco de los espacios temporales, para diagnóstico del
    administrador."""
    raiz = raiz_espacios()
    espacios = [d for d in raiz.glob(f"{PREFIJO_ESPACIO}*") if d.is_dir()]
    total = sum(
        f.stat().st_size for d in espacios for f in d.rglob("*") if f.is_file()
    )
    return {
        "n_espacios": len(espacios),
        "total_mb": round(total / 1024 / 1024, 2),
    }


# ======================================================================
# Empaquetado
# ======================================================================


def _construir_manifiesto(
    proyecto: Any, lineas: list[str], usuario: Optional[str]
) -> dict[str, Any]:
    """Metadatos que acompañan al paquete."""
    return {
        "formato": "isev",
        "version_formato": VERSION_FORMATO,
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "generado_por": usuario or "desconocido",
        "proyecto": {
            "nombre": proyecto.nombre,
            "cliente": proyecto.cliente,
            "anio": proyecto.anio,
        },
        "lineas": lineas,
        "n_lineas": len(lineas),
    }


def empaquetar_proyecto(
    proyecto: Any,
    lineas: Optional[list[str]] = None,
    usuario: Optional[str] = None,
) -> bytes:
    """Comprime un proyecto completo en un paquete ``.isev``.

    Se devuelve en **bytes** —no se escribe a disco— porque su destino
    es el botón de descarga del navegador; materializar un archivo
    intermedio en el servidor solo dejaría basura que limpiar.

    Parameters
    ----------
    proyecto : Proyecto
        Proyecto a empaquetar.
    lineas : list[str], optional
        Líneas incluidas, solo para el manifiesto (el contenido se
        empaqueta completo).
    usuario : str, optional
        Licencia que genera el paquete, para trazabilidad.

    Returns
    -------
    bytes
        Contenido del ``.isev``.
    """
    import io

    memoria = io.BytesIO()
    manifiesto = _construir_manifiesto(proyecto, lineas or [], usuario)

    with zipfile.ZipFile(memoria, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as paquete:
        paquete.writestr(
            NOMBRE_MANIFIESTO, json.dumps(manifiesto, indent=2, ensure_ascii=False)
        )
        for archivo in sorted(proyecto.ruta.rglob("*")):
            if not archivo.is_file():
                continue
            else:
                pass
            # La marca de actividad es interna del servidor: no forma parte
            # del proyecto y no debe viajar en el paquete.
            if archivo.name == ".actividad":
                continue
            else:
                pass
            paquete.write(archivo, archivo.relative_to(proyecto.ruta))

    contenido = memoria.getvalue()
    logger.info(
        "Proyecto '%s' empaquetado: %.2f MB", proyecto.nombre, len(contenido) / 1024 / 1024
    )
    return contenido


def nombre_paquete(nombre_proyecto: str) -> str:
    """Nombre de archivo sugerido para la descarga, con fecha."""
    seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(nombre_proyecto))
    fecha = datetime.now().strftime("%Y%m%d")
    return f"{seguro}_{fecha}{EXTENSION_PROYECTO}"


# ======================================================================
# Restauración
# ======================================================================


def leer_manifiesto(contenido: bytes) -> dict[str, Any]:
    """Lee el manifiesto de un paquete sin extraerlo por completo.

    Permite validar compatibilidad y mostrar al usuario qué contiene el
    archivo **antes** de restaurarlo.

    Raises
    ------
    PaqueteInvalidoError
        Si no es un ZIP válido o le falta el manifiesto.
    """
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(contenido)) as paquete:
            if NOMBRE_MANIFIESTO not in paquete.namelist():
                raise PaqueteInvalidoError(
                    "El archivo no contiene un manifiesto de proyecto: no parece "
                    f"un paquete {EXTENSION_PROYECTO} generado por esta aplicación."
                )
            else:
                pass
            return json.loads(paquete.read(NOMBRE_MANIFIESTO).decode("utf-8"))
    except zipfile.BadZipFile as exc:
        raise PaqueteInvalidoError(
            f"El archivo no es un paquete {EXTENSION_PROYECTO} válido (no se pudo "
            f"descomprimir): {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PaqueteInvalidoError(f"El manifiesto del paquete está dañado: {exc}") from exc


def restaurar_proyecto(
    contenido: bytes, espacio: Union[str, Path], nombre_destino: Optional[str] = None
) -> Path:
    """Extrae un paquete ``.isev`` dentro del espacio de trabajo.

    Parameters
    ----------
    contenido : bytes
        Contenido del archivo subido.
    espacio : str or Path
        Espacio de trabajo de la sesión.
    nombre_destino : str, optional
        Carpeta destino. Si es ``None``, se toma del manifiesto.

    Returns
    -------
    pathlib.Path
        Ruta del proyecto restaurado.

    Raises
    ------
    PaqueteInvalidoError, PaqueteIncompatibleError
    """
    import io

    manifiesto = leer_manifiesto(contenido)
    if manifiesto.get("formato") != "isev":
        raise PaqueteIncompatibleError(
            "El archivo no es un proyecto de SEV (.isev): parece de otra aplicación "
            f"(formato '{manifiesto.get('formato', 'desconocido')}', p. ej. TlalliRes2D)."
        )
    else:
        pass
    version = int(manifiesto.get("version_formato", 0))
    if version > VERSION_FORMATO:
        raise PaqueteIncompatibleError(
            f"El paquete fue creado con una versión más reciente de la aplicación "
            f"(formato {version}, esta versión admite hasta {VERSION_FORMATO}). "
            "Actualice la aplicación para abrirlo."
        )
    else:
        pass

    nombre = nombre_destino or manifiesto.get("proyecto", {}).get("nombre") or "proyecto"
    seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(nombre))
    destino = Path(espacio) / seguro

    if destino.exists():
        shutil.rmtree(destino, ignore_errors=True)
    else:
        pass
    destino.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(contenido)) as paquete:
        for miembro in paquete.namelist():
            # Defensa contra rutas maliciosas ("zip slip"): un paquete
            # manipulado podría contener entradas como '../../etc/algo' y
            # escribir fuera del espacio de trabajo.
            ruta_final = (destino / miembro).resolve()
            if not str(ruta_final).startswith(str(destino.resolve())):
                raise PaqueteInvalidoError(
                    f"El paquete contiene una ruta no permitida: '{miembro}'."
                )
            else:
                pass
        paquete.extractall(destino)

    (destino / NOMBRE_MANIFIESTO).unlink(missing_ok=True)
    marcar_actividad(espacio)
    logger.info("Proyecto restaurado en %s (%d líneas)", destino, manifiesto.get("n_lineas", 0))
    return destino
