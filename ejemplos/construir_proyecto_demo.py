"""
construir_proyecto_demo.py
==================================================================
Procesa los SEV sintéticos de punta a punta sin interfaz y escribe el
proyecto ``ejemplos/Sinteticos_demo.isev``, listo para abrirlo en la
pantalla 1 de la aplicación.

Uso::

    python ejemplos/generar_sinteticos.py
    python ejemplos/construir_proyecto_demo.py

    # Modelo B en su propia carpeta
    python ejemplos/generar_sinteticos.py --modelo B --carpeta ejemplos/modelo_B
    python ejemplos/construir_proyecto_demo.py --carpeta ejemplos/modelo_B --nombre "Sinteticos modelo B"
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

import acciones_app as acc  # noqa: E402
import almacenamiento_sev as alm  # noqa: E402
import lectura_sev as lec  # noqa: E402
import pipeline_sev as pl  # noqa: E402

CARPETA = Path(__file__).resolve().parent


def main(carpeta: Path = CARPETA, nombre: str = "Sinteticos demo") -> None:
    base = Path(tempfile.mkdtemp())
    CARPETA = Path(carpeta)
    try:
        proyecto = alm.crear_proyecto(base, nombre, cliente="Validación", anio="2026")
        ruta = acc.guardar_archivo_subido(proyecto, "sev_sinteticos.txt",
                                          (CARPETA / "sev_sinteticos.txt").read_bytes())
        resultado = acc.accion_ejecutar_archivo(proyecto, ruta, pl.ConfigPipeline())
        print(resultado.mensaje)
        print(acc.accion_cargar_coordenadas(proyecto, CARPETA / "coordenadas_sinteticos.txt").mensaje)
        referencia = lec.leer_modelo_referencia(CARPETA / "modelo_verdadero.txt")
        for sondeo in alm.listar_sondeos(proyecto)["sondeo"]:
            alm.guardar_tabla(proyecto, referencia, "crudo", "modelo_referencia", sondeo)
        print(acc.accion_unidades(proyecto).mensaje)
        coords = alm.coordenadas(proyecto)
        import perfil_sev as pf

        print(acc.accion_perfil(proyecto, "P1", pf.ruta_automatica(coords), 250.0, True, 20.0, 1.0).mensaje)
        paquete = acc.accion_empaquetar_proyecto(proyecto, usuario="demo")
        destino = CARPETA / (nombre.replace(" ", "_") + ".isev")
        destino.write_bytes(paquete.datos["contenido"])
        print(f"Proyecto escrito en {destino} ({paquete.metricas['tamano_mb']} MB)")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proyecto .isev de demostración.")
    parser.add_argument("--carpeta", default=str(CARPETA), help="carpeta con los archivos sintéticos")
    parser.add_argument("--nombre", default="Sinteticos demo", help="nombre del proyecto")
    argumentos = parser.parse_args()
    main(Path(argumentos.carpeta), argumentos.nombre)
