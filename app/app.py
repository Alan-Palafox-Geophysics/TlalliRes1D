"""
app.py
==================================================================
Punto de entrada de **TLALLI-RES 1D** — SEV/IP.

Ejecutar con::

    streamlit run app/app.py

Arquitectura (idéntica a TlalliRes2D)
----------------------------------------
- Páginas separadas (``pages/``): cada una es un script independiente.
- ``st.session_state`` guarda solo punteros; los datos se releen de
  disco con caché acotado (``estado_app``).
- La lógica vive en ``acciones_app`` y ``src/``; las pantallas solo
  dibujan y delegan.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging

import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import auth_app
import marca_sev
import estado_app as estado

VERSION_APP = "2026.10.09-a"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")

st.set_page_config(page_title="TLALLI-RES 1D — SEV/IP", page_icon="⚡", layout="wide")

sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("⚡ Sondeos Eléctricos Verticales e IP — procesamiento e inversión")
st.caption(
    "Flujo: lectura y QA/QC → promedio de direcciones → preprocesamiento → Zohdy → "
    "inversión multicapa (pyGIMLi) → reducción de Dar Zarrouk + BIC → modelo simplificado "
    "(pyGIMLi) → IP → parámetros de Dar Zarrouk y corrosividad → perfiles 2D (LCI)."
)

proyecto = estado.obtener_proyecto()
if proyecto is None:
    st.info("**Comience en la pantalla «Proyecto y datos»** (menú lateral) para crear un proyecto "
            "nuevo o abrir un archivo `.isev`.")
else:
    resumen = acc.resumen_proyecto(proyecto)
    st.success(f"Proyecto activo: **{resumen['nombre']}**")
    tarjetas = st.columns(5)
    tarjetas[0].metric("Sondeos", resumen["n_sondeos"])
    tarjetas[1].metric("Completos", resumen["n_completos"])
    tarjetas[2].metric("Con IP", resumen["n_con_ip"])
    tarjetas[3].metric("Con coordenadas", resumen["n_con_coordenadas"])
    tarjetas[4].metric("Tamaño", f"{resumen['tamano_mb']:.2f} MB")

    import almacenamiento_sev as alm

    df_estado = alm.estado_pipeline(proyecto)
    if len(df_estado):
        st.subheader("Estado del proyecto")
        tabla = (df_estado[df_estado["sondeo"] != alm.CAMPANA]
                 .pivot_table(index="sondeo", columns="etapa", values="n_filas", aggfunc="count"))
        st.dataframe(tabla.notna().replace({True: "✅", False: "—"}), use_container_width=True)
    else:
        st.caption("El proyecto aún no tiene datos procesados.")

with st.sidebar:
    st.header("Navegación")
    st.markdown(
        "1. **Proyecto y datos** — proyecto, archivo de SEV, coordenadas\n"
        "2. **Preprocesamiento** — direcciones, empalmes y QA/QC\n"
        "3. **Inversión** — Zohdy → multicapa → simplificado → IP\n"
        "4. **Parámetros y unidades** — Dar Zarrouk, corrosividad, unidades\n"
        "5. **Perfiles 2D** — rutas, secciones y LCI\n"
        "6. **Exportar datos** — CSV y modelos"
    )
    st.divider()
    st.caption(f"Versión del código: `{VERSION_APP}`")
    if st.button("🧹 Limpiar caché de datos", use_container_width=True):
        estado.limpiar_cache_datos()
        st.toast("Caché limpiado.")
    else:
        pass

auth_app.panel_administracion(sesion)
