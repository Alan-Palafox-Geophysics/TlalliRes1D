"""
Pantalla 4 — Parámetros y unidades
==================================================================
Parámetros de Dar Zarrouk (S, T, ρL, ρT, λ, ρm), corrosividad del
suelo, mapas de campaña y unidades geoeléctricas (mezcla gaussiana).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import almacenamiento_sev as alm
import auth_app
import estado_app as estado
import graficos_sev as gr
import marca_sev
import parametros_sev as pa

st.set_page_config(page_title="TLALLI-RES 1D — Parámetros", page_icon="📐", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("📐 Parámetros de Dar Zarrouk, corrosividad y unidades")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass
config = estado.obtener_config()

with st.container(border=True):
    st.subheader("Configuración")
    c = st.columns(3)
    usar_doi = c[0].checkbox("Calcular hasta la DOI de cada sondeo", value=config.prof_parametros is None,
                             key="chk_prof_doi")
    if usar_doi:
        config.prof_parametros = None
    else:
        config.prof_parametros = c[0].number_input("Profundidad de cálculo (m)", 0.5, 2000.0,
                                                   float(config.prof_parametros or 30.0), key="n_prof")
    config.prof_corrosion = c[1].number_input("Profundidad del suelo para corrosividad (m)", 0.5, 50.0,
                                              float(config.prof_corrosion), key="n_prof_corr",
                                              help="Típicamente la profundidad de la tubería o estructura enterrada.")
    texto = c[2].text_input("Umbrales de corrosividad (Ω·m)", ", ".join(f"{u:g}" for u in config.umbrales_corrosion),
                            key="txt_umbrales",
                            help=("Cuatro cortes ascendentes: Muy alta < u1 ≤ Alta < u2 ≤ Moderada < u3 ≤ Baja < u4 ≤ "
                                  "Muy baja. Por defecto 5, 10, 20, 100 Ω·m (500, 1000, 2000, 10 000 Ω·cm)."))
    try:
        umbrales = tuple(sorted(float(v) for v in texto.replace(";", ",").split(",") if v.strip()))
        if len(umbrales) == 4:
            config.umbrales_corrosion = umbrales
        else:
            st.warning("Escriba exactamente cuatro umbrales.")
    except ValueError:
        st.warning("Umbrales no numéricos: se conservan los anteriores.")
    estado.fijar_config(config)

    b = st.columns(3)
    if b[0].button("📐 Recalcular parámetros de todos", type="primary", key="btn_param"):
        resultado, _ = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, estado.sondeos_disponibles(), config,
                                                 "parametros", "parametros", cb, tk))
        estado.registrar_resultado(resultado)
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    n_max = int(b[1].number_input("Máximo de unidades", 2, 10, 6, key="n_unid"))
    if b[2].button("🧭 Calcular unidades geoeléctricas", key="btn_unid"):
        resultado = acc.accion_unidades(proyecto, n_max=n_max)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass

# ----------------------------------------------------------------------
# Campaña
# ----------------------------------------------------------------------
campana = alm.metricas_campana(proyecto, "parametros")
if len(campana):
    st.subheader("Resumen de campaña (modelo multicapa)")
    columnas = {"sondeo": "Sondeo", "profundidad": "H (m)", "S": "S (siemens)", "T": "T (Ω·m²)",
                "rho_L": "ρL (Ω·m)", "rho_T": "ρT (Ω·m)", "lambda": "λ", "rho_m": "ρm (Ω·m)",
                "rho_suelo": "ρ suelo (Ω·m)", "corrosividad": "Corrosividad",
                "diferencia_S_pct": "ΔS simp. (%)", "diferencia_T_pct": "ΔT simp. (%)"}
    tabla = campana[[c for c in columnas if c in campana.columns]].rename(columns=columnas)
    for col in tabla.columns:
        if col not in ("Sondeo", "Corrosividad"):
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(3)
        else:
            pass
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    coords = alm.coordenadas(proyecto)
    if len(coords):
        PARAMS = {"S": "S (siemens)", "T": "T (Ω·m²)", "rho_m": "ρm (Ω·m)", "rho_L": "ρL (Ω·m)",
                  "rho_T": "ρT (Ω·m)", "lambda": "λ", "rho_suelo": "ρ suelo (Ω·m)"}
        parametro = st.selectbox("Parámetro del mapa", list(PARAMS), format_func=lambda k: PARAMS[k], key="sel_mapa")
        datos = coords.merge(campana.rename(columns={"sondeo": "Sondeo"}), on="Sondeo", how="inner")
        if len(datos):
            st.plotly_chart(gr.mapa_sondeos(datos, pd.to_numeric(datos[parametro], errors="coerce"),
                                            PARAMS[parametro], usar_log=parametro != "lambda"),
                            use_container_width=True)
        else:
            pass
    else:
        st.caption("Cargue coordenadas en la pantalla 1 para ver los mapas de parámetros.")
else:
    st.info("Aún no hay parámetros calculados. Ejecute la inversión (pantalla 3) o el pipeline completo.")

# ----------------------------------------------------------------------
# Sondeo activo
# ----------------------------------------------------------------------
st.divider()
sondeo = estado.selector_sondeo(key="sel_sondeo_p4")
if sondeo:
    parametros = estado.cargar_tabla("parametros", "parametros")
    multicapa = estado.cargar_tabla("inversion", "multicapa")
    simplificado = estado.cargar_tabla("inversion", "simplificado")
    if parametros is not None:
        st.markdown(f"**{sondeo}** — los valores del multicapa y del simplificado deben coincidir: "
                    "Dar Zarrouk conserva S y T.")
        st.dataframe(estado.tabla_legible(parametros), use_container_width=True, hide_index=True)
    else:
        pass
    if simplificado is not None:
        capas = simplificado[["capa", "z_techo", "espesor", "rho", "S", "T", "m_ip"]].copy()
        capas["corrosividad"] = pa.corrosividad_por_capa(capas, config.umbrales_corrosion)
        st.markdown("**Capas del modelo simplificado**")
        st.dataframe(estado.tabla_legible(capas), use_container_width=True, hide_index=True)
    else:
        pass
else:
    pass

# ----------------------------------------------------------------------
# Unidades geoeléctricas
# ----------------------------------------------------------------------
unidades = estado.cargar_tabla("parametros", "unidades", alm.CAMPANA)
resumen_u = estado.cargar_tabla("parametros", "resumen_unidades", alm.CAMPANA)
if unidades is not None and len(unidades):
    st.divider()
    st.subheader("Unidades geoeléctricas de la campaña")
    st.caption("Mezcla gaussiana sobre (log ρ, profundidad, IP) de las capas simplificadas de todos los "
               "sondeos; el número de unidades se elige por BIC. U1 es la más conductora.")
    st.plotly_chart(gr.columnas_unidades(unidades), use_container_width=True)
    if resumen_u is not None:
        st.dataframe(resumen_u.round(3), use_container_width=True, hide_index=True)
    else:
        pass
else:
    pass

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
