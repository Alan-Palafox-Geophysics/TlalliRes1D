"""
Pantalla 2 — Preprocesamiento
==================================================================
Promedio de direcciones (con corrección de empalmes) y QA/QC de la
curva final: detección de datos anómalos y su tratamiento. Muestra la
curva original contra la preprocesada.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import auth_app
import estado_app as estado
import graficos_sev as gr
import marca_sev

st.set_page_config(page_title="TLALLI-RES 1D — Preprocesamiento", page_icon="🧹", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🧹 Preprocesamiento y QA/QC")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p2")
with c[1]:
    if sondeo:
        estado.mostrar_estado_pipeline(sondeo)
    else:
        pass
if sondeo is None:
    st.stop()
else:
    pass

config = estado.obtener_config()
pp = config.preproceso

with st.container(border=True):
    st.subheader("Configuración")
    modo = st.radio(
        "Modo", ["Auto", "Manual"], index=0 if pp.modo == "Auto" else 1, horizontal=True, key="radio_modo",
        help=("Auto: referencia física (inversión suave robusta con pyGIMLi), z robusto > 3 y "
              "error > 10 %, dispersión entre direcciones > 20 %, pendiente > 45°. "
              "Manual: usted fija curva de referencia, umbrales y tratamiento."),
    )
    pp.modo = modo
    c = st.columns(3)
    pp.corregir_empalmes = c[0].checkbox(
        "Corregir empalmes de MN", value=pp.corregir_empalmes, key="chk_empalmes",
        help="Factor multiplicativo por tramo de MN calculado en los AB/2 repetidos (Schlumberger).")
    TRATAMIENTOS = {"suavizar": "Suavizar (sustituir por la curva de referencia)",
                    "ponderar": "Ponderar (conservar y elevar su error)",
                    "eliminar": "Eliminar del ajuste"}
    pp.tratamiento = c[1].selectbox("Tratamiento de datos anómalos", list(TRATAMIENTOS),
                                    index=list(TRATAMIENTOS).index(pp.tratamiento),
                                    format_func=lambda k: TRATAMIENTOS[k], key="sel_trat")
    pp.piso_error = c[2].number_input("Error mínimo de cada dato (%)", 0.05, 20.0, pp.piso_error * 100,
                                      step=0.1, format="%.2f", key="n_piso",
                                      help=("El error de cada dato es el mayor entre este piso y el ruido "
                                            "medido en los propios datos (entre direcciones o por "
                                            "diferencias). Con datos de campo manda el ruido medido.")) / 100

    if modo == "Manual":
        METODOS = {"fisico": "Ajuste físico robusto (pyGIMLi)", "lowess": "LOWESS robusto (log-log)",
                   "mediana": "Mediana móvil", "savgol": "Savitzky-Golay"}
        c = st.columns(4)
        pp.metodo = c[0].selectbox("Curva de referencia", list(METODOS), index=list(METODOS).index(pp.metodo),
                                   format_func=lambda k: METODOS[k], key="sel_metodo")
        pp.umbral_rel = c[1].number_input("Umbral de error (%)", 1.0, 100.0, pp.umbral_rel * 100, step=1.0,
                                          key="n_umbral") / 100
        pp.umbral_z = c[2].number_input("z robusto mínimo", 1.0, 10.0, pp.umbral_z, step=0.5, key="n_z")
        pp.umbral_disp = c[3].number_input("Umbral de dispersión entre direcciones (%)", 1.0, 200.0,
                                           pp.umbral_disp * 100, step=1.0, key="n_disp") / 100
        c = st.columns(3)
        pp.ventana = int(c[0].number_input("Ventana (mediana / S-G)", 3, 15, pp.ventana, step=2, key="n_vent"))
        pp.verificar_pendiente = c[1].checkbox("Restricción de pendiente 45° (SCH/WE)", value=pp.verificar_pendiente,
                                               key="chk_pend")
        pp.ip_umbral_abs = c[2].number_input("Umbral de IP (mV/V)", 0.1, 50.0, pp.ip_umbral_abs, key="n_ipu")
    else:
        pass
    config.preproceso = pp
    estado.fijar_config(config)
    st.caption(
        "Criterio: las direcciones se promedian **geométricamente**. Con 2 direcciones no se descarta "
        "ninguna (no se puede saber cuál es la buena): su discrepancia solo aumenta el error del dato. "
        "Con 3 o más, se descarta la que contradice a la mayoría. Un dato se trata como anómalo solo si "
        "se sale de la tendencia de **su sondeo** y también de la dispersión típica de **la campaña** "
        "(mediana de los demás sondeos preprocesados).")

    b = st.columns([1, 1, 2])
    if b[0].button("🧹 Preprocesar este sondeo", type="primary", key="btn_prep"):
        resultado = acc.accion_reprocesar(proyecto, [sondeo], config, "promedio", "procesado")
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    if b[1].button("🧹 Preprocesar todos", key="btn_prep_todos"):
        resultado, _ = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, estado.sondeos_disponibles(), config,
                                                 "promedio", "procesado", cb, tk))
        estado.registrar_resultado(resultado)
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    b[2].caption("Después de preprocesar, invierta en la pantalla 3 (o use «Reprocesar» en la 1).")

procesado = estado.cargar_tabla("procesado", "procesado")
promedio = estado.cargar_tabla("promedio", "promedio")
crudo = estado.cargar_tabla("crudo", "lectura")

if promedio is not None:
    with st.expander("Direcciones y promedio geométrico", expanded=procesado is None):
        st.plotly_chart(gr.curva_direcciones(crudo, promedio), use_container_width=True)
        st.plotly_chart(gr.anisotropia(promedio), use_container_width=True)
        empalmes = estado.cargar_tabla("crudo", "empalmes")
        if empalmes is not None and "MN2" in empalmes:
            st.markdown("**Empalmes de MN**")
            st.dataframe(empalmes, use_container_width=True, hide_index=True)
        else:
            pass
else:
    pass

if procesado is None:
    st.info("Este sondeo aún no tiene preprocesamiento.")
    st.stop()
else:
    pass

metricas = acc.metricas_sondeo(proyecto, sondeo)
t = st.columns(5)
t[0].metric("Puntos", int(metricas.get("procesado.n_puntos", len(procesado))))
t[1].metric("Datos tratados", int(metricas.get("procesado.n_anomalos", 0)))
t[2].metric("IP tratados", int(metricas.get("procesado.n_ip_anomalos", 0)))
t[3].metric("Error mediano asignado", f"{100 * metricas.get('procesado.error_mediano', 0):.1f} %")
t[4].metric("Dispersión entre direcciones", f"{100 * metricas.get('promedio.dispersion_mediana', 0):.1f} %")
if str(metricas.get("procesado.requirio_preproceso", "True")) == "False":
    st.success("La curva no requirió preprocesamiento: la preprocesada es igual a la observada.")
else:
    pass

st.plotly_chart(gr.curva_preproceso(procesado), use_container_width=True)
if (procesado["ip_obs"].abs() > 0).any():
    st.plotly_chart(gr.ip_preproceso(procesado), use_container_width=True)
else:
    pass

anomalos = procesado[procesado["anomalo"] | procesado["ip_anomalo"].astype(bool)]
if len(anomalos):
    st.markdown("**Datos tratados**")
    tabla = anomalos[["AB2", "MN2", "rhoa_obs", "rhoa_ref", "rhoa_proc", "error_rel", "z", "motivo",
                      "ip_obs", "ip_proc"]].copy()
    tabla["error_rel"] = (100 * tabla["error_rel"]).round(1)
    st.dataframe(tabla.rename(columns={"error_rel": "error (%)"}), use_container_width=True, hide_index=True)
else:
    pass

with st.expander("Tabla completa de la curva preprocesada"):
    st.dataframe(procesado, use_container_width=True, height=320)

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
