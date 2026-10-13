"""
Pantalla 1 — Proyecto y datos
==================================================================
Crear/abrir proyecto, cargar el archivo de SEV (uno o varios sondeos,
varias direcciones), cargar o editar coordenadas y revisar los datos
crudos por dirección.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import almacenamiento_sev as alm
import auth_app
import estado_app as estado
import geometria_sev as geo
import graficos_sev as gr
import marca_sev
import pipeline_sev as pl

st.set_page_config(page_title="TLALLI-RES 1D — Proyecto y datos", page_icon="📁", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("📁 Proyecto y datos")
estado.mostrar_aviso()

# ----------------------------------------------------------------------
# Proyecto
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Proyecto")
    st.caption("Sus datos se procesan en un espacio temporal que **se elimina al cerrar sesión**. "
               "Descargue el archivo `.isev` para conservar el proyecto.")
    pestana_nuevo, pestana_abrir = st.tabs(["Crear nuevo", "Abrir archivo .isev"])
    with pestana_nuevo:
        c = st.columns([2, 1, 1])
        nombre_proyecto = c[0].text_input("Nombre del proyecto", key="nuevo_nombre")
        cliente = c[1].text_input("Cliente", key="nuevo_cliente")
        anio = c[2].text_input("Año", key="nuevo_anio")
        if st.button("Crear proyecto", type="primary", key="btn_crear"):
            resultado = acc.accion_crear_proyecto(estado.espacio_trabajo(), nombre_proyecto, cliente, anio)
            estado.registrar_resultado(resultado)
            if resultado:
                estado.fijar_proyecto(resultado.datos["proyecto"])
                st.rerun()
            else:
                st.error(resultado.mensaje)
        else:
            pass
    with pestana_abrir:
        archivo_proyecto = st.file_uploader("Archivo de proyecto (.isev)", type=["isev", "zip"], key="up_isev")
        if st.button("Abrir proyecto", key="btn_abrir", disabled=archivo_proyecto is None):
            resultado = acc.accion_restaurar_proyecto(archivo_proyecto.getvalue(), estado.espacio_trabajo())
            estado.registrar_resultado(resultado)
            if resultado:
                estado.fijar_proyecto(resultado.datos["proyecto"])
                st.rerun()
            else:
                st.error(resultado.mensaje)
        else:
            pass

proyecto = estado.obtener_proyecto()
if proyecto is None:
    st.info("Cree o abra un proyecto para continuar.")
    st.stop()
else:
    pass

resumen = acc.resumen_proyecto(proyecto)
t = st.columns(5)
t[0].metric("Proyecto", resumen["nombre"])
t[1].metric("Sondeos", resumen["n_sondeos"])
t[2].metric("Completos", resumen["n_completos"])
t[3].metric("Con IP", resumen["n_con_ip"])
t[4].metric("Con coordenadas", resumen["n_con_coordenadas"])

config = estado.obtener_config()
ETAPAS = list(pl.ORDEN_ETAPAS)

# ----------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Cargar sondeos")
    st.caption("Formato: `Sondeo, ID, a, AB/2, MN/2, K, I, V, R, ResApp, Direccion, IP` "
               "(IP y Direccion opcionales). Un archivo puede traer varios sondeos y varias "
               "direcciones por sondeo; el separador y la coma decimal se detectan solos.")
    c = st.columns([3, 2, 2])
    archivo_datos = c[0].file_uploader("Archivo de SEV (.txt, .csv, .dat)", type=None, key="up_datos")
    opciones_arreglo = ["AUTO"] + list(geo.ARREGLOS)
    arreglo = c[1].selectbox(
        "Arreglo", opciones_arreglo,
        format_func=lambda k: "Automático (por la columna K)" if k == "AUTO" else f"{geo.ARREGLOS[k]} ({k})",
        index=opciones_arreglo.index(config.arreglo_forzado or "AUTO"), key="sel_arreglo",
        help=("Automático: prueba todos los arreglos y convenciones de columnas y elige el que "
              "reproduce la columna K del archivo. Forzado: solo elige la convención."),
    )
    config.arreglo_forzado = None if arreglo == "AUTO" else arreglo
    config.preproceso.arreglo_forzado = config.arreglo_forzado
    hasta = c[2].selectbox("Ejecutar hasta", ETAPAS, index=len(ETAPAS) - 1,
                           format_func=lambda k: pl.ETIQUETAS_ETAPA[k], key="sel_hasta")
    estado.fijar_config(config)

    b = st.columns(3)
    if b[0].button("📥 Solo cargar datos", use_container_width=True, key="btn_cargar"):
        if archivo_datos is None:
            st.warning("Seleccione primero un archivo de datos.")
        else:
            ruta = acc.guardar_archivo_subido(proyecto, archivo_datos.name, archivo_datos.getvalue())
            resultado = acc.accion_cargar_datos(proyecto, ruta, config)
            estado.registrar_resultado(resultado)
            estado.avisar(resultado)
            if resultado:
                estado.fijar_sondeo(resultado.datos["sondeos"][0])
                estado.limpiar_cache_datos()
            else:
                pass
            st.rerun()
    else:
        pass

    if b[1].button("▶️ Correr pipeline", type="primary", use_container_width=True, key="btn_correr"):
        if archivo_datos is None:
            st.warning("Seleccione primero un archivo de datos.")
        else:
            ruta = acc.guardar_archivo_subido(proyecto, archivo_datos.name, archivo_datos.getvalue())
            resultado, token = estado.ejecutar_con_progreso(
                lambda cb, tk: acc.accion_ejecutar_archivo(proyecto, ruta, config, hasta, cb, tk),
                "Leyendo archivo…")
            estado.registrar_resultado(resultado)
            if resultado.datos.get("sondeos"):
                estado.fijar_sondeo(resultado.datos["sondeos"][0])
            else:
                pass
            estado.avisar(resultado, cancelado=token.cancelado)
            st.rerun()
    else:
        pass

    if b[2].button("⏹️ Detener ejecución", use_container_width=True, key="btn_detener",
                   help="Se detiene al terminar el paso en curso; lo completado queda guardado."):
        estado.solicitar_detener()
        st.warning("Detención solicitada.")
    else:
        pass

# ----------------------------------------------------------------------
# Reprocesar
# ----------------------------------------------------------------------
sondeos = estado.sondeos_disponibles()
if sondeos:
    with st.container(border=True):
        st.subheader("Reprocesar sondeos ya cargados")
        st.caption("Rehace etapas con la configuración actual de las pantallas 2 y 3, sin volver a subir archivos.")
        c = st.columns([3, 1, 1])
        seleccion = c[0].multiselect("Sondeos", sondeos, default=sondeos, key="ms_reproc")
        desde = c[1].selectbox("Desde", ETAPAS[1:], index=0, format_func=lambda k: pl.ETIQUETAS_ETAPA[k],
                               key="sel_desde_rep")
        hasta_rep = c[2].selectbox("Hasta", ETAPAS[1:], index=len(ETAPAS) - 2,
                                   format_func=lambda k: pl.ETIQUETAS_ETAPA[k], key="sel_hasta_rep")
        if st.button("♻️ Reprocesar", type="primary", key="btn_reproc", disabled=not seleccion):
            if ETAPAS.index(hasta_rep) < ETAPAS.index(desde):
                st.error("«Hasta» no puede ir antes que «Desde».")
            else:
                resultado, _ = estado.ejecutar_con_progreso(
                    lambda cb, tk: acc.accion_reprocesar(proyecto, seleccion, config, desde, hasta_rep, cb, tk))
                estado.registrar_resultado(resultado)
                estado.avisar(resultado)
                st.rerun()
        else:
            pass
else:
    pass

# ----------------------------------------------------------------------
# Coordenadas
# ----------------------------------------------------------------------
if sondeos:
    with st.container(border=True):
        st.subheader("Coordenadas de los sondeos")
        st.caption("Necesarias para el mapa y los perfiles 2D. Suba un archivo `Sondeo, X, Y, Z` "
                   "o escríbalas en la tabla.")
        c = st.columns([2, 3])
        archivo_coord = c[0].file_uploader("Archivo de coordenadas", type=["txt", "csv", "dat"], key="up_coord")
        if c[0].button("📍 Cargar archivo de coordenadas", disabled=archivo_coord is None, key="btn_coord"):
            resultado = acc.accion_cargar_coordenadas(proyecto, archivo_coord.getvalue())
            estado.registrar_resultado(resultado)
            estado.avisar(resultado)
            st.rerun()
        else:
            pass
        tabla_sondeos = alm.listar_sondeos(proyecto)
        editable = pd.DataFrame({"Sondeo": tabla_sondeos["sondeo"], "X": tabla_sondeos["x"],
                                 "Y": tabla_sondeos["y"], "Z": tabla_sondeos["z"]})
        editada = c[1].data_editor(editable, disabled=["Sondeo"], hide_index=True, key="ed_coord",
                                   use_container_width=True)
        if c[1].button("💾 Guardar coordenadas de la tabla", key="btn_guardar_coord"):
            resultado = acc.accion_editar_coordenadas(proyecto, editada)
            estado.registrar_resultado(resultado)
            estado.avisar(resultado)
            st.rerun()
        else:
            pass
        coords = alm.coordenadas(proyecto)
        if len(coords):
            st.plotly_chart(gr.mapa_sondeos(coords), use_container_width=True)
        else:
            pass
else:
    pass

# ----------------------------------------------------------------------
# Datos del sondeo activo
# ----------------------------------------------------------------------
st.divider()
c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p1")
with c[1]:
    if sondeo:
        estado.mostrar_estado_pipeline(sondeo)
    else:
        pass

if sondeo:
    crudo = estado.cargar_tabla("crudo", "lectura")
    if crudo is None:
        st.info("Este sondeo aún no tiene datos leídos.")
    else:
        metricas = acc.metricas_sondeo(proyecto, sondeo)
        m = st.columns(5)
        codigo = str(metricas.get("crudo.arreglo", "—"))
        m[0].metric("Arreglo", geo.ARREGLOS.get(codigo, codigo))
        m[1].metric("Convención", str(metricas.get("crudo.convencion", "—")))
        error_k = metricas.get("crudo.error_k_mediano")
        m[2].metric("Error de K", "sin K" if error_k is None or pd.isna(error_k) else f"{100 * error_k:.2f} %")
        m[3].metric("Direcciones", str(metricas.get("crudo.direcciones", "—")))
        m[4].metric("Registros válidos", f"{int(metricas.get('crudo.n_validos', 0))} / "
                                         f"{int(metricas.get('crudo.n_registros', 0))}")
        promedio = estado.cargar_tabla("promedio", "promedio")
        st.plotly_chart(gr.curva_direcciones(crudo, promedio), use_container_width=True)
        if promedio is not None:
            st.plotly_chart(gr.anisotropia(promedio), use_container_width=True)
        else:
            pass
        with st.expander("Tabla de datos crudos con geometría recalculada y banderas de calidad"):
            columnas = [c for c in ("Direccion", "ID", "a", "AB2", "MN2", "K", "K_geom", "I", "V", "R",
                                    "ResApp", "rhoa", "factor_empalme", "IP", "z_mediana", "qc_k",
                                    "qc_rhoa", "valido") if c in crudo.columns]
            st.dataframe(crudo[columnas], use_container_width=True, height=320)
else:
    pass

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
