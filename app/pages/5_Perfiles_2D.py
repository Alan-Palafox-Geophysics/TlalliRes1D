"""
Pantalla 5 — Perfiles 2D
==================================================================
Proyección de los sondeos sobre una ruta definida por vértices,
pseudosecciones (observada, preprocesada y calculada), secciones
interpoladas de los modelos 1D e inversión lateralmente restringida
(LCI) con pyGIMLi.
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
import perfil_sev as pf
import pipeline_sev as pl

st.set_page_config(page_title="TLALLI-RES 1D — Perfiles 2D", page_icon="🗺️", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🗺️ Perfiles 2D")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

coords = alm.coordenadas(proyecto)
if len(coords) < 2:
    st.info("Los perfiles necesitan coordenadas de al menos dos sondeos. Cárguelas en la pantalla 1.")
    st.stop()
else:
    pass

perfiles = pl.listar_perfiles(proyecto)
# Tras construir un perfil, la página se relanza y debe abrir ese perfil:
# el valor del selector solo puede fijarse antes de dibujarlo.
_pendiente = st.session_state.pop("perfil_recien_creado", None)
if _pendiente in perfiles:
    st.session_state["sel_perfil"] = _pendiente
else:
    pass

with st.container(border=True):
    st.subheader("Ruta del perfil")
    c = st.columns([2, 2])
    opcion = c[0].selectbox("Perfil", ["➕ Nuevo perfil"] + perfiles, key="sel_perfil")
    if opcion == "➕ Nuevo perfil":
        nombre = c[1].text_input("Nombre del perfil", value=f"P{len(perfiles) + 1}", key="txt_perfil")
        ruta_guardada = None
    else:
        nombre = opcion
        c[1].markdown(f"Editando **{nombre}**")
        ruta_guardada = alm.cargar_tabla(proyecto, "perfiles", "perfil_ruta", nombre)

    clave_vertices = f"vertices_{nombre}"
    if clave_vertices not in st.session_state:
        if ruta_guardada is not None:
            st.session_state[clave_vertices] = ruta_guardada[["X", "Y"]].values.tolist()
        else:
            st.session_state[clave_vertices] = pf.ruta_automatica(coords)
    else:
        pass

    c = st.columns([3, 2])
    with c[0]:
        st.caption("Vértices de la ruta (agregue filas para una ruta quebrada). Los sondeos se proyectan "
                   "sobre el segmento más cercano.")
        vertices = st.data_editor(pd.DataFrame(st.session_state[clave_vertices], columns=["X", "Y"]),
                                  num_rows="dynamic", use_container_width=True, key=f"ed_{nombre}")
        if st.button("↺ Ruta automática por el eje de los sondeos", key="btn_auto"):
            st.session_state[clave_vertices] = pf.ruta_automatica(coords)
            st.rerun()
        else:
            pass
    with c[1]:
        buffer = st.number_input("Distancia máxima a la ruta (m)", 1.0, 1e6, 250.0, key="n_buffer")
        hacer_lci = st.checkbox("Inversión lateralmente restringida (LCI)", value=True, key="chk_lci",
                                help="Invierte todos los sondeos del perfil juntos con continuidad lateral.")
        lam = st.number_input("λ de la LCI", 0.5, 2000.0, 20.0, key="n_lam_lci", disabled=not hacer_lci)
        acop = st.slider("Acoplamiento lateral", 0.1, 5.0, 1.0, step=0.1, key="sl_acop", disabled=not hacer_lci,
                         help="1 = mismo peso que la suavidad vertical; >1 secciones más continuas.")
        n_capas = int(st.number_input("Capas de la LCI", 10, 50, 25, key="n_capas_lci", disabled=not hacer_lci))

    vert = vertices.dropna().to_numpy(float).tolist()
    proyectados_vista = None
    if len(vert) >= 2:
        proyectados_vista = pf.proyectar(coords, vert, buffer=buffer)
    else:
        pass
    st.plotly_chart(gr.mapa_sondeos(coords, ruta=np.array(vert) if len(vert) >= 2 else None,
                                    proyectados=proyectados_vista), use_container_width=True)

    if st.button("🗺️ Construir perfil", type="primary", key="btn_perfil", disabled=len(vert) < 2):
        st.session_state[clave_vertices] = vert
        with st.spinner("Proyectando e invirtiendo…"):
            resultado = acc.accion_perfil(proyecto, nombre, vert, buffer, hacer_lci, lam, acop, n_capas)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        if resultado:
            st.session_state["perfil_recien_creado"] = alm.sanear_id(nombre)
            st.session_state[f"vertices_{alm.sanear_id(nombre)}"] = vert
        else:
            pass
        st.rerun()
    else:
        pass

# ----------------------------------------------------------------------
# Resultados del perfil
# ----------------------------------------------------------------------
nombre_id = alm.sanear_id(nombre)
proyectados = estado.cargar_tabla("perfiles", "perfil_sondeos", nombre_id)
if proyectados is None:
    st.info("Construya el perfil para ver pseudosecciones y secciones.")
    st.stop()
else:
    pass
dentro = proyectados[proyectados["dentro"].astype(bool)].copy()
dentro["dentro"] = True
nombres = dentro["Sondeo"].astype(str).tolist()

procesados = {s: estado.cargar_tabla("procesado", "procesado", s) for s in nombres}
ajustes = {s: estado.cargar_tabla("inversion", "ajuste", s) for s in nombres}
multicapas = {s: estado.cargar_tabla("inversion", "multicapa", s) for s in nombres}
simplificados = {s: estado.cargar_tabla("inversion", "simplificado", s) for s in nombres}
dois = {s: pl.metrica(proyecto, s, "inversion", "doi") for s in nombres}
dois = {k: v for k, v in dois.items() if v}
referencias = {}
for s_ in nombres:
    ref_ = estado.cargar_tabla("crudo", "modelo_referencia", s_)
    if ref_ is not None and len(ref_):
        esp_ = ref_["espesor"].to_numpy(float)
        bases_ = np.cumsum(np.where(np.isfinite(esp_), esp_, np.inf))
        referencias[s_] = pd.DataFrame({"z_techo": np.concatenate([[0.0], bases_[:-1]]), "z_base": bases_,
                                        "rho": ref_["rho"].to_numpy(float)})
    else:
        pass
lci = estado.cargar_tabla("perfiles", "perfil_lci", nombre_id)
ajuste_lci = estado.cargar_tabla("perfiles", "perfil_ajuste", nombre_id)

st.markdown(f"**{len(dentro)} sondeos en el perfil:** " + ", ".join(
    f"{r.Sondeo} (s={r.s:.0f} m, a {r.distancia:.0f} m)" for r in dentro.itertuples()))

st.subheader("Pseudosecciones")
curvas_proc = {k: v for k, v in procesados.items() if v is not None}
todas = pd.concat([v["rhoa_obs"] for v in curvas_proc.values()]) if curvas_proc else pd.Series([1.0])
rango = tuple(np.percentile(np.log10(todas), [2, 98]))
c = st.columns(2)
c[0].plotly_chart(gr.pseudoseccion(dentro, curvas_proc, "rhoa_obs", "Observada (promedio de direcciones)", rango),
                  use_container_width=True)
c[1].plotly_chart(gr.pseudoseccion(dentro, curvas_proc, "rhoa_proc", "Preprocesada (filtrada)", rango),
                  use_container_width=True)
c = st.columns(2)
curvas_mc = {k: v for k, v in ajustes.items() if v is not None}
c[0].plotly_chart(gr.pseudoseccion(dentro, curvas_mc, "rhoa_multicapa", "Calculada · modelos multicapa 1D", rango),
                  use_container_width=True)
if ajuste_lci is not None:
    curvas_lci = {str(k): g for k, g in ajuste_lci.groupby("Sondeo")}
    c[1].plotly_chart(gr.pseudoseccion(dentro, curvas_lci, "rhoa_lci",
                                       f"Calculada · LCI (χ²={pl.metrica(proyecto, nombre_id, 'perfiles', 'lci_chi2') or float('nan'):.2f})",
                                       rango), use_container_width=True)
else:
    c[1].plotly_chart(gr.pseudoseccion(dentro, curvas_mc, "rhoa_simplificado",
                                       "Calculada · modelos simplificados 1D", rango), use_container_width=True)

st.subheader("Secciones")
OPCIONES = {"lci": "LCI (inversión conjunta)", "multicapa": "Multicapa 1D interpolado",
            "simplificado": "Simplificado 1D interpolado", "ip": "IP · multicapa interpolado",
            "referencia": "Modelo de referencia (pozo o sintético)"}
disponibles = [k for k in OPCIONES if (k != "lci" or lci is not None) and (k != "referencia" or len(referencias) >= 2)]
c = st.columns([2, 1, 1])
# Con modelo de referencia (validación) se muestra por defecto junto a las
# secciones invertidas.
por_defecto = [k for k in ("referencia", "lci", "multicapa", "simplificado") if k in disponibles][:4]
seleccion = c[0].multiselect("Secciones a mostrar", disponibles, default=por_defecto,
                             format_func=lambda k: OPCIONES[k], key="ms_secciones")
z_max = c[1].number_input("Profundidad máxima (m)", 0.0, 5000.0,
                          float(round(np.median(list(dois.values())))) if dois else 0.0, key="n_zmax_sec",
                          help="Por defecto la DOI mediana de los sondeos; 0 = la mayor DOI.")
mostrar_doi = c[2].checkbox("Marcar DOI", value=True, key="chk_doi_sec")
escala = estado.escala_color_activa()

secciones = []
for clave in seleccion:
    try:
        if clave == "lci":
            modelos = pf.modelos_desde_lci(lci)
            secciones.append((clave, pf.seccion_interpolada(dentro, modelos, "rho", z_max=z_max or None, dois=dois)))
        else:
            if clave == "ip":
                modelos = {k: v for k, v in multicapas.items() if v is not None and (v["m_ip"] > 0).any()}
                secciones.append((clave, pf.seccion_interpolada(dentro, modelos, "m_ip", z_max=z_max or None,
                                                                usar_log=False, dois=dois)))
            else:
                fuente = {"multicapa": multicapas, "referencia": referencias}.get(clave, simplificados)
                modelos = {k: v for k, v in fuente.items() if v is not None}
                secciones.append((clave, pf.seccion_interpolada(dentro, modelos, "rho", z_max=z_max or None,
                                                                dois=dois)))
    except pf.PerfilError as exc:
        st.warning(f"{OPCIONES[clave]}: {exc}")

# Una sola escala de color para todas las secciones de resistividad, para
# poder compararlas entre sí (y con el modelo de referencia).
valores_rho = [sec["valores"][np.isfinite(sec["valores"])] for clave, sec in secciones if clave != "ip"]
valores_rho = np.concatenate(valores_rho) if valores_rho else np.array([])
rango_comun = tuple(np.percentile(valores_rho, [1.5, 98.5])) if len(valores_rho) else None
for clave, sec in secciones:
    if clave == "ip":
        figura = gr.seccion(sec, OPCIONES[clave], invertir=True, unidad="mV/V", mostrar_doi=mostrar_doi)
    else:
        figura = gr.seccion(sec, OPCIONES[clave], escala=escala, mostrar_doi=mostrar_doi, rango=rango_comun)
    st.plotly_chart(figura, use_container_width=True)

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
