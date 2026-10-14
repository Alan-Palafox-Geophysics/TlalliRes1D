"""
Pantalla 3 — Inversión
==================================================================
1. Zohdy (z = AB/n) → 2. multicapa (pyGIMLi, λ por Optuna) → 3. modelo de
bloques por Dar Zarrouk o por profundidad → 4. inversión de bloques con
Optuna → IP. Muestra la curva observada contra
las calculadas y los modelos multicapa y simplificado.
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
import json

import graficos_sev as gr
import parametros_sev as ps
import pipeline_sev as pl
import lectura_sev as lec
import marca_sev

st.set_page_config(page_title="TLALLI-RES 1D — Inversión", page_icon="⚡", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_sev.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("⚡ Inversión 1D")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p3")
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
ci = config.inversion

with st.container(border=True):
    st.subheader("Configuración del proceso de inversión")
    st.caption("1. Zohdy → 2. multicapa con pyGIMLi (λ por Optuna) → 3. modelo de bloques por Dar Zarrouk "
               "(de Zohdy y del multicapa; se usa el que mejor ajusta) o por profundidad → 4. inversión de "
               "bloques con Optuna. Los parámetros se pueden guardar en el proyecto para toda la campaña.")
    b0 = st.columns([1, 1, 1, 1])
    if b0[0].button("↺ Automático (parámetros recomendados)", key="btn_auto"):
        config.inversion = pl.ConfigInversion()
        estado.fijar_config(config)
        st.rerun()
    else:
        pass
    if b0[1].button("💾 Guardar parámetros de la campaña", key="btn_guardar_par"):
        pl.guardar_parametros(proyecto, config)
        st.success("Parámetros guardados en el proyecto: se usarán al abrirlo y viajan en el .isev.")
    else:
        pass
    b0[2].download_button("⬇️ Exportar parámetros (.json)",
                          data=json.dumps(pl.config_a_dict(config), indent=2, ensure_ascii=False, default=str),
                          file_name="parametros_tlallires1d.json", mime="application/json", key="btn_exp_par")
    archivo_par = b0[3].file_uploader("Importar parámetros (.json)", type=["json"], key="up_par",
                                      label_visibility="collapsed")
    if archivo_par is not None and st.session_state.get("par_importado") != archivo_par.name:
        try:
            config = pl.config_desde_dict(json.loads(archivo_par.getvalue().decode("utf-8")))
            estado.fijar_config(config)
            st.session_state["par_importado"] = archivo_par.name
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudieron leer los parámetros: {exc}")
    else:
        pass
    ci = config.inversion

    st.markdown("**Paso 1 · Zohdy** (modelo multicapa inicial)")
    cz = st.columns([2, 1, 1, 1])
    ZOH = {"edwards": "Alrededor de la profundidad de Edwards (n_E ± ventana)",
           "rango": "Rango fijo de n (AB/n)",
           "factor_edwards": "Factor sobre Edwards (versión previa)"}
    modo_z = {"ab_n": "rango"}.get(getattr(ci, "zohdy", "edwards"), getattr(ci, "zohdy", "edwards"))
    ci.zohdy = cz[0].selectbox("Profundidad z = AB/n", list(ZOH), index=list(ZOH).index(modo_z),
                               format_func=lambda k: ZOH[k], key="sel_zohdy",
                               help=("Cada capa de Zohdy tiene su base en AB/n. Se prueba cada n con el paso "
                                     "indicado y se queda el que mejor ajusta la curva, con las resistividades "
                                     "ya corregidas."))
    if ci.zohdy == "edwards":
        ci.zohdy_ventana = float(cz[1].number_input("Ventana ± en n", 0.25, 3.0, float(ci.zohdy_ventana), 0.25,
                                                    key="n_zven"))
        cz[2].caption("n_E = AB / z_e (≈ 5.3 Schlumberger, ≈ 5.8 Wenner)")
    else:
        if ci.zohdy == "rango":
            ci.zohdy_n_min = float(cz[1].number_input("n mínimo", 2.0, 12.0, float(ci.zohdy_n_min), 0.25,
                                                      key="n_zmin"))
            ci.zohdy_n_max = float(cz[2].number_input("n máximo", 2.5, 15.0, float(ci.zohdy_n_max), 0.25,
                                                      key="n_zmax"))
        else:
            pass
    if ci.zohdy != "factor_edwards":
        pasos = [0.0625, 0.125, 0.25, 0.5]
        ci.zohdy_paso = float(cz[3].selectbox("Paso de n", pasos,
                                              index=pasos.index(ci.zohdy_paso) if ci.zohdy_paso in pasos else 1,
                                              key="sel_zpaso"))
    else:
        pass

    st.markdown("**Paso 2 · Multicapa con pyGIMLi**")
    c2 = st.columns([2, 2, 1])
    REGS = {"suave": "Suave (Occam, L2) — estándar", "robusta": "Robusta (datos en L1)",
            "nitida": "Nítida (gradiente mínimo, bloques)"}
    ci.regularizacion = c2[0].selectbox("Regularización", list(REGS), index=list(REGS).index(ci.regularizacion),
                                        format_func=lambda k: REGS[k], key="sel_reg",
                                        help="Robusta: reduce el peso de las lecturas que no ajustan (norma L1).")
    METODOS = {"optuna": "Optuna (χ² → 1, rugosidad, residuales)", "discrepancia": "Principio de discrepancia",
               "manual": "Manual"}
    ci.metodo_lambda = c2[1].selectbox("Elección de λ", list(METODOS), index=list(METODOS).index(ci.metodo_lambda),
                                       format_func=lambda k: METODOS[k], key="sel_lam")
    if ci.metodo_lambda == "optuna":
        ci.n_trials = int(c2[2].number_input("Pruebas de Optuna", 5, 60, ci.n_trials, key="n_trials"))
    else:
        if ci.metodo_lambda == "manual":
            ci.lam = c2[2].number_input("λ", 0.1, 5000.0, ci.lam, key="n_lam")
        else:
            c2[2].caption("Rejilla de λ entre 0.1 y 1000.")
    MALLAS = {"zohdy": "Interfaces de Zohdy hasta la profundidad viable",
              "logaritmica": "Logarítmica regular hasta 2.5·máx z_e"}
    ci.malla = st.selectbox("Malla del multicapa", list(MALLAS), index=list(MALLAS).index(ci.malla),
                            format_func=lambda k: MALLAS[k], key="sel_malla")

    st.markdown("**Pasos 3 y 4 · Modelo de bloques e inversión optimizada con Optuna**")
    c3 = st.columns([2, 1, 1, 1])
    MODOS = {"dar_zarrouk": "Dar Zarrouk de Zohdy y del multicapa (el que mejor ajusta)",
             "profundidad": "Por profundidad: Edwards dividida log-uniforme en k capas"}
    ci.modo_bloques = c3[0].selectbox("Modelo de bloques inicial", list(MODOS),
                                      index=list(MODOS).index(ci.modo_bloques), format_func=lambda k: MODOS[k],
                                      key="sel_modo_bloques")
    if ci.modo_bloques == "profundidad":
        ci.k_profundidad = int(c3[1].number_input("Número de capas", 2, 12, int(ci.k_profundidad), key="n_kprof"))
    else:
        ci.k_max = int(c3[1].number_input("Máximo de capas", 2, 12, ci.k_max, key="n_kmax"))
        ci.k_fijo = int(c3[2].number_input("Capas fijas (0 = BIC)", 0, 12, int(ci.k_fijo), key="n_kfijo"))
    ci.n_trials_bloques = int(c3[3].number_input("Pruebas de Optuna (bloques)", 10, 200,
                                                 int(ci.n_trials_bloques), 10, key="n_tbloq"))
    c4 = st.columns([1, 2])
    ci.criterio = c4[0].selectbox("Criterio", ["BIC", "AIC"], index=0 if ci.criterio == "BIC" else 1, key="sel_crit",
                                  help="BIC penaliza más cada capa (modelos más simples); AIC es más permisivo.")
    ci.espesor_min_fraccion = float(c4[1].slider(
        "Espesor mínimo de capa (fracción de la profundidad de su techo)", 0.0, 1.0,
        float(ci.espesor_min_fraccion), 0.05, key="sl_hmin",
        help=("La primera capa, además, no puede ser más delgada que la profundidad de investigación de la "
              "apertura más corta. 0 = sin restricción (no recomendado).")))
    if ci.modo_bloques == "dar_zarrouk" and not ci.k_fijo:
        ci.k_desde_estructura = st.checkbox(
            "Número de capas según la estructura de Zohdy/multicapa (Dar Zarrouk)",
            value=bool(ci.k_desde_estructura), key="chk_kdz",
            help=("Se conservan las capas que la reducción de Dar Zarrouk de la fuente elegida muestra como "
                  "distintas (contraste ≥ 1.5) salvo que los datos den evidencia fuerte en contra (ΔBIC > 6). "
                  "Desactivado: k solo por el criterio de información (ΔIC < 2 → el más simple)."))
    else:
        pass

    st.markdown("**Otros**")
    c = st.columns(3)
    ci.calcular_doi = c[0].checkbox("Calcular DOI (Oldenburg-Li)", value=ci.calcular_doi, key="chk_doi")
    ci.invertir_ip = c[1].checkbox("Invertir IP si existe", value=ci.invertir_ip, key="chk_ip")
    ci.motor = c[2].selectbox("Motor directo", ["numpy", "pygimli"], index=0 if ci.motor == "numpy" else 1,
                              format_func=lambda k: {"numpy": "Vectorizado + jacobiano analítico (rápido)",
                                                     "pygimli": "Núcleo C++ de pyGIMLi"}[k], key="sel_motor")
    config.inversion = ci
    estado.fijar_config(config)

    b = st.columns([1, 1, 2])
    if b[0].button("⚡ Invertir este sondeo", type="primary", key="btn_inv"):
        with st.spinner("Invirtiendo…"):
            resultado = acc.accion_reprocesar(proyecto, [sondeo], config, "inversion", "parametros")
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    if b[1].button("⚡ Invertir todos", key="btn_inv_todos"):
        resultado, _ = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, estado.sondeos_disponibles(), config,
                                                 "inversion", "parametros", cb, tk))
        estado.registrar_resultado(resultado)
        estado.avisar(resultado)
        st.rerun()
    else:
        pass

    with st.expander("Modelo de referencia (pozo o sintético) para comparar"):
        archivo_ref = st.file_uploader("Archivo con columnas espesor, rho[, m_ip] o z_techo, rho[, m_ip]",
                                       type=["txt", "csv", "dat"], key="up_ref")
        cr = st.columns(2)
        if cr[0].button("Guardar referencia de este sondeo", disabled=archivo_ref is None, key="btn_ref"):
            try:
                ref = lec.leer_modelo_referencia(archivo_ref.getvalue())
                alm.guardar_tabla(proyecto, ref, "crudo", "modelo_referencia", sondeo)
                estado.limpiar_cache_datos()
                st.success(f"Modelo de referencia de {len(ref)} capas guardado.")
            except Exception as exc:
                st.error(str(exc))
        else:
            pass
        if cr[1].button("Quitar referencia", key="btn_quitar_ref"):
            alm.eliminar_tabla(proyecto, "crudo", "modelo_referencia", sondeo)
            estado.limpiar_cache_datos()
        else:
            pass

ajuste = estado.cargar_tabla("inversion", "ajuste")
multicapa = estado.cargar_tabla("inversion", "multicapa")
simplificado = estado.cargar_tabla("inversion", "simplificado")
if ajuste is None:
    st.info("Este sondeo aún no tiene inversión.")
    st.stop()
else:
    pass
referencia = estado.cargar_tabla("crudo", "modelo_referencia")
metricas = acc.metricas_sondeo(proyecto, sondeo)

t = st.columns(6)
t[0].metric("Zohdy RMS", f"{metricas.get('inversion.zohdy_rms', np.nan):.1f} %",
            help=(f"{int(metricas.get('inversion.zohdy_n_capas', 0) or 0)} capas (una por dato); profundidad "
                  f"AB/{metricas.get('inversion.zohdy_n_ab', np.nan):.3g} (factor "
                  f"{metricas.get('inversion.zohdy_factor', np.nan):.2f} sobre Edwards). Es el modelo inicial de "
                  "pyGIMLi, remuestreado en la malla del multicapa."))
t[1].metric("Multicapa χ²", f"{metricas.get('inversion.multicapa_chi2', np.nan):.2f}",
            help=f"RMS {metricas.get('inversion.multicapa_rms', np.nan):.1f} %")
t[2].metric("λ", f"{metricas.get('inversion.multicapa_lambda', np.nan):.3g}")
t[3].metric("Bloques (invertido)", f"{int(metricas.get('inversion.simplificado_n_capas', 0))} capas",
            help=f"Arranque: {metricas.get('inversion.simplificado_arranque', '')}")
t[4].metric("Bloques χ²", f"{metricas.get('inversion.simplificado_chi2', np.nan):.2f}",
            help=f"RMS {metricas.get('inversion.simplificado_rms', np.nan):.1f} %")
doi = metricas.get("inversion.doi")
_zv = metricas.get("inversion.profundidad_viable")
t[5].metric("DOI", f"{doi:.1f} m" if doi else "—",
            help=(f"Profundidad viable (Zohdy, f·máx z_e): {_zv:.1f} m. " if _zv else "")
                 + "DOI de Oldenburg-Li, acotada por la profundidad viable.")
proceso_inv = estado.cargar_tabla("inversion", "proceso")
with st.container(border=True):
    st.markdown("**Proceso de inversión aplicado a este sondeo**")
    if proceso_inv is not None and len(proceso_inv):
        for linea in proceso_inv["descripcion"]:
            st.markdown(f"- {linea}")
    else:
        st.caption("Este sondeo se invirtió con una versión anterior: vuelva a invertirlo para ver el detalle.")

chi2 = metricas.get("inversion.multicapa_chi2", 1.0)
if chi2 is not None and chi2 > 2.0:
    st.warning(f"χ² del multicapa = {chi2:.2f}: el modelo no alcanza el nivel de ruido. El error asignado puede "
               "estar subestimado o hay efectos no 1D.")
else:
    if chi2 is not None and chi2 < 0.3:
        st.info(f"χ² del multicapa = {chi2:.2f}: el ajuste está por debajo del error asignado. Es normal con datos "
                "casi sin ruido (sintéticos), donde el error es el piso de 0.1 %; con datos de campo indica un "
                "error sobrestimado.")
    else:
        pass

# Profundidad por defecto del gráfico de modelos: la parte interpretable,
# no todo hasta la DOI (que puede ser varias veces más profunda que la
# última interfaz y aplastaría el detalle somero).
if referencia is not None and np.isfinite(referencia["z_base"]).any():
    _z_def = 1.25 * float(referencia["z_base"].replace(np.inf, np.nan).max())
else:
    _bases = simplificado["z_base"].replace(np.inf, np.nan).dropna()
    _z_def = max(3.0 * float(_bases.max()) if len(_bases) else 20.0, 20.0)
    _z_def = min(_z_def, doi * 1.1) if doi else _z_def
c = st.columns([1, 1, 2])
mostrar_zohdy = c[0].checkbox("Mostrar Zohdy (modelo y curva calculada)", value=True, key="chk_zohdy",
                              help="Modelo de Zohdy con sus propias capas y la resistividad aparente que genera.")
zohdy = estado.cargar_tabla("inversion", "zohdy")
z_graf = c[1].number_input("Profundidad del gráfico (m)", 1.0, 1e5, float(round(_z_def, 0)), key="n_zgraf")
c = st.columns(2)
c[0].plotly_chart(gr.ajuste_inversion(ajuste, mostrar_zohdy=mostrar_zohdy), use_container_width=True)
c[1].plotly_chart(gr.modelos_resistividad(multicapa, simplificado, doi=doi, real=referencia,
                                          mostrar_zohdy=mostrar_zohdy, z_max=z_graf, zohdy=zohdy),
                  use_container_width=True)

if (ajuste["ip_proc"].abs() > 0).any():
    c = st.columns(2)
    c[0].plotly_chart(gr.ajuste_ip(ajuste), use_container_width=True)
    c[1].plotly_chart(gr.modelos_ip(multicapa, simplificado, doi=doi, real=referencia, z_max=z_graf),
                      use_container_width=True)
else:
    pass

st.markdown("**Modelo invertido (bloques)** — rangos ±1σ de la covarianza linealizada; "
            "«resolución» indica qué parámetro determinan realmente los datos.")
tabla = simplificado[["capa", "z_techo", "espesor", "espesor_min", "espesor_max", "rho", "rho_min", "rho_max",
                      "S", "T", "m_ip", "resolucion", "bajo_doi"]].copy()
st.dataframe(estado.tabla_legible(tabla), use_container_width=True, hide_index=True)

# Parámetros acumulados de Dar Zarrouk (se calculan también para proyectos
# guardados antes de que existieran estas columnas).
simp_acum = ps.acumulados(simplificado)
mc_acum = ps.acumulados(multicapa)
ref_acum = None
if referencia is not None and len(referencia):
    ref_acum = referencia.copy()
    if "z_techo" not in ref_acum:
        _esp = ref_acum["espesor"].to_numpy(float)
        _bases = np.cumsum(np.where(np.isfinite(_esp), _esp, np.inf))
        ref_acum["z_techo"] = np.concatenate([[0.0], _bases[:-1]])
        ref_acum["z_base"] = _bases
    else:
        pass
else:
    pass
st.subheader("Interpretación geoeléctrica")
st.plotly_chart(gr.parametros_acumulados(mc_acum, simp_acum, doi=doi, real=ref_acum, z_max=z_graf, zohdy=zohdy),
                use_container_width=True)
c = st.columns([1, 1])
with c[0]:
    st.markdown("**Modelo invertido (bloques): S, T y resistividad promedio** (de la superficie a la base de cada capa)")
    st.caption("ρL = H/S (promedio longitudinal), ρT = T/H (transversal), ρ promedio = √(ρL·ρT). "
               "«Contraste» = ρ de la capa / ρ promedio de lo que tiene encima; ≥ 2 aumento fuerte, ≤ 0.5 "
               "disminución fuerte.")
    tabla_acum = simp_acum[["capa", "z_techo", "z_base", "rho", "S", "T", "S_acum", "T_acum", "rho_L_prom",
                            "rho_T_prom", "rho_prom", "contraste_vs_arriba", "cambio"]].copy()
    st.dataframe(estado.tabla_legible(tabla_acum), use_container_width=True, hide_index=True)
with c[1]:
    equivalentes = ps.agrupar_equivalentes(multicapa)
    st.markdown(f"**Multicapa agrupado en capas equivalentes** — {len(multicapa)} celdas → "
                f"{len(equivalentes)} capas")
    st.caption(f"Celdas contiguas se unen mientras la anisotropía del grupo λ = √(T·S)/H no pase de "
               f"{ps.LAMBDA_EQUIVALENTE}: eléctricamente son una sola capa de ρ = √(T/S). Compárelas con el "
               "simplificado.")
    st.dataframe(estado.tabla_legible(equivalentes), use_container_width=True, hide_index=True)

c = st.columns(2)
seleccion = estado.cargar_tabla("inversion", "seleccion")
lambdas = estado.cargar_tabla("inversion", "lambdas")
c[0].plotly_chart(gr.seleccion_capas(seleccion, str(metricas.get("inversion.criterio", "BIC"))),
                  use_container_width=True)
c[1].plotly_chart(gr.busqueda_lambda(lambdas), use_container_width=True)

st.subheader("Corrosividad del suelo")
cc = st.columns([1, 1, 2])
prof_corr = float(cc[0].number_input("Profundidad evaluada (m)", 0.5, 50.0,
                                     float(getattr(config, "prof_corrosion", 3.0)), 0.5, key="n_prof_corr"))
z_corr = float(cc[1].number_input("Profundidad del gráfico (m)", 1.0, 500.0, max(2.0 * prof_corr, 10.0), 1.0,
                                  key="n_z_corr"))
cc[2].caption("Clases por resistividad (escala de protección catódica NACE/Roberge): < 5 Ω·m muy alta, 5–10 alta, "
              "10–20 moderada, 20–100 baja, > 100 muy baja. Es un indicador: la corrosividad real depende también "
              "de pH, cloruros, sulfatos, potencial redox y humedad.")
st.plotly_chart(gr.corrosividad(simplificado, multicapa, z_max=z_corr, umbrales=config.umbrales_corrosion,
                                prof_corrosion=prof_corr), use_container_width=True)

with st.expander("Modelo multicapa completo (con S, T y resistividad promedio acumuladas)"):
    st.dataframe(estado.tabla_legible(mc_acum), use_container_width=True, height=320)
with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
