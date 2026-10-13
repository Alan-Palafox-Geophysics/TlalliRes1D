"""
Pantalla 3 — Inversión
==================================================================
Zohdy → multicapa (pyGIMLi, λ por Optuna) → reducción de Dar Zarrouk
+ BIC → simplificado (pyGIMLi) → IP. Muestra la curva observada contra
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
import graficos_sev as gr
import parametros_sev as ps
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
    st.subheader("Configuración")
    METODOS = {"optuna": "Optuna (búsqueda bayesiana, χ² → 1)", "discrepancia": "Principio de discrepancia",
               "manual": "Manual"}
    c = st.columns(4)
    ci.metodo_lambda = c[0].selectbox("Regularización λ (multicapa)", list(METODOS),
                                      index=list(METODOS).index(ci.metodo_lambda),
                                      format_func=lambda k: METODOS[k], key="sel_lam",
                                      help=("Elige λ para que el ajuste quede al nivel del ruido (χ² ≈ 1): "
                                            "el modelo más suave que explica los datos."))
    if ci.metodo_lambda == "optuna":
        ci.n_trials = int(c[1].number_input("Pruebas de Optuna", 5, 60, ci.n_trials, key="n_trials"))
    else:
        if ci.metodo_lambda == "manual":
            ci.lam = c[1].number_input("λ", 0.1, 5000.0, ci.lam, key="n_lam")
        else:
            c[1].caption("Rejilla de 13 valores de λ entre 0.1 y 1000.")
    ci.k_max = int(c[2].number_input("Máximo de capas (simplificado)", 2, 12, ci.k_max, key="n_kmax"))
    ci.criterio = c[3].selectbox("Criterio de selección", ["BIC", "AIC"], index=0 if ci.criterio == "BIC" else 1,
                                 key="sel_crit",
                                 help="BIC penaliza más cada capa (modelos más simples); AIC es más permisivo.")
    REGS = {"nitida": "Nítida (gradiente mínimo): capas con interfaces definidas",
            "suave": "Suave (Occam L2): transiciones graduales"}
    c_reg = st.columns([3, 1, 1])
    ci.k_fijo = int(c_reg[2].number_input(
        "Capas fijas (0 = automático)", 0, 12, int(getattr(ci, "k_fijo", 0)), key="n_kfijo",
        help=("Fuerza el número de capas del simplificado en lugar de elegirlo por BIC/AIC. Útil para "
              "revisar candidatos equivalentes en la tabla de selección.")))
    ci.n_arranques = int(c_reg[1].number_input(
        "Arranques globales por k", 0, 80, int(getattr(ci, "n_arranques", 24)), step=4, key="n_arr",
        help=("Posiciones de interfaces que se exploran (hipercubo latino) para cada número de capas del "
              "simplificado. Más arranques = menos riesgo de un mínimo local, más tiempo. 0 = solo Dar Zarrouk.")))
    ci.regularizacion = c_reg[0].selectbox("Regularización del multicapa", list(REGS),
                                     index=list(REGS).index(ci.regularizacion), format_func=lambda k: REGS[k],
                                     key="sel_reg",
                                     help=("Nítida: el modelo multicapa se organiza en bloques y la reducción de "
                                           "Dar Zarrouk encuentra las interfaces reales. Suave: el clásico "
                                           "Occam, útil cuando se esperan cambios graduales."))
    MALLAS = {"zohdy": "Interfaces de Zohdy hasta la profundidad viable (f·máx z_e)",
              "logaritmica": "Logarítmica regular hasta 2.5·máx z_e"}
    ci.malla = st.selectbox("Malla del multicapa", list(MALLAS), index=list(MALLAS).index(getattr(ci, "malla", "zohdy")),
                            format_func=lambda k: MALLAS[k], key="sel_malla",
                            help=("Zohdy: el factor de profundidad que ajusta la curva calibra la escala de "
                                  "profundidades; la malla subdivide sus interfaces y termina en la profundidad "
                                  "máxima que los datos pueden ver. El modelo de Zohdy entra exacto como inicial."))
    ci.espesor_min_fraccion = float(st.slider(
        "Espesor mínimo de capa del simplificado (fracción de la profundidad de su techo)", 0.0, 1.0,
        float(getattr(ci, "espesor_min_fraccion", 0.25)), 0.05, key="sl_hmin",
        help=("Resolución vertical del SEV: una capa más delgada que esta fracción de su profundidad no se "
              "distingue de sus vecinas. La primera capa, además, no puede ser más delgada que la profundidad "
              "de investigación de la apertura más corta. 0 = sin restricción (no recomendado).")))
    cz = st.columns([2, 1, 1, 1])
    ZOH = {"ab_n": "Zohdy con z = AB/n (búsqueda de n con resistividades corregidas)",
           "factor_edwards": "Zohdy con factor sobre la profundidad de Edwards (versión previa)"}
    ci.zohdy = cz[0].selectbox("Profundidad de atribución de Zohdy", list(ZOH),
                               index=list(ZOH).index(getattr(ci, "zohdy", "ab_n")), format_func=lambda k: ZOH[k],
                               key="sel_zohdy")
    if ci.zohdy == "ab_n":
        ci.zohdy_n_min = float(cz[1].number_input("n mínimo (AB/n)", 2.0, 12.0, float(getattr(ci, "zohdy_n_min", 5.0)),
                                                  step=0.25, key="n_zmin"))
        ci.zohdy_n_max = float(cz[2].number_input("n máximo (AB/n)", 2.5, 15.0, float(getattr(ci, "zohdy_n_max", 8.0)),
                                                  step=0.25, key="n_zmax"))
        ci.zohdy_paso = float(cz[3].selectbox("Paso de n", [0.125, 0.25, 0.5],
                                              index=[0.125, 0.25, 0.5].index(getattr(ci, "zohdy_paso", 0.125)),
                                              key="sel_zpaso"))
    else:
        pass
    c = st.columns(4)
    ci.calcular_doi = c[0].checkbox("Calcular DOI (Oldenburg-Li)", value=ci.calcular_doi, key="chk_doi")
    ci.invertir_ip = c[1].checkbox("Invertir IP si existe", value=ci.invertir_ip, key="chk_ip")
    ci.robusto = c[2].checkbox("Ajuste robusto (L1)", value=ci.robusto, key="chk_rob",
                               help="Reponderación de datos que reduce el peso de los que no ajustan.")
    ci.motor = c[3].selectbox("Motor directo", ["numpy", "pygimli"], index=0 if ci.motor == "numpy" else 1,
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
t[3].metric("Simplificado", f"{int(metricas.get('inversion.simplificado_n_capas', 0))} capas",
            help=f"Arranque: {metricas.get('inversion.simplificado_arranque', '')}")
t[4].metric("Simplificado χ²", f"{metricas.get('inversion.simplificado_chi2', np.nan):.2f}",
            help=f"RMS {metricas.get('inversion.simplificado_rms', np.nan):.1f} %")
doi = metricas.get("inversion.doi")
_zv = metricas.get("inversion.profundidad_viable")
t[5].metric("DOI", f"{doi:.1f} m" if doi else "—",
            help=(f"Profundidad viable (Zohdy, f·máx z_e): {_zv:.1f} m. " if _zv else "")
                 + "DOI de Oldenburg-Li, acotada por la profundidad viable.")
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

st.markdown("**Modelo simplificado** — rangos ±1σ de la covarianza linealizada; "
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
st.plotly_chart(gr.parametros_acumulados(mc_acum, simp_acum, doi=doi, real=ref_acum, z_max=z_graf),
                use_container_width=True)
c = st.columns([1, 1])
with c[0]:
    st.markdown("**Simplificado: S, T y resistividad promedio** (de la superficie a la base de cada capa)")
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

with st.expander("Modelo multicapa completo (con S, T y resistividad promedio acumuladas)"):
    st.dataframe(estado.tabla_legible(mc_acum), use_container_width=True, height=320)
with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
