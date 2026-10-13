"""
Pantalla 6 — Exportar datos
==================================================================
Descarga en CSV de las tablas de procesamiento y de resultados, con
selección de columnas por tabla (desplegable con una casilla por
columna, al estilo de los filtros de Power BI) y vista previa de lo
que se va a descargar.

Toda la lógica vive en ``exportacion_sev``: esta pantalla solo dibuja
widgets y delega.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import _ruta_src  # noqa: F401  (registra src en sys.path)
import auth_app
import marca_sev as marca_tre
import estado_app as estado
import acciones_app as acc
import almacenamiento_sev as alm
import exportacion_sev as expo

st.set_page_config(page_title="TLALLI-RES 1D — Exportar datos", page_icon="📤", layout="wide")

sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tre.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("📤 Exportar datos")

proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

columna_selector, columna_opciones = st.columns([1, 2])
with columna_selector:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p6")
with columna_opciones:
    if sondeo:
        estado.mostrar_estado_pipeline(sondeo)
    else:
        pass

if sondeo is None:
    st.stop()
else:
    pass

with st.container(border=True):
    st.subheader("Opciones de formato")
    columnas_formato = st.columns(3)
    etiqueta_separador = columnas_formato[0].selectbox(
        "Separador", ["Coma (,)", "Punto y coma (;)", "Tabulador"], key="sep_csv",
        help="Use punto y coma si su Excel está configurado en español.",
    )
    separador = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulador": "\t"}[etiqueta_separador]

    redondear = columnas_formato[1].checkbox("Redondear decimales", value=False, key="chk_redondeo")
    decimales = None
    if redondear:
        decimales = columnas_formato[2].number_input(
            "Decimales", min_value=0, max_value=10, value=4, key="n_decimales"
        )
    else:
        pass

    n_vista = columnas_formato[2].slider(
        "Filas en la vista previa", 5, 20, 8, key="n_vista_previa", disabled=redondear
    )

# ----------------------------------------------------------------------
# Tablas exportables, agrupadas
# ----------------------------------------------------------------------
for grupo, titulo_grupo in expo.GRUPOS_EXPORTACION.items():
    st.divider()
    st.header(titulo_grupo)

    tablas_grupo = expo.tablas_del_grupo(grupo)
    hay_alguna = False

    for descriptor in tablas_grupo:
        df_tabla = estado.cargar_tabla(descriptor.etapa, descriptor.tabla)
        if df_tabla is None or len(df_tabla) == 0:
            continue
        else:
            pass

        hay_alguna = True
        with st.container(border=True):
            st.markdown(f"### {descriptor.etiqueta}")
            st.caption(descriptor.descripcion)

            todas = expo.columnas_disponibles(df_tabla)
            clave_estado = f"cols_{sondeo}_{descriptor.clave}"
            if clave_estado not in st.session_state:
                st.session_state[clave_estado] = list(todas)
            else:
                pass

            seleccionadas = [c for c in st.session_state[clave_estado] if c in todas]

            fila_controles = st.columns([2, 1, 1, 2])
            with fila_controles[0]:
                with st.popover(
                    f"🔽 Columnas ({len(seleccionadas)} de {len(todas)})",
                    use_container_width=True,
                ):
                    st.caption("Desmarque las columnas que no quiera exportar.")
                    botones = st.columns(2)
                    if botones[0].button("Todas", key=f"todas_{descriptor.clave}"):
                        st.session_state[clave_estado] = list(todas)
                        st.rerun()
                    else:
                        pass
                    if botones[1].button("Ninguna", key=f"ninguna_{descriptor.clave}"):
                        st.session_state[clave_estado] = []
                        st.rerun()
                    else:
                        pass

                    st.divider()
                    marcadas: list[str] = []
                    for nombre_columna in todas:
                        activa = st.checkbox(
                            nombre_columna,
                            value=nombre_columna in seleccionadas,
                            key=f"chk_{descriptor.clave}_{nombre_columna}",
                        )
                        if activa:
                            marcadas.append(nombre_columna)
                        else:
                            pass
                    st.session_state[clave_estado] = marcadas
                    seleccionadas = marcadas

            if not seleccionadas:
                fila_controles[3].warning("Seleccione al menos una columna.")
                continue
            else:
                pass

            resumen = expo.resumen_exportacion(df_tabla, seleccionadas)
            fila_controles[1].metric("Filas", f"{resumen['n_filas']:,}")
            fila_controles[2].metric("Columnas", resumen["n_columnas"])

            contenido = expo.a_csv(
                df_tabla, seleccionadas, separador=separador, decimales=decimales
            )
            with fila_controles[3]:
                st.download_button(
                    f"⬇️ Descargar CSV ({resumen['tamano_kb']} KB)",
                    contenido,
                    file_name=f"{sondeo}_{descriptor.clave}.csv",
                    mime="text/csv",
                    key=f"dl_{descriptor.clave}",
                    use_container_width=True,
                )

            st.dataframe(
                expo.vista_previa(df_tabla, seleccionadas, n_filas=n_vista),
                use_container_width=True, height=min(38 * (n_vista + 1), 420),
            )

    if not hay_alguna:
        st.info(
            f"Este sondeo aún no tiene datos de {titulo_grupo.lower()}. "
            "Ejecute las etapas correspondientes del pipeline."
        )
    else:
        pass

# ----------------------------------------------------------------------
# Modelos en texto plano
# ----------------------------------------------------------------------
st.divider()
st.subheader("Modelos de capas en texto")
st.caption("Espesor, resistividad e IP por capa, para otros programas o para el informe.")
_c = st.columns(2)
for _col, (_tabla, _titulo) in zip(_c, (("multicapa", "Multicapa"), ("simplificado", "Simplificado"))):
    _modelo = estado.cargar_tabla("inversion", _tabla)
    if _modelo is not None:
        _col.download_button(f"⬇️ Modelo {_titulo.lower()} (.txt)", expo.modelo_a_texto(_modelo, sondeo, _titulo),
                             file_name=f"{sondeo}_{_tabla}.txt", mime="text/plain",
                             use_container_width=True, key=f"dl_txt_{_tabla}")
    else:
        _col.caption(f"Sin modelo {_titulo.lower()}.")

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
