"""
graficos_sev.py
==================================================================
Figuras Plotly de la aplicación SEV/IP.

Convenciones visuales
------------------------
- Curvas de resistividad aparente en **log-log** (ρa contra AB/2), como
  se leen tradicionalmente los SEV.
- Modelos como **escalones** de resistividad contra profundidad, con el
  eje de profundidad hacia abajo y la DOI marcada.
- Secciones con la escala oficial del producto
  (``generar_escala_tre``): resistividad tal cual, IP invertida.
- Colores fijos por concepto en todas las figuras: así "simplificado"
  es siempre el mismo azul y el ojo no tiene que releer la leyenda.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generar_escala_tre import generar_escala_tre

PLANTILLA = "plotly_white"

#: Colores por concepto.
COLOR = {
    "observado": "#8A8F98",
    "preprocesado": "#00838F",
    "referencia": "#5E6A75",
    "anomalo": "#D62828",
    "multicapa": "#F28C28",      # naranja, grosor normal
    "simplificado": "#0000FF",   # modelo invertido: azul (0, 0, 255), fuerte y claro
    "zohdy": "#E6B800",          # amarillo, delgado
    "real": "#111111",
    "doi": "#7B2CBF",
    "ip": "#8A5FBF",
}
PALETA_DIRECCIONES = ["#00838F", "#E07A1F", "#3A7D44", "#9C2C77", "#4C6EF5", "#7A5C3A"]


def _escala_plotly(invertir: bool = False, n: int = 32) -> list:
    cmap = generar_escala_tre()
    posiciones = np.linspace(0, 1, n)
    muestras = posiciones[::-1] if invertir else posiciones
    salida = []
    for p, m in zip(posiciones, muestras):
        r, g, b, _ = cmap(float(m))
        salida.append([float(p), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return salida


ESCALA_RESISTIVIDAD = _escala_plotly(False)
ESCALA_IP = _escala_plotly(True)


def _vacia(mensaje: str, altura: int = 300) -> go.Figure:
    figura = go.Figure()
    figura.add_annotation(text=mensaje, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color="#888"))
    figura.update_layout(template=PLANTILLA, height=altura, xaxis=dict(visible=False),
                         yaxis=dict(visible=False))
    return figura


def marcas_log(valores) -> dict:
    """Marcas 1-2-5 para un eje logarítmico, rotuladas en valor real.

    Plotly, en ejes log que abarcan pocas décadas, rotula las marcas
    menores con la mantisa sola ("2, 3, 5") y se lee mal; aquí se
    eligen marcas redondas de la serie 1-2-5 y se rotulan completas.
    """
    v = np.asarray(valores, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return {}
    else:
        pass
    bajo, alto = np.log10(v.min()), np.log10(v.max())
    marcas = []
    for decada in range(int(np.floor(bajo)) - 1, int(np.ceil(alto)) + 2):
        for mantisa in (1, 2, 5):
            valor = mantisa * 10.0 ** decada
            if 10 ** (bajo - 0.15) <= valor <= 10 ** (alto + 0.15):
                marcas.append(valor)
            else:
                pass
    if len(marcas) > 12:
        marcas = [m for m in marcas if str(m).lstrip("0.").startswith("1")] or marcas
    else:
        pass
    texto = [f"{m:g}" for m in marcas]
    return {"tickmode": "array", "tickvals": marcas, "ticktext": texto}


def _ejes_loglog(figura: go.Figure, fila: Optional[int] = None, col: Optional[int] = None,
                 titulo_y: str = "ρa (Ω·m)", titulo_x: str = "AB/2 (m)",
                 valores_x=None, valores_y=None) -> None:
    kwargs = {} if fila is None else {"row": fila, "col": col}
    figura.update_xaxes(type="log", title=titulo_x, showgrid=True, gridcolor="#E6E6E6",
                        **(marcas_log(valores_x) if valores_x is not None else {}), **kwargs)
    figura.update_yaxes(type="log", title=titulo_y, showgrid=True, gridcolor="#E6E6E6",
                        **(marcas_log(valores_y) if valores_y is not None else {}), **kwargs)


def _por_tramo(df: pd.DataFrame, columna_y: str) -> tuple[list, list]:
    """Coordenadas de una curva calculada dibujada **por tramo de MN**.

    Con AB/2 repetidos (cambio de MN) la respuesta tiene dos valores en
    la misma abscisa; unirlos con una sola línea dibuja un zigzag que no
    existe. Se separan los tramos con ``None``.
    """
    ordenado = df.sort_values(["AB2", "MN2"])
    if not ordenado["AB2"].round(6).duplicated().any():
        # Sin aperturas repetidas (Wenner, dipolares o Schlumberger sin
        # traslapes): una sola curva.
        return list(ordenado["AB2"]), list(ordenado[columna_y])
    else:
        pass
    xs, ys = [], []
    for _, tramo in df.sort_values(["MN2", "AB2"]).groupby("MN2", sort=True):
        xs += list(tramo["AB2"]) + [None]
        ys += list(tramo[columna_y]) + [None]
    return xs, ys


def _barras_error(valores: np.ndarray, error_rel: np.ndarray) -> dict:
    """Barras de error multiplicativas (simétricas en log)."""
    v = np.asarray(valores, float)
    e = np.asarray(error_rel, float)
    return dict(type="data", symmetric=False, array=v * (np.exp(e) - 1),
                arrayminus=v * (1 - np.exp(-e)), thickness=1, width=2, color="#B0B6BD")


# ======================================================================
# Datos crudos y promedio
# ======================================================================


def curva_direcciones(crudo: pd.DataFrame, promedio: Optional[pd.DataFrame] = None,
                      titulo: str = "Curvas por dirección y promedio") -> go.Figure:
    """ρa de cada dirección (con su tramo de MN) y la curva promedio."""
    if crudo is None or len(crudo) == 0:
        return _vacia("Sin datos crudos")
    else:
        pass
    figura = go.Figure()
    columna = "rhoa_corr" if "rhoa_corr" in crudo.columns else "rhoa"
    for i, (direccion, grupo) in enumerate(crudo[crudo["valido"]].groupby("Direccion")):
        color = PALETA_DIRECCIONES[i % len(PALETA_DIRECCIONES)]
        for j, (mn2, tramo) in enumerate(grupo.groupby("MN2")):
            tramo = tramo.sort_values("AB2")
            figura.add_trace(go.Scatter(
                x=tramo["AB2"], y=tramo[columna], mode="lines+markers",
                name=f"{direccion} · MN/2={mn2:g}", legendgroup=direccion,
                line=dict(color=color, width=1.2, dash="solid" if j % 2 == 0 else "dot"),
                marker=dict(size=6, symbol=["circle", "square", "diamond", "triangle-up"][j % 4]),
                customdata=np.column_stack([tramo["MN2"], tramo["K_geom"], tramo["rhoa"]]),
                hovertemplate=("AB/2=%{x:.3g} m<br>ρa=%{y:.4g} Ω·m<br>MN/2=%{customdata[0]:.3g}"
                               "<br>K=%{customdata[1]:.4g}<br>ρa sin empalme=%{customdata[2]:.4g}<extra></extra>"),
            ))
    invalidos = crudo[~crudo["valido"]]
    if len(invalidos):
        figura.add_trace(go.Scatter(
            x=invalidos["AB2"], y=np.abs(invalidos["rhoa"]).replace(0, np.nan), mode="markers",
            name="Descartados", marker=dict(color=COLOR["anomalo"], symbol="x", size=9)))
    else:
        pass
    if promedio is not None and len(promedio):
        p = promedio.sort_values(["AB2", "MN2"])
        figura.add_trace(go.Scatter(
            x=p["AB2"], y=p["rhoa_obs"], mode="lines+markers", name="Promedio geométrico",
            line=dict(color=COLOR["real"], width=2.5), marker=dict(size=7),
            error_y=_barras_error(p["rhoa_obs"], np.nan_to_num(p["sigma_dir"], nan=0.0)),
            customdata=np.column_stack([p["n_dir"], np.nan_to_num(p["anisotropia"], nan=1.0)]),
            hovertemplate=("AB/2=%{x:.3g} m<br>ρa=%{y:.4g} Ω·m<br>direcciones=%{customdata[0]}"
                           "<br>ρmáx/ρmín=%{customdata[1]:.2f}<extra></extra>"),
        ))
    else:
        pass
    _ejes_loglog(figura, valores_x=crudo["AB2"], valores_y=crudo.loc[crudo["valido"], columna])
    figura.update_layout(template=PLANTILLA, height=480, title=titulo,
                         legend=dict(orientation="h", y=-0.18), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def anisotropia(promedio: pd.DataFrame) -> go.Figure:
    """Cociente ρmáx/ρmín entre direcciones por apertura."""
    if promedio is None or "anisotropia" not in promedio or promedio["anisotropia"].isna().all():
        return _vacia("Una sola dirección: sin índice de anisotropía", 260)
    else:
        pass
    p = promedio.sort_values("AB2")
    figura = go.Figure(go.Scatter(x=p["AB2"], y=p["anisotropia"], mode="lines+markers",
                                  line=dict(color=COLOR["preprocesado"])))
    figura.add_hline(y=1.3, line=dict(color=COLOR["anomalo"], dash="dash"),
                     annotation_text="1.3 · posible efecto lateral", annotation_position="top left")
    figura.update_xaxes(type="log", title="AB/2 (m)", **marcas_log(p["AB2"]))
    figura.update_yaxes(title="ρmáx / ρmín")
    figura.update_layout(template=PLANTILLA, height=260, title="Anisotropía azimutal aparente",
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura


# ======================================================================
# Preprocesamiento
# ======================================================================


def curva_preproceso(procesado: pd.DataFrame, titulo: str = "Curva observada y preprocesada") -> go.Figure:
    """Observada, referencia física, preprocesada y anómalos marcados."""
    if procesado is None or len(procesado) == 0:
        return _vacia("Sin preprocesamiento")
    else:
        pass
    p = procesado.sort_values(["AB2", "MN2"])
    figura = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26],
                           vertical_spacing=0.04)
    figura.add_trace(go.Scatter(
        x=p["AB2"], y=p["rhoa_obs"], mode="markers", name="Observada (promedio)",
        marker=dict(color=COLOR["observado"], size=8, symbol="circle-open", line=dict(width=1.5)),
        error_y=_barras_error(p["rhoa_obs"], np.maximum(np.nan_to_num(p["sigma_dir"], nan=0.0), 0)),
    ), row=1, col=1)
    xr, yr = _por_tramo(p, "rhoa_ref")
    figura.add_trace(go.Scatter(
        x=xr, y=yr, mode="lines", name="Referencia (ajuste físico robusto)",
        line=dict(color=COLOR["referencia"], dash="dash", width=1.5),
    ), row=1, col=1)
    usados = p[p["usar"]]
    xs, ys = _por_tramo(usados, "rhoa_proc")
    figura.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=COLOR["preprocesado"], width=2),
                                showlegend=False, hoverinfo="skip"), row=1, col=1)
    figura.add_trace(go.Scatter(
        x=usados["AB2"], y=usados["rhoa_proc"], mode="markers", name="Preprocesada",
        marker=dict(size=6, color=COLOR["preprocesado"]),
        error_y=_barras_error(usados["rhoa_proc"], usados["error_rel"]),
    ), row=1, col=1)
    anom = p[p["anomalo"]]
    if len(anom):
        figura.add_trace(go.Scatter(
            x=anom["AB2"], y=anom["rhoa_obs"], mode="markers", name="Dato tratado",
            marker=dict(color=COLOR["anomalo"], symbol="x", size=12, line=dict(width=2)),
            customdata=anom[["motivo"]].to_numpy(),
            hovertemplate="AB/2=%{x:.3g} m<br>ρa=%{y:.4g} Ω·m<br>%{customdata[0]}<extra></extra>",
        ), row=1, col=1)
    else:
        pass
    residuo_pct = 100 * (np.exp(p["residuo"]) - 1)
    figura.add_trace(go.Bar(
        x=p["AB2"], y=residuo_pct, name="Residual vs referencia (%)", showlegend=False,
        marker_color=np.where(p["anomalo"], COLOR["anomalo"], COLOR["observado"]),
        width=p["AB2"] * 0.08,
    ), row=2, col=1)
    _ejes_loglog(figura, 1, 1, valores_x=p["AB2"], valores_y=pd.concat([p["rhoa_obs"], p["rhoa_proc"]]))
    figura.update_xaxes(type="log", title="AB/2 (m)", row=2, col=1, **marcas_log(p["AB2"]))
    figura.update_xaxes(title=None, row=1, col=1)
    figura.update_yaxes(title="Residual (%)", row=2, col=1, zeroline=True)
    figura.update_layout(template=PLANTILLA, height=600, title=titulo,
                         legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def ip_preproceso(procesado: pd.DataFrame) -> go.Figure:
    """Cargabilidad aparente observada y preprocesada."""
    if procesado is None or not (procesado["ip_obs"].abs() > 0).any():
        return _vacia("El sondeo no tiene IP", 260)
    else:
        pass
    p = procesado.sort_values("AB2")
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=p["AB2"], y=p["ip_obs"], mode="markers", name="IP observada",
                                marker=dict(color=COLOR["observado"], symbol="circle-open", size=8)))
    figura.add_trace(go.Scatter(x=p["AB2"], y=p["ip_proc"], mode="lines+markers", name="IP preprocesada",
                                line=dict(color=COLOR["ip"]),
                                error_y=dict(type="data", array=p["error_ip"], color="#C9B6E4")))
    anom = p[p["ip_anomalo"].astype(bool)]
    if len(anom):
        figura.add_trace(go.Scatter(x=anom["AB2"], y=anom["ip_obs"], mode="markers", name="IP tratada",
                                    marker=dict(color=COLOR["anomalo"], symbol="x", size=11)))
    else:
        pass
    figura.update_xaxes(type="log", title="AB/2 (m)", **marcas_log(p["AB2"]))
    figura.update_yaxes(title="Cargabilidad aparente (mV/V)")
    figura.update_layout(template=PLANTILLA, height=340, title="IP aparente",
                         legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=20, t=50, b=40))
    return figura


# ======================================================================
# Inversión
# ======================================================================


def ajuste_inversion(ajuste: pd.DataFrame, mostrar_zohdy: bool = False,
                     titulo: str = "Resistividad aparente: observada vs calculada") -> go.Figure:
    """Curva observada (preprocesada) contra las respuestas de los modelos."""
    if ajuste is None or len(ajuste) == 0:
        return _vacia("Sin inversión")
    else:
        pass
    a = ajuste.sort_values(["AB2", "MN2"])
    figura = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26],
                           vertical_spacing=0.04)
    figura.add_trace(go.Scatter(
        x=a["AB2"], y=a["rhoa_obs"], mode="markers", name="Observada",
        marker=dict(color=COLOR["observado"], symbol="circle-open", size=8)), row=1, col=1)
    figura.add_trace(go.Scatter(
        x=a["AB2"], y=a["rhoa_proc"], mode="markers", name="Preprocesada (dato invertido)",
        marker=dict(color=COLOR["preprocesado"], size=7),
        error_y=_barras_error(a["rhoa_proc"], a["error_rel"])), row=1, col=1)
    if mostrar_zohdy and "rhoa_zohdy" in a:
        xs, ys = _por_tramo(a, "rhoa_zohdy")
        figura.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Calculada · Zohdy",
                                    line=dict(color=COLOR["zohdy"], width=1.3, dash="dot")), row=1, col=1)
        if "error_zohdy_pct" in a:
            figura.add_trace(go.Scatter(x=a["AB2"], y=a["error_zohdy_pct"], mode="markers+lines", showlegend=False,
                                        name="error Zohdy", line=dict(color=COLOR["zohdy"], width=1, dash="dot"),
                                        marker=dict(size=4)), row=2, col=1)
        else:
            pass
    else:
        pass
    xs, ys = _por_tramo(a, "rhoa_multicapa")
    figura.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Calculada · multicapa",
                                line=dict(color=COLOR["multicapa"], width=2)), row=1, col=1)
    xs, ys = _por_tramo(a, "rhoa_simplificado")
    figura.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Calculada · bloques (invertido)",
                                line=dict(color=COLOR["simplificado"], width=2.5)), row=1, col=1)
    for nombre, col, color in (("multicapa", "error_multicapa_pct", COLOR["multicapa"]),
                               ("simplificado", "error_simplificado_pct", COLOR["simplificado"])):
        figura.add_trace(go.Scatter(x=a["AB2"], y=a[col], mode="markers+lines", showlegend=False,
                                    name=f"error {nombre}", line=dict(color=color, width=1),
                                    marker=dict(size=5)), row=2, col=1)
    _ejes_loglog(figura, 1, 1, valores_x=a["AB2"],
                 valores_y=pd.concat([a["rhoa_obs"], a["rhoa_proc"], a["rhoa_multicapa"]]))
    figura.update_xaxes(type="log", title="AB/2 (m)", row=2, col=1, **marcas_log(a["AB2"]))
    figura.update_xaxes(title=None, row=1, col=1)
    figura.update_yaxes(title="Calc/obs − 1 (%)", row=2, col=1, zeroline=True)
    figura.update_layout(template=PLANTILLA, height=600, title=titulo,
                         legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def _escalones(modelo: pd.DataFrame, columna: str, z_fondo: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (valor, profundidad) de un modelo en escalones."""
    techos = modelo["z_techo"].to_numpy(float)
    bases = np.where(np.isfinite(modelo["z_base"]), modelo["z_base"], z_fondo).astype(float)
    valores = modelo[columna].to_numpy(float)
    x, z = [], []
    for t, b, v in zip(techos, bases, valores):
        x += [v, v]
        z += [t, min(b, z_fondo)]
    return np.array(x), np.array(z)


def modelos_resistividad(
    multicapa: Optional[pd.DataFrame], simplificado: Optional[pd.DataFrame],
    doi: Optional[float] = None, real: Optional[pd.DataFrame] = None,
    mostrar_zohdy: bool = False, z_max: Optional[float] = None,
    titulo: str = "Modelos de resistividad", zohdy: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Multicapa y simplificado en escalones (con incertidumbre y DOI)."""
    if multicapa is None and simplificado is None:
        return _vacia("Sin modelos")
    else:
        pass
    if z_max is None:
        candidatos = [doi * 1.3 if doi else np.nan]
        if simplificado is not None:
            finitas = simplificado["z_base"].replace(np.inf, np.nan).dropna()
            candidatos.append(float(finitas.max()) * 1.6 if len(finitas) else np.nan)
        else:
            pass
        z_max = float(np.nanmax(candidatos)) if np.isfinite(np.nanmax(candidatos)) else 100.0
    else:
        pass

    figura = go.Figure()
    barras = None
    if simplificado is not None and "rho_min" in simplificado:
        # Incertidumbre ±1σ de ρ como barra horizontal en el centro de cada
        # capa (acotada a ×/÷10 para no deformar el eje cuando una capa está
        # en equivalencia y su ρ por separado no está determinada).
        techos = simplificado["z_techo"].to_numpy(float)
        bases = np.where(np.isfinite(simplificado["z_base"]), simplificado["z_base"], z_max).astype(float)
        medios = np.sqrt(np.maximum(techos, 0.05 * bases) * np.minimum(bases, z_max))
        rho_s = simplificado["rho"].to_numpy(float)
        arriba = np.minimum(simplificado["rho_max"].to_numpy(float), rho_s * 10) - rho_s
        abajo = rho_s - np.maximum(simplificado["rho_min"].to_numpy(float), rho_s / 10)
        visibles = techos < z_max
        barras = go.Scatter(
            x=rho_s[visibles], y=medios[visibles], mode="markers", name="Bloques ±1σ (ρ)",
            marker=dict(color=COLOR["simplificado"], size=5),
            error_x=dict(type="data", symmetric=False, array=arriba[visibles], arrayminus=abajo[visibles],
                         color="rgba(0,0,255,0.45)", thickness=1.5, width=4),
            hovertemplate="ρ=%{x:.3g} Ω·m<extra>±1σ</extra>",
        )
    else:
        pass
    if real is not None:
        x, z = _escalones(real, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Modelo de referencia",
                                    line=dict(color="rgba(17,17,17,0.35)", width=5)))
    else:
        pass
    if mostrar_zohdy and zohdy is not None and len(zohdy):
        # Modelo de Zohdy propio: una capa por dato, con sus profundidades.
        x, z = _escalones(zohdy, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name=f"Zohdy ({len(zohdy)} capas)",
                                    line=dict(color=COLOR["zohdy"], width=1.2)))
    else:
        pass
    sin_tabla_zohdy = mostrar_zohdy and (zohdy is None or not len(zohdy))
    if sin_tabla_zohdy and multicapa is not None and "rho_zohdy" in multicapa:
        zo = multicapa.dropna(subset=["rho_zohdy"])
        x, z = _escalones(zo.assign(z_base=list(zo["z_base"].iloc[:-1]) + [np.inf]), "rho_zohdy", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Zohdy (inicial)",
                                    line=dict(color=COLOR["zohdy"], width=1.2)))
    else:
        pass
    if multicapa is not None:
        x, z = _escalones(multicapa, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Multicapa",
                                    line=dict(color=COLOR["multicapa"], width=2)))
    else:
        pass
    if simplificado is not None:
        x, z = _escalones(simplificado, "rho", z_max)
        texto = [f"Capa {c}: {r:.3g} Ω·m · {h:.3g} m · {res}" for c, r, h, res in zip(
            simplificado["capa"], simplificado["rho"], simplificado["espesor"],
            simplificado.get("resolucion", [""] * len(simplificado)))]
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Invertido (bloques)",
                                    line=dict(color=COLOR["simplificado"], width=3),
                                    text=np.repeat(texto, 2), hovertemplate="%{text}<br>z=%{y:.3g} m<extra></extra>"))
    else:
        pass
    valores_eje = []
    for tr in figura.data:
        if tr.x is not None:
            valores_eje += [v for v in tr.x if v is not None]
        else:
            pass
    if barras is not None:
        figura.add_trace(barras)
    else:
        pass
    if doi and doi <= z_max:
        figura.add_hline(y=doi, line=dict(color=COLOR["doi"], dash="dash", width=1.5),
                         annotation_text=f"DOI ≈ {doi:.1f} m", annotation_position="bottom right")
        figura.add_hrect(y0=doi, y1=z_max, fillcolor="rgba(120,120,120,0.08)", line_width=0)
    else:
        pass
    figura.update_xaxes(type="log", title="Resistividad (Ω·m)", showgrid=True, gridcolor="#E6E6E6",
                        **marcas_log(valores_eje))
    figura.update_yaxes(title="Profundidad (m)", range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6")
    figura.update_layout(template=PLANTILLA, height=600, title=titulo,
                         legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def parametros_acumulados(
    multicapa: Optional[pd.DataFrame], simplificado: Optional[pd.DataFrame],
    doi: Optional[float] = None, real: Optional[pd.DataFrame] = None, z_max: float = 100.0,
    zohdy: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Parámetros de Dar Zarrouk acumulados desde la superficie hasta z.

    1. T(z) = ∫₀ᶻ ρ dz: pendiente = ρ (tramo empinado → capa resistiva).
    2. S(z) = ∫₀ᶻ dz/ρ: pendiente = 1/ρ (tramo empinado → capa conductora).
    3. ρm(z) = √(ρL·ρT) = √(T/S), con ρL = z/S y ρT = T/z: resistividad
       promedio del paquete 0–z. Es un promedio **corrido**: sube mientras
       se atraviesa una capa más resistiva que el promedio de lo de arriba
       y baja en una más conductora; sus picos y valles marcan las
       interfaces donde cambia esa tendencia.
    4. λ(z) = √(ρT/ρL) = √(T·S)/z: pseudo-anisotropía del paquete 0–z
       (1 = homogéneo). Crece con cualquier contraste, sin distinguir si la
       resistividad sube o baja.
    5. Contraste de cada capa del simplificado: ρ_capa / ρm del paquete que
       tiene encima, en escala log₂ centrada en 1: a la derecha más
       resistiva, a la izquierda más conductora; ≥ ×2 o ≤ ×0.5 = cambio
       fuerte.
    """
    import parametros_sev as ps

    if multicapa is None and simplificado is None:
        return _vacia("Sin modelos")
    else:
        pass
    z = np.linspace(0.0, float(z_max), 400)[1:]
    figura = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.025,
                           column_widths=[1.1, 1.0, 1.0, 1.0, 0.6, 0.7],
                           subplot_titles=("Modelos de inversión", "T acumulada", "S acumulada", "ρ promedio 0–z",
                                           "Pseudo-anisotropía λ", "Contraste por capa"))
    series = [(real, "Modelo de referencia", dict(color="rgba(17,17,17,0.35)", width=5)),
              (multicapa, "Multicapa", dict(color=COLOR["multicapa"], width=2)),
              (simplificado, "Invertido (bloques)", dict(color=COLOR["simplificado"], width=3))]
    # Panel 1: los modelos de inversión (Zohdy, multicapa, invertido).
    for modelo, nombre, linea in ([(real, "Modelo de referencia", dict(color="rgba(17,17,17,0.35)", width=5)),
                                   (zohdy, "Zohdy", dict(color=COLOR["zohdy"], width=1.2))] + series[1:]):
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        x, zz = _escalones(modelo, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name=nombre, line=linea, legendgroup=nombre,
                                    hovertemplate="z=%{y:.1f} m<br>ρ=%{x:.3g} Ω·m<extra>" + nombre + "</extra>"),
                         row=1, col=1)
    ejes = {1: [], 2: [], 3: [], 4: []}
    for modelo, nombre, linea in series:
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        T = ps.curva_acumulada(modelo, z, "T")
        S = ps.curva_acumulada(modelo, z, "S")
        rm = np.sqrt(T / S)
        lam = np.sqrt(T * S) / z
        for col, valores, unidad in ((1, T, "Ω·m²"), (2, S, "S"), (3, rm, "Ω·m"), (4, lam, "")):
            ejes[col].append(valores)
            figura.add_trace(go.Scatter(
                x=valores, y=z, mode="lines", name=nombre, line=linea, legendgroup=nombre, showlegend=False,
                hovertemplate="z=%{y:.1f} m<br>%{x:.4g} " + unidad + "<extra>" + nombre + "</extra>",
            ), row=1, col=col + 1)
    # Contraste por capa (simplificado; el de referencia como contorno).
    for modelo, nombre, relleno in ((real, "Referencia", False), (simplificado, "Simplificado", True)):
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        m = ps.acumulados(modelo)
        m = m[np.isfinite(m["contraste_vs_arriba"]) & (m["z_techo"] < z_max)]
        if not len(m):
            continue
        else:
            pass
        techos = m["z_techo"].to_numpy(float)
        bases = np.minimum(np.where(np.isfinite(m["z_base"]), m["z_base"], z_max), z_max)
        c = m["contraste_vs_arriba"].to_numpy(float)
        x = np.log2(c)
        colores = np.where(c >= 1.0, COLOR["multicapa"], COLOR["simplificado"])
        figura.add_trace(go.Bar(
            x=x, y=(techos + bases) / 2, width=(bases - techos) * 0.92, orientation="h", name=f"Contraste · {nombre}",
            marker=dict(color=colores if relleno else "rgba(0,0,0,0)",
                        line=dict(color="rgba(17,17,17,0.55)" if not relleno else colores, width=2 if not relleno else 0)),
            customdata=np.stack([c, techos, bases], axis=-1), showlegend=False,
            hovertemplate=("capa %{customdata[1]:.1f}–%{customdata[2]:.1f} m<br>ρ capa / ρ promedio de arriba = "
                           "×%{customdata[0]:.2f}<extra>" + nombre + "</extra>"),
        ), row=1, col=6)
    for umbral in (-1.0, 1.0):
        figura.add_vline(x=umbral, line=dict(color="#999", dash="dot", width=1), row=1, col=6)
    figura.add_vline(x=0.0, line=dict(color="#666", width=1), row=1, col=6)
    figura.update_xaxes(title="ρ capa / ρ prom. arriba", tickvals=[-3, -2, -1, 0, 1, 2, 3],
                        ticktext=["×⅛", "×¼", "×½", "×1", "×2", "×4", "×8"], range=[-3.6, 3.6],
                        showgrid=True, gridcolor="#E6E6E6", row=1, col=6)
    valores_rho = [m["rho"].to_numpy(float) for m in (real, zohdy, multicapa, simplificado)
                   if m is not None and len(m)]
    vals_rho = np.concatenate(valores_rho) if valores_rho else np.array([1.0, 100.0])
    figura.update_xaxes(type="log", title="ρ (Ω·m)", showgrid=True, gridcolor="#E6E6E6", row=1, col=1,
                        **marcas_log(vals_rho[np.isfinite(vals_rho) & (vals_rho > 0)]))
    for col, titulo in ((1, "T (Ω·m²)"), (2, "S (siemens)"), (3, "ρ promedio (Ω·m)")):
        vals = np.concatenate(ejes[col]) if ejes[col] else np.array([1.0, 10.0])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        figura.update_xaxes(type="log", title=titulo, showgrid=True, gridcolor="#E6E6E6", row=1, col=col + 1,
                            **marcas_log(vals))
    lam_max = float(np.nanmax(np.concatenate(ejes[4]))) if ejes[4] else 1.5
    figura.update_xaxes(title="λ = √(ρT/ρL)", range=[0.98, max(1.1, lam_max * 1.05)], showgrid=True,
                        gridcolor="#E6E6E6", row=1, col=5)
    for col in range(1, 7):
        if doi and doi <= z_max:
            figura.add_hline(y=doi, line=dict(color=COLOR["doi"], dash="dash", width=1.2), row=1, col=col)
        else:
            pass
        figura.update_yaxes(range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6", row=1, col=col)
    figura.update_yaxes(title="Profundidad (m)", row=1, col=1)
    figura.update_layout(template=PLANTILLA, height=600,
                         title="Interpretación geoeléctrica: modelos y parámetros de Dar Zarrouk", bargap=0,
                         legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=20, t=70, b=40))
    return figura


COLORES_CORROSIVIDAD = {"Muy alta": "#B2182B", "Alta": "#EF6548", "Moderada": "#FDB863",
                        "Baja": "#A6D96A", "Muy baja": "#1A9850"}


def corrosividad(
    simplificado: Optional[pd.DataFrame], multicapa: Optional[pd.DataFrame] = None, z_max: float = 20.0,
    umbrales=(5.0, 10.0, 20.0, 100.0), prof_corrosion: float = 3.0,
) -> go.Figure:
    """Corrosividad del suelo estimada con el modelo invertido.

    1. Modelo de resistividad sobre las franjas de clase (ρ < 5: muy alta,
       5–10: alta, 10–20: moderada, 20–100: baja, > 100 Ω·m: muy baja; escala
       de protección catódica NACE/Roberge).
    2. Clase de cada capa del modelo invertido.
    3. Clase del suelo de 0 a z con la resistividad longitudinal ρL(z) =
       z / S(z), la que ve una corriente de corrosión (dominada por las
       capas conductoras); se marca la profundidad de evaluación.
    """
    import parametros_sev as ps

    if simplificado is None or not len(simplificado):
        return _vacia("Sin modelo invertido")
    else:
        pass
    clases = list(COLORES_CORROSIVIDAD)
    limites = [1e-3] + list(umbrales) + [1e6]
    figura = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                           column_widths=[1.2, 0.6, 1.2],
                           subplot_titles=("Modelo invertido y clases de corrosividad", "Clase por capa",
                                           "Suelo de 0 a z (ρL acumulada)"))
    for col in (1, 3):
        for clase, x0, x1 in zip(clases, limites[:-1], limites[1:]):
            # plotly 6+ (plotly.js 3): las figuras en ejes log usan valores de datos.
            figura.add_vrect(x0=x0, x1=x1, fillcolor=COLORES_CORROSIVIDAD[clase],
                             opacity=0.18, line_width=0, layer="below", row=1, col=col,
                             exclude_empty_subplots=False)
    if multicapa is not None and len(multicapa):
        x, zz = _escalones(multicapa, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name="Multicapa",
                                    line=dict(color=COLOR["multicapa"], width=2)), row=1, col=1)
    else:
        pass
    x, zz = _escalones(simplificado, "rho", z_max)
    figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name="Invertido (bloques)",
                                line=dict(color=COLOR["simplificado"], width=3)), row=1, col=1)
    # Clase por capa
    for fila in simplificado.itertuples():
        if fila.z_techo >= z_max:
            continue
        else:
            pass
        clase = ps.clase_corrosividad(float(fila.rho), umbrales)
        base = min(float(fila.z_base) if np.isfinite(fila.z_base) else z_max, z_max)
        figura.add_trace(go.Bar(
            x=[1.0], y=[(fila.z_techo + base) / 2], width=[(base - fila.z_techo) * 0.96], orientation="h",
            marker=dict(color=COLORES_CORROSIVIDAD.get(clase, "#CCC"), line=dict(color="#FFFFFF", width=1)),
            text=[f"{clase}<br>{fila.rho:.3g} Ω·m"], textposition="inside", insidetextanchor="middle",
            showlegend=False, hovertemplate=f"{fila.z_techo:.2f}–{base:.2f} m · {clase}<extra></extra>",
        ), row=1, col=2)
    # ρL acumulada
    z = np.linspace(0.0, float(z_max), 300)[1:]
    S = ps.curva_acumulada(simplificado, z, "S")
    rho_l = z / S
    figura.add_trace(go.Scatter(x=rho_l, y=z, mode="lines", name="ρL de 0 a z",
                                line=dict(color="#111111", width=2.5)), row=1, col=3)
    if prof_corrosion and prof_corrosion <= z_max:
        rl = float(prof_corrosion / ps.curva_acumulada(simplificado, [prof_corrosion], "S")[0])
        clase = ps.clase_corrosividad(rl, umbrales)
        figura.add_trace(go.Scatter(x=[rl], y=[prof_corrosion], mode="markers+text", name="Evaluación",
                                    marker=dict(size=11, color="#111111", symbol="diamond"),
                                    text=[f"  0–{prof_corrosion:g} m: {clase} ({rl:.3g} Ω·m)"],
                                    textposition="middle right", showlegend=False), row=1, col=3)
    else:
        pass
    todos = np.concatenate([simplificado["rho"].to_numpy(float), rho_l])
    todos = todos[np.isfinite(todos) & (todos > 0)]
    rango = [np.log10(max(min(todos.min() / 2, 2.0), 1e-2)), np.log10(max(todos.max() * 2, 200.0))]
    for col in (1, 3):
        figura.update_xaxes(type="log", range=rango, title="ρ (Ω·m)", showgrid=True, gridcolor="#E6E6E6",
                            row=1, col=col)
    figura.update_xaxes(visible=False, range=[0, 1], row=1, col=2)
    for col in (1, 2, 3):
        figura.update_yaxes(range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6", row=1, col=col)
    figura.update_yaxes(title="Profundidad (m)", row=1, col=1)
    for clase in clases:
        figura.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Corrosividad {clase.lower()}",
                                    marker=dict(size=12, symbol="square", color=COLORES_CORROSIVIDAD[clase])),
                         row=1, col=1)
    figura.update_layout(template=PLANTILLA, height=520, bargap=0,
                         title="Corrosividad del suelo estimada con el modelo invertido",
                         legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=20, t=70, b=40))
    return figura


def ajuste_ip(ajuste: pd.DataFrame) -> go.Figure:
    """IP aparente observada y calculada por ambos modelos."""
    if ajuste is None or not (ajuste["ip_proc"].abs() > 0).any():
        return _vacia("Sin IP", 260)
    else:
        pass
    a = ajuste.sort_values("AB2")
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=a["AB2"], y=a["ip_proc"], mode="markers", name="IP preprocesada",
                                marker=dict(color=COLOR["ip"], size=7),
                                error_y=dict(type="data", array=a["error_ip"], color="#C9B6E4")))
    figura.add_trace(go.Scatter(x=a["AB2"], y=a["ip_multicapa"], mode="lines", name="Calculada · multicapa",
                                line=dict(color=COLOR["multicapa"], width=2)))
    figura.add_trace(go.Scatter(x=a["AB2"], y=a["ip_simplificado"], mode="lines",
                                name="Calculada · bloques (invertido)",
                                line=dict(color=COLOR["simplificado"], dash="dash", width=2)))
    figura.update_xaxes(type="log", title="AB/2 (m)", **marcas_log(a["AB2"]))
    figura.update_yaxes(title="Cargabilidad aparente (mV/V)")
    figura.update_layout(template=PLANTILLA, height=420, title="IP aparente: observada vs calculada",
                         legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def modelos_ip(multicapa, simplificado, doi=None, real=None, z_max=None) -> go.Figure:
    """Cargabilidad intrínseca por capa (escalones)."""
    if multicapa is None or not (multicapa["m_ip"] > 0).any():
        return _vacia("Sin modelo de IP", 260)
    else:
        pass
    if z_max is None:
        z_max = (doi * 1.3) if doi else float(multicapa["z_techo"].max())
    else:
        pass
    figura = go.Figure()
    if real is not None and "m_ip" in real:
        x, z = _escalones(real, "m_ip", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Modelo de referencia",
                                    line=dict(color="rgba(17,17,17,0.35)", width=5)))
    else:
        pass
    for modelo, nombre, color, ancho in ((multicapa, "Multicapa", COLOR["multicapa"], 2.5),
                                         (simplificado, "Invertido (bloques)", COLOR["simplificado"], 3)):
        if modelo is not None:
            x, z = _escalones(modelo, "m_ip", z_max)
            figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name=nombre, line=dict(color=color, width=ancho)))
        else:
            pass
    if doi and doi <= z_max:
        figura.add_hline(y=doi, line=dict(color=COLOR["doi"], dash="dash"))
    else:
        pass
    figura.update_xaxes(title="Cargabilidad (mV/V)")
    figura.update_yaxes(title="Profundidad (m)", range=[z_max, 0])
    figura.update_layout(template=PLANTILLA, height=520, title="Modelos de IP",
                         legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def seleccion_capas(seleccion: pd.DataFrame, criterio: str = "BIC") -> go.Figure:
    """Criterio de información contra número de capas."""
    if seleccion is None or len(seleccion) == 0:
        return _vacia("Sin selección", 260)
    else:
        pass
    s = seleccion.sort_values("k")
    columna = criterio if criterio in s else "BIC"
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(go.Bar(x=s["k"], y=s[columna], name=columna,
                            marker_color=np.where(s["seleccionado"], COLOR["simplificado"], "#C7CDD6")),
                     secondary_y=False)
    figura.add_trace(go.Scatter(x=s["k"], y=s["chi2"], name="χ²", mode="lines+markers",
                                line=dict(color=COLOR["multicapa"])), secondary_y=True)
    figura.add_hline(y=1.0, line=dict(color=COLOR["multicapa"], dash="dot", width=1), secondary_y=True)
    figura.update_xaxes(title="Número de capas", dtick=1)
    # Escala log: el criterio cae varios órdenes de magnitud entre 2 capas y
    # el óptimo; en escala lineal solo se verían las primeras barras.
    positivo = s[columna].to_numpy(float)
    if np.all(positivo > 0):
        figura.update_yaxes(title=f"{columna} (log)", type="log", secondary_y=False)
    else:
        figura.update_yaxes(title=columna, secondary_y=False)
    figura.update_yaxes(title="χ²", type="log", exponentformat="power", secondary_y=True)
    figura.update_layout(template=PLANTILLA, height=320, title=f"Selección del número de capas ({columna})",
                         legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=60, t=50, b=40))
    return figura


def busqueda_lambda(lambdas: pd.DataFrame) -> go.Figure:
    """χ² y rugosidad de cada λ probado."""
    if lambdas is None or len(lambdas) == 0:
        return _vacia("Sin búsqueda de λ", 260)
    else:
        pass
    l = lambdas.sort_values("lam")
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(go.Scatter(x=l["lam"], y=l["chi2"], mode="markers+lines", name="χ²",
                                line=dict(color=COLOR["multicapa"])), secondary_y=False)
    figura.add_trace(go.Scatter(x=l["lam"], y=l["rugosidad"], mode="markers+lines", name="Rugosidad",
                                line=dict(color=COLOR["simplificado"], dash="dot")), secondary_y=True)
    sel = l[l["seleccionado"]]
    figura.add_trace(go.Scatter(x=sel["lam"], y=sel["chi2"], mode="markers", name="λ elegido",
                                marker=dict(size=14, symbol="star", color=COLOR["anomalo"])), secondary_y=False)
    figura.add_hline(y=1.0, line=dict(dash="dash", color="#999"), secondary_y=False)
    figura.update_xaxes(type="log", title="λ", **marcas_log(l["lam"]))
    figura.update_yaxes(type="log", title="χ²", secondary_y=False)
    figura.update_yaxes(title="Rugosidad (log₁₀ Ω·m)", secondary_y=True)
    figura.update_layout(template=PLANTILLA, height=320, title="Búsqueda de λ",
                         legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=60, t=50, b=40))
    return figura


# ======================================================================
# Campaña: mapa, columnas y perfiles
# ======================================================================


def mapa_sondeos(coords: pd.DataFrame, valores: Optional[pd.Series] = None, etiqueta_valor: str = "",
                 ruta: Optional[np.ndarray] = None, usar_log: bool = True,
                 proyectados: Optional[pd.DataFrame] = None) -> go.Figure:
    """Ubicación de los sondeos (coloreados por un parámetro opcional)."""
    if coords is None or len(coords) == 0:
        return _vacia("Sin coordenadas de sondeos")
    else:
        pass
    figura = go.Figure()
    if valores is not None:
        v = np.asarray(valores, float)
        color = np.log10(v) if usar_log else v
        figura.add_trace(go.Scatter(
            x=coords["X"], y=coords["Y"], mode="markers+text", text=coords["Sondeo"],
            textposition="top center", name="Sondeos",
            marker=dict(size=16, color=color, colorscale=ESCALA_RESISTIVIDAD, showscale=True,
                        line=dict(color="#333", width=1),
                        colorbar=dict(title=etiqueta_valor + (" (log₁₀)" if usar_log else ""))),
            customdata=v, hovertemplate="%{text}<br>" + etiqueta_valor + "=%{customdata:.4g}<extra></extra>",
        ))
    else:
        figura.add_trace(go.Scatter(x=coords["X"], y=coords["Y"], mode="markers+text", text=coords["Sondeo"],
                                    textposition="top center", name="Sondeos",
                                    marker=dict(size=12, color=COLOR["preprocesado"], line=dict(color="#333", width=1))))
    if ruta is not None and len(ruta) >= 2:
        r = np.asarray(ruta, float)
        figura.add_trace(go.Scatter(x=r[:, 0], y=r[:, 1], mode="lines+markers", name="Ruta",
                                    line=dict(color=COLOR["anomalo"], width=2), marker=dict(symbol="diamond", size=9)))
    else:
        pass
    if proyectados is not None and len(proyectados):
        for _, f in proyectados[proyectados["dentro"]].iterrows():
            figura.add_trace(go.Scatter(x=[f["X"], f["Xp"]], y=[f["Y"], f["Yp"]], mode="lines",
                                        showlegend=False, line=dict(color="#888", dash="dot", width=1)))
    else:
        pass
    figura.update_yaxes(scaleanchor="x", scaleratio=1, title="Y (m)", tickformat=",.0f")
    figura.update_xaxes(title="X (m)", tickformat=",.0f")
    figura.update_layout(template=PLANTILLA, height=520, title="Ubicación de sondeos",
                         margin=dict(l=60, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.15))
    return figura


def seccion(sec: dict[str, Any], titulo: str, escala=None, invertir: bool = False,
            unidad: str = "Ω·m", mostrar_doi: bool = True, rango: Optional[tuple[float, float]] = None) -> go.Figure:
    """Sección 2D (salida de ``perfil_sev.seccion_interpolada``).

    ``rango`` fija los límites de color (en log10 si la sección es
    logarítmica) para comparar varias secciones con la misma escala.
    """
    valores = sec["valores"]
    usar_log = sec.get("usar_log", True)
    datos = valores
    if escala is not None:
        datos = escala.recortar(valores)
        colorscale = escala.colorscale()
        zmin, zmax = escala.rango_dibujo
        mostrar_barra = False
    else:
        colorscale = ESCALA_IP if invertir else ESCALA_RESISTIVIDAD
        finitos = valores[np.isfinite(valores)]
        if rango is not None:
            zmin, zmax = rango
        else:
            zmin, zmax = (np.percentile(finitos, [1.5, 98.5]) if len(finitos) else (0, 1))
        mostrar_barra = True
    texto = np.where(np.isfinite(valores), 10 ** valores if usar_log else valores, np.nan)
    figura = go.Figure(go.Heatmap(
        x=sec["s"], y=sec["elev"], z=datos, colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=mostrar_barra, customdata=texto, zsmooth="best",
        colorbar=dict(title=f"log₁₀ {unidad}" if usar_log else unidad),
        hovertemplate="s=%{x:.1f} m<br>cota=%{y:.1f} m<br>valor=%{customdata:.4g} " + unidad + "<extra></extra>",
    ))
    if escala is not None:
        figura.add_trace(escala.traza_barra(unidad))
    else:
        pass
    figura.add_trace(go.Scatter(x=sec["s"], y=sec["topo"], mode="lines", name="Terreno",
                                line=dict(color="#333", width=2), showlegend=False))
    s_sev = sec["sondeos"]["s"].to_numpy(float)
    z_sev = sec["sondeos"]["Z"].to_numpy(float)
    figura.add_trace(go.Scatter(x=s_sev, y=z_sev, mode="markers+text", text=sec["sondeos"]["Sondeo"],
                                textposition="top center", marker=dict(symbol="triangle-down", size=12, color="#111"),
                                showlegend=False, hoverinfo="text"))
    for s_i, z_i in zip(s_sev, z_sev):
        figura.add_trace(go.Scatter(x=[s_i, s_i], y=[z_i, z_i - sec["z_max"]], mode="lines",
                                    line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                                    showlegend=False, hoverinfo="skip"))
    if mostrar_doi and sec.get("doi") is not None:
        # Solo el tramo de la DOI que cae dentro de la sección dibujada.
        doi = np.where(np.asarray(sec["doi"], float) <= sec["z_max"], sec["doi"], np.nan)
        figura.add_trace(go.Scatter(x=sec["s"], y=sec["topo"] - doi, mode="lines", name="DOI",
                                    line=dict(color="#FFFFFF", width=2, dash="dash"), showlegend=False,
                                    hovertemplate="DOI<extra></extra>"))
    else:
        pass
    figura.update_yaxes(range=[float(np.min(sec["topo"])) - sec["z_max"],
                               float(np.max(sec["topo"])) + 0.06 * sec["z_max"]])
    figura.update_xaxes(title="Distancia sobre la ruta (m)")
    figura.update_yaxes(title="Cota (m)" if np.ptp(sec["topo"]) > 0 or sec["topo"][0] != 0 else "Profundidad (m)")
    figura.update_layout(template=PLANTILLA, height=460, title=titulo,
                         margin=dict(l=60, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    return figura


def pseudoseccion(proyectados: pd.DataFrame, curvas: dict[str, pd.DataFrame], columna: str,
                  titulo: str, rango: Optional[tuple[float, float]] = None) -> go.Figure:
    """ρa en (cadenamiento, −profundidad mediana): la pseudosección del perfil."""
    filas = []
    for _, f in proyectados[proyectados["dentro"]].iterrows():
        tabla = curvas.get(str(f["Sondeo"]))
        if tabla is None or columna not in tabla:
            continue
        else:
            pass
        filas.append(pd.DataFrame({"s": f["s"], "z": tabla["z_mediana"], "valor": tabla[columna],
                                   "Sondeo": f["Sondeo"], "AB2": tabla["AB2"]}))
    if not filas:
        return _vacia("Sin curvas para la pseudosección")
    else:
        pass
    datos = pd.concat(filas, ignore_index=True)
    log_v = np.log10(datos["valor"].to_numpy(float))
    zmin, zmax = rango if rango else (np.percentile(log_v, 2), np.percentile(log_v, 98))
    figura = go.Figure(go.Scatter(
        x=datos["s"], y=datos["z"], mode="markers",
        marker=dict(size=15, symbol="square", color=log_v, colorscale=ESCALA_RESISTIVIDAD,
                    cmin=zmin, cmax=zmax, colorbar=dict(title="log₁₀ ρa")),
        customdata=np.column_stack([datos["valor"], datos["AB2"], datos["Sondeo"]]),
        hovertemplate="%{customdata[2]}<br>AB/2=%{customdata[1]:.3g} m<br>ρa=%{customdata[0]:.4g} Ω·m<extra></extra>",
    ))
    figura.update_xaxes(title="Distancia sobre la ruta (m)")
    # Eje log invertido: las aperturas de un SEV crecen geométricamente y en
    # escala lineal las someras quedarían apiñadas arriba.
    figura.update_yaxes(title="Profundidad mediana de investigación (m)", type="log",
                        autorange="reversed", **marcas_log(datos["z"]))
    figura.update_layout(template=PLANTILLA, height=380, title=titulo, margin=dict(l=60, r=20, t=50, b=40))
    return figura


def columnas_unidades(unidades: pd.DataFrame, orden: Optional[list[str]] = None,
                      z_max: Optional[float] = None) -> go.Figure:
    """Columnas estratigráficas por sondeo coloreadas por unidad."""
    if unidades is None or len(unidades) == 0:
        return _vacia("Sin unidades geoeléctricas")
    else:
        pass
    nombres = orden or sorted(unidades["Sondeo"].astype(str).unique())
    finitas = unidades["z_base"].replace(np.inf, np.nan)
    z_max = z_max or float(np.nanmax(finitas) * 1.4) if finitas.notna().any() else 50.0
    etiquetas = sorted(unidades["unidad"].unique(), key=lambda u: int(str(u)[1:]))
    paleta = [ESCALA_RESISTIVIDAD[int(i)][1] for i in np.linspace(2, len(ESCALA_RESISTIVIDAD) - 3, max(len(etiquetas), 2))]
    color_de = {u: paleta[i] for i, u in enumerate(etiquetas)}
    figura = go.Figure()
    vistos = set()
    for i, nombre in enumerate(nombres):
        capas = unidades[unidades["Sondeo"].astype(str) == nombre]
        for _, c in capas.iterrows():
            base = c["z_base"] if np.isfinite(c["z_base"]) else z_max
            figura.add_trace(go.Bar(
                x=[nombre], y=[min(base, z_max) - c["z_techo"]], base=[c["z_techo"]],
                marker_color=color_de[c["unidad"]], marker_line=dict(color="#333", width=0.5),
                name=c["unidad"], legendgroup=c["unidad"], showlegend=c["unidad"] not in vistos,
                text=f"{c['rho']:.3g}", textposition="inside",
                hovertemplate=(f"{nombre} · {c['unidad']}<br>ρ={c['rho']:.4g} Ω·m<br>"
                               f"{c['z_techo']:.3g}–{base:.3g} m<extra></extra>"),
            ))
            vistos.add(c["unidad"])
    figura.update_yaxes(range=[z_max, 0], title="Profundidad (m)")
    figura.update_layout(template=PLANTILLA, barmode="overlay", height=520, bargap=0.45,
                         title="Unidades geoeléctricas por sondeo (modelo simplificado)",
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura
