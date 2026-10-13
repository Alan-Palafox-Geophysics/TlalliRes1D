"""
escalas_color_tre.py
==================================================================
Modelos de escala de color para las secciones TRE/IP.

Este módulo es el **único** sitio donde se decide dónde caen los
cortes de color de una sección y qué color le toca a cada banda. Todos
los modos de graficación (puntos, contorno y 3D) consumen el mismo
objeto :class:`EscalaColor`, así que una misma configuración colorea
igual en las tres vistas.

Modelos disponibles
----------------------
``automatica``
    Los cortes se reparten de forma geométrica (equiespaciados en
    log₁₀) entre los percentiles de la campaña. Es la propuesta de la
    aplicación: la resistividad es aproximadamente log-normal, así que
    el reparto geométrico es el que distribuye el detalle de manera
    uniforme sobre la población de valores.
``lineal``
    Cortes equiespaciados en Ω·m entre los extremos del rango. Útil
    cuando el interés está en el contraste absoluto —una diferencia de
    100 Ω·m pesa lo mismo en todo el rango— y no en el relativo.
``logaritmica``
    Igual que la automática, pero con los extremos redondeados a
    valores de la serie 1-2-5 (10, 20, 50, 100…) y los cortes a dos
    cifras significativas. Es la escala que se lee sin esfuerzo en un
    informe, porque los números de la barra son redondos.
``gausiana``
    Cortes situados en cuantiles de una normal ajustada al rango (en
    log₁₀). Concentra el detalle donde está la masa de datos y deja
    bandas anchas en las colas, que es donde solo hay valores
    aislados. Es la escala que más estructura revela en una sección
    homogénea.
``customizable``
    Los cortes son exactamente los que indica el usuario, en cualquier
    posición. No hay ninguna restricción de reparto.

Cómo se consigue que los cortes arbitrarios funcionen de verdad
------------------------------------------------------------------
``go.Contour`` de Plotly **solo admite cortes equiespaciados**
(``start``/``end``/``size``), que es la razón por la que la versión
anterior de la escala personalizable no hacía nada: se le daban los
valores del usuario y Plotly los ignoraba, quedándose con el rango y
el paso medio.

La solución aquí no pasa por los cortes de Plotly, sino por la
**escala de color**: se construye una paleta *escalonada* cuyas
transiciones de color caen exactamente en la posición normalizada de
cada corte del usuario, y se colorea de forma continua
(``coloring="heatmap"`` en contorno, color por marcador en puntos).
El resultado son bandas con la frontera exacta pedida, sin aproximar,
y además con el dato original intacto en el *hover* y en la malla
—nada se transforma—.

La barra de color se dibuja aparte, como una traza invisible con
**una ranura por banda, todas del mismo ancho** y rotuladas con el
valor en Ω·m. Así la barra se lee como la tabla de la interfaz
(valor ↔ color), sin que los cortes apretados queden ilegibles.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``: toda condición adicional se
anida dentro del ``else`` de la anterior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.special import ndtri

from generar_escala_tre import generar_escala_tre

logger = logging.getLogger("tre.graficos.escala")

# ======================================================================
# Catálogo de modos
# ======================================================================

MODO_AUTOMATICA = "automatica"
MODO_LINEAL = "lineal"
MODO_LOGARITMICA = "logaritmica"
MODO_GAUSIANA = "gausiana"
MODO_CUSTOMIZABLE = "customizable"

#: Modos de escala disponibles -> etiqueta visible en la interfaz.
MODOS_ESCALA = {
    MODO_AUTOMATICA: "Automática (propuesta de la app)",
    MODO_LINEAL: "Lineal",
    MODO_GAUSIANA: "Gausiana",
    MODO_LOGARITMICA: "Logarítmica",
    MODO_CUSTOMIZABLE: "Customizable",
}

#: Modo por defecto: el que la aplicación propone sin intervención.
MODO_ESCALA_DEFECTO = MODO_AUTOMATICA

#: Número de bandas de color por defecto, y límites admitidos.
#:
#: Doce bandas es el punto en que la sección ya se lee como imagen
#: continua y la tabla de cortes de la interfaz sigue cabiendo en
#: pantalla sin desplazarse.
N_NIVELES_DEFECTO = 12
N_NIVELES_MIN = 4
N_NIVELES_MAX = 24

#: Colormap oficial del proyecto (ver ``generar_escala_tre``).
_COLORMAP = generar_escala_tre()


def colores_niveles(n_bandas: int, invertir: bool = False) -> list[str]:
    """Color hexadecimal de cada banda, muestreado del colormap oficial.

    Se muestrea en el **centro** de cada banda, no en su frontera: es
    el color que de verdad va a rellenar la banda, así que el recuadro
    que la interfaz dibuja junto a un corte coincide exactamente con
    lo que el usuario ve en la sección. (La versión anterior usaba una
    lista fija de diez colores, que dejaba de corresponder en cuanto el
    número de bandas no era diez.)
    """
    n = max(1, int(n_bandas))
    centros = (np.arange(n) + 0.5) / n
    if invertir:
        centros = 1.0 - centros
    else:
        pass

    colores: list[str] = []
    for centro in centros:
        r, g, b, _ = _COLORMAP(float(centro))
        colores.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return colores


# ======================================================================
# Generación de cortes por modo
# ======================================================================


def _mantisa_redonda(valor: float, hacia_arriba: bool) -> float:
    """Redondea ``valor`` al elemento más cercano de la serie 1-2-5.

    Hacia arriba o hacia abajo según ``hacia_arriba``. Es lo que
    convierte un extremo como 37.4 Ω·m en 50 (o en 20), que es lo que
    se espera leer en la barra de un informe.
    """
    if not np.isfinite(valor) or valor <= 0:
        return 1.0
    else:
        pass

    decada = np.floor(np.log10(valor))
    mantisa = valor / (10.0 ** decada)
    serie = (1.0, 2.0, 5.0, 10.0)

    if hacia_arriba:
        elegida = next((s for s in serie if s >= mantisa - 1e-9), 10.0)
    else:
        candidatas = [s for s in serie if s <= mantisa + 1e-9]
        elegida = candidatas[-1] if candidatas else 1.0
    return float(elegida * (10.0 ** decada))


def _cifras_significativas(valor: float, cifras: int = 2) -> float:
    """Redondea a ``cifras`` cifras significativas (para que los cortes
    de la escala logarítmica salgan en números legibles)."""
    if not np.isfinite(valor) or valor == 0:
        return float(valor)
    else:
        pass
    magnitud = np.floor(np.log10(abs(valor)))
    factor = 10.0 ** (cifras - 1 - magnitud)
    return float(np.round(valor * factor) / factor)


def niveles_automaticos(rango_dibujo: Sequence[float], n_niveles: int) -> list[float]:
    """Cortes equiespaciados en el espacio de dibujo (log₁₀ para
    resistividad), entre los extremos del rango."""
    bajo, alto = float(rango_dibujo[0]), float(rango_dibujo[1])
    return [float(v) for v in np.linspace(bajo, alto, int(n_niveles) + 1)]


def niveles_lineales(
    rango_dibujo: Sequence[float], n_niveles: int, usar_log: bool
) -> list[float]:
    """Cortes equiespaciados en **unidades de la variable** (Ω·m o mV/V).

    En escala logarítmica esto concentra las bandas en los valores
    altos, lo cual es exactamente lo que se pide cuando interesa el
    contraste absoluto y no el relativo.
    """
    bajo, alto = float(rango_dibujo[0]), float(rango_dibujo[1])
    if usar_log:
        inicio, fin = 10.0 ** bajo, 10.0 ** alto
    else:
        inicio, fin = bajo, alto

    datos = np.linspace(inicio, fin, int(n_niveles) + 1)
    if usar_log:
        datos = np.clip(datos, 1e-9, None)
        return [float(np.log10(v)) for v in datos]
    else:
        return [float(v) for v in datos]


def niveles_logaritmicos(
    rango_dibujo: Sequence[float], n_niveles: int, usar_log: bool
) -> list[float]:
    """Cortes geométricos entre extremos redondeados a la serie 1-2-5,
    con los valores intermedios a dos cifras significativas.

    Es la variante "de informe" de la automática: mismo reparto, pero
    números redondos en la barra de color.
    """
    bajo, alto = float(rango_dibujo[0]), float(rango_dibujo[1])
    if not usar_log:
        # Sin logaritmo no hay nada que redondear por décadas: se cae al
        # reparto uniforme sobre el propio rango.
        return niveles_automaticos(rango_dibujo, n_niveles)
    else:
        pass

    inicio = _mantisa_redonda(10.0 ** bajo, hacia_arriba=False)
    fin = _mantisa_redonda(10.0 ** alto, hacia_arriba=True)
    if fin <= inicio:
        fin = inicio * 10.0
    else:
        pass

    crudos = np.logspace(np.log10(inicio), np.log10(fin), int(n_niveles) + 1)
    redondeados = [_cifras_significativas(v) for v in crudos]
    # El redondeo puede empatar dos cortes consecutivos; se fuerza la
    # monotonía estricta para que ninguna banda quede de ancho cero.
    saneados: list[float] = []
    for valor in redondeados:
        if saneados and valor <= saneados[-1]:
            valor = saneados[-1] * 1.02
        else:
            pass
        saneados.append(float(valor))
    return [float(np.log10(max(v, 1e-9))) for v in saneados]


def niveles_gausianos(
    rango_dibujo: Sequence[float],
    n_niveles: int,
    percentil_bajo: float,
    percentil_alto: float,
) -> list[float]:
    """Cortes en cuantiles de una normal ajustada al rango.

    Los extremos del rango son, por construcción, los percentiles
    ``percentil_bajo`` y ``percentil_alto`` de los datos en el espacio
    de dibujo. De ahí se despeja la media y la desviación de la normal
    equivalente, y los cortes se sitúan en cuantiles equiespaciados en
    **probabilidad**: cada banda recoge aproximadamente la misma
    cantidad de celdas.

    El efecto práctico es el inverso de la escala lineal: mucho detalle
    en el cuerpo de la distribución —donde está la estructura
    geológica— y bandas anchas en las colas, que suelen ser artefactos
    o celdas de borde.
    """
    bajo, alto = float(rango_dibujo[0]), float(rango_dibujo[1])
    probabilidad_baja = max(min(percentil_bajo / 100.0, 0.49), 1e-4)
    probabilidad_alta = min(max(percentil_alto / 100.0, 0.51), 1 - 1e-4)

    z_bajo = float(ndtri(probabilidad_baja))
    z_alto = float(ndtri(probabilidad_alta))
    if z_alto <= z_bajo:
        return niveles_automaticos(rango_dibujo, n_niveles)
    else:
        pass

    media = (alto * z_bajo - bajo * z_alto) / (z_bajo - z_alto)
    desviacion = (alto - bajo) / (z_alto - z_bajo)

    probabilidades = np.linspace(probabilidad_baja, probabilidad_alta, int(n_niveles) + 1)
    return [float(media + desviacion * float(ndtri(p))) for p in probabilidades]


def niveles_por_modo(
    modo: str,
    rango_dibujo: Optional[Sequence[float]],
    n_niveles: int = N_NIVELES_DEFECTO,
    usar_log: bool = True,
    niveles_usuario: Optional[Sequence[float]] = None,
    percentil_bajo: float = 1.5,
    percentil_alto: float = 98.5,
) -> list[float]:
    """Cortes de la escala, **en el espacio de dibujo** (log₁₀ si
    ``usar_log``), para el modo indicado.

    Returns
    -------
    list[float]
        Fronteras de banda en orden ascendente, con
        ``n_niveles + 1`` elementos para los modos automáticos. Lista
        vacía si no hay rango con el que trabajar.
    """
    if modo == MODO_CUSTOMIZABLE:
        return _niveles_usuario_saneados(niveles_usuario, usar_log)
    else:
        pass

    if not rango_dibujo or len(rango_dibujo) != 2:
        return []
    else:
        pass
    bajo, alto = float(rango_dibujo[0]), float(rango_dibujo[1])
    if not (np.isfinite(bajo) and np.isfinite(alto)) or alto <= bajo:
        return []
    else:
        pass

    n = int(np.clip(int(n_niveles), N_NIVELES_MIN, N_NIVELES_MAX))

    if modo == MODO_LINEAL:
        return niveles_lineales((bajo, alto), n, usar_log)
    else:
        pass
    if modo == MODO_LOGARITMICA:
        return niveles_logaritmicos((bajo, alto), n, usar_log)
    else:
        pass
    if modo == MODO_GAUSIANA:
        return niveles_gausianos((bajo, alto), n, percentil_bajo, percentil_alto)
    else:
        pass
    return niveles_automaticos((bajo, alto), n)


def _niveles_usuario_saneados(
    niveles_usuario: Optional[Sequence[float]], usar_log: bool
) -> list[float]:
    """Convierte la lista del usuario (en Ω·m / mV/V) al espacio de
    dibujo, ordenada, sin repetidos y sin valores no representables.

    Que esta función exista —y que se llame siempre— es lo que impide
    que un corte mal escrito (negativo con escala logarítmica, o dos
    cortes iguales) deje la sección en blanco o con una banda de ancho
    cero.
    """
    if not niveles_usuario:
        return []
    else:
        pass

    validos: list[float] = []
    for valor in niveles_usuario:
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numero):
            continue
        else:
            pass
        if usar_log:
            if numero <= 0:
                continue
            else:
                validos.append(float(np.log10(numero)))
        else:
            validos.append(numero)

    validos.sort()
    saneados: list[float] = []
    for valor in validos:
        if saneados and abs(valor - saneados[-1]) < 1e-9:
            continue
        else:
            saneados.append(valor)
    return saneados


def parsear_vector(texto: Optional[str]) -> tuple[list[float], list[str]]:
    """Interpreta un vector de valores separados por coma.

    Es el mismo formato que usa el plot 3D de la campaña: es más rápido
    de escribir que una casilla por valor, y admite cualquier valor sin
    tener que existir previamente como corte de la escala.

    Returns
    -------
    tuple[list[float], list[str]]
        Valores reconocidos y los fragmentos que no eran numéricos
        (para que la interfaz los avise en vez de ignorarlos en
        silencio).
    """
    valores: list[float] = []
    rechazados: list[str] = []
    if not texto:
        return valores, rechazados
    else:
        pass

    for fragmento in str(texto).replace(";", ",").replace("\n", ",").split(","):
        limpio = fragmento.strip()
        if not limpio:
            continue
        else:
            pass
        try:
            valores.append(float(limpio))
        except ValueError:
            rechazados.append(limpio)
    return valores, rechazados


def formato_valor(valor: float) -> str:
    """Formato de lectura de un valor de la barra de color."""
    if not np.isfinite(valor):
        return "—"
    else:
        pass
    magnitud = abs(valor)
    if magnitud >= 1000:
        return f"{valor:,.0f}"
    else:
        pass
    if magnitud >= 100:
        return f"{valor:.0f}"
    else:
        pass
    if magnitud >= 10:
        return f"{valor:.1f}"
    else:
        pass
    return f"{valor:.2f}"


# ======================================================================
# Objeto de escala
# ======================================================================


@dataclass
class EscalaColor:
    """Escala de color resuelta: cortes, paleta escalonada y barra.

    Attributes
    ----------
    niveles_dibujo : list[float]
        Fronteras de banda en el espacio en que se colorea (log₁₀ para
        resistividad). Es el dato primario: todo lo demás se deriva.
    usar_log : bool
        Si el espacio de dibujo es logarítmico.
    invertir : bool
        Usar el colormap invertido (es lo que corresponde a IP, por
        especificación del proyecto).
    modo : str
        Modo con que se generaron los cortes (informativo).
    unidad : str
        Unidad para rotular la barra.
    """

    niveles_dibujo: list[float]
    usar_log: bool = True
    invertir: bool = False
    modo: str = MODO_ESCALA_DEFECTO
    unidad: str = "Ω·m"

    # ---- Derivados ----------------------------------------------------

    @property
    def n_bandas(self) -> int:
        """Número de bandas de color (una menos que cortes)."""
        return max(1, len(self.niveles_dibujo) - 1)

    @property
    def rango_dibujo(self) -> list[float]:
        """``[mínimo, máximo]`` en el espacio de dibujo."""
        return [float(self.niveles_dibujo[0]), float(self.niveles_dibujo[-1])]

    @property
    def niveles_datos(self) -> list[float]:
        """Cortes en unidades de la variable (Ω·m / mV/V)."""
        if self.usar_log:
            return [float(10.0 ** v) for v in self.niveles_dibujo]
        else:
            return [float(v) for v in self.niveles_dibujo]

    @property
    def colores(self) -> list[str]:
        """Color de cada banda, en orden ascendente de valor."""
        return colores_niveles(self.n_bandas, invertir=self.invertir)

    def es_valida(self) -> bool:
        """Una escala sirve si tiene al menos dos cortes distintos."""
        if len(self.niveles_dibujo) < 2:
            return False
        else:
            pass
        bajo, alto = self.rango_dibujo
        return bool(np.isfinite(bajo) and np.isfinite(alto) and alto > bajo)

    # ---- Paletas ------------------------------------------------------

    def colorscale(self) -> list:
        """Paleta **escalonada** cuyas transiciones caen exactamente en
        los cortes, expresadas en posición normalizada dentro del rango.

        Es el mecanismo que hace funcionar los cortes arbitrarios: no se
        toca el dato ni se pide a Plotly que reparta las bandas, se le
        da una paleta que ya cambia de color en el sitio correcto. Con
        ``cmin``/``cmax`` igual a los extremos del rango, la frontera de
        cada banda queda en el valor pedido, sin aproximación.
        """
        bajo, alto = self.rango_dibujo
        amplitud = alto - bajo
        if amplitud <= 0:
            amplitud = 1e-9
        else:
            pass

        colores = self.colores
        escala: list = []
        for indice in range(self.n_bandas):
            inicio = (self.niveles_dibujo[indice] - bajo) / amplitud
            fin = (self.niveles_dibujo[indice + 1] - bajo) / amplitud
            color = colores[indice]
            escala.append([float(np.clip(inicio, 0.0, 1.0)), color])
            escala.append([float(np.clip(fin, 0.0, 1.0)), color])
        # Plotly exige que la paleta empiece en 0 y acabe en 1.
        escala[0][0] = 0.0
        escala[-1][0] = 1.0
        return escala

    def colorscale_barra(self) -> list:
        """Paleta de la barra de color, con **una ranura por banda,
        todas del mismo ancho**.

        Separar la paleta de la barra de la del dibujo es deliberado:
        en el dibujo las bandas deben ocupar su posición real (si no, el
        color miente), pero en la barra deben ocupar lo mismo, porque la
        barra es una leyenda —la tabla valor ↔ color— y con reparto
        proporcional los cortes apretados se vuelven ilegibles.
        """
        colores = self.colores
        n = self.n_bandas
        escala: list = []
        for indice in range(n):
            color = colores[indice]
            escala.append([float(indice) / n, color])
            escala.append([float(indice + 1) / n, color])
        escala[0][0] = 0.0
        escala[-1][0] = 1.0
        return escala

    # ---- Transformación y barra --------------------------------------

    def recortar(self, valores: Any) -> np.ndarray:
        """Acota los valores al rango de la escala.

        Sin esto, todo lo que cae fuera del rango se dibuja igual que el
        extremo (Plotly satura por su cuenta), lo cual es correcto, pero
        recortar explícitamente deja el ``hover`` coherente con el color.
        """
        bajo, alto = self.rango_dibujo
        return np.clip(np.asarray(valores, dtype="float64"), bajo, alto)

    def barra(
        self, titulo: str, horizontal: bool = False, longitud: float = 0.6
    ) -> dict:
        """Configuración de ``colorbar`` con una marca por corte,
        rotulada en unidades de la variable.

        El título pierde el prefijo ``log₁₀``: la barra ya no muestra
        logaritmos, muestra Ω·m, y dejarlo haría leer "500" como si
        fuera un exponente.
        """
        etiqueta_eje = str(titulo).replace("log₁₀", "").strip() or self.unidad
        etiquetas = [formato_valor(v) for v in self.niveles_datos]
        posiciones = list(range(self.n_bandas + 1))

        barra: dict = {
            "thickness": 14 if horizontal else 12,
            "tickvals": posiciones,
            "ticktext": etiquetas,
            "tickmode": "array",
            "tickfont": dict(size=10),
            # Rotulada al costado y no encima: con una marca por corte, la
            # marca superior llega al borde de la barra y un título arriba
            # se le monta encima.
            "title": dict(text=etiqueta_eje, side="right"),
        }
        if horizontal:
            barra.update({
                "title": dict(text=etiqueta_eje, side="top"),
                "orientation": "h", "x": 0.5, "xanchor": "center",
                "y": -0.02, "yanchor": "top", "len": longitud,
            })
        else:
            pass
        return barra

    def traza_barra(
        self,
        titulo: str,
        horizontal: bool = False,
        longitud: float = 0.6,
        tridimensional: bool = False,
    ) -> Any:
        """Traza invisible que **solo** dibuja la barra de color.

        No lleva datos (un único punto nulo), así que no añade peso a la
        figura; existe para que la barra pueda tener su propio reparto
        —una ranura por banda— independientemente de cómo estén
        repartidos los cortes en el dibujo.

        ``tridimensional`` devuelve un ``Scatter3d`` en vez de un
        ``Scatter``: en una figura con escena 3D, una traza cartesiana
        crearía un sistema de ejes 2D superpuesto y rompería el
        encuadre de la escena.
        """
        marcador = dict(
            size=0.1, color=[0.0, float(self.n_bandas)],
            colorscale=self.colorscale_barra(),
            cmin=0.0, cmax=float(self.n_bandas),
            showscale=True,
            colorbar=self.barra(titulo, horizontal=horizontal, longitud=longitud),
        )
        if tridimensional:
            return go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=marcador, showlegend=False, hoverinfo="skip", name="escala",
            )
        else:
            return go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=marcador, showlegend=False, hoverinfo="skip", name="escala",
            )

    def resumen(self) -> str:
        """Descripción corta para la interfaz."""
        datos = self.niveles_datos
        return (
            f"{MODOS_ESCALA.get(self.modo, self.modo)} · {self.n_bandas} bandas · "
            f"{formato_valor(datos[0])} a {formato_valor(datos[-1])} {self.unidad}"
        )


def construir_escala(
    modo: str = MODO_ESCALA_DEFECTO,
    rango_dibujo: Optional[Sequence[float]] = None,
    n_niveles: int = N_NIVELES_DEFECTO,
    usar_log: bool = True,
    niveles_usuario: Optional[Sequence[float]] = None,
    invertir: bool = False,
    unidad: str = "Ω·m",
    percentil_bajo: float = 1.5,
    percentil_alto: float = 98.5,
) -> Optional[EscalaColor]:
    """Construye la :class:`EscalaColor` de un modo, o ``None`` si no hay
    información suficiente (sin rango y sin cortes del usuario).

    Devolver ``None`` —y no una escala degenerada— es intencionado: los
    módulos de graficación interpretan ``None`` como "colorea como
    siempre", así que una configuración incompleta nunca deja una
    sección en blanco.
    """
    niveles = niveles_por_modo(
        modo, rango_dibujo, n_niveles=n_niveles, usar_log=usar_log,
        niveles_usuario=niveles_usuario,
        percentil_bajo=percentil_bajo, percentil_alto=percentil_alto,
    )
    if len(niveles) < 2:
        return None
    else:
        pass

    escala = EscalaColor(
        niveles_dibujo=[float(v) for v in niveles], usar_log=usar_log,
        invertir=invertir, modo=modo, unidad=unidad,
    )
    if not escala.es_valida():
        return None
    else:
        pass
    return escala
