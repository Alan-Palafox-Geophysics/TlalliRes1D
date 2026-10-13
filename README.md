# TLALLI-RES 1D — Sondeos Eléctricos Verticales e IP

Aplicación Streamlit para el flujo completo de SEV/IP, con la misma arquitectura que
TlalliRes2D (autenticación y licencias, proyecto `.isev` descargable, Parquet + SQLite +
DuckDB, lógica fuera de la interfaz y pruebas sin navegador):

```
lectura y QA/QC → empalmes → promedio de direcciones → preprocesamiento
→ 1. Zohdy (búsqueda de AB/n) → 2. multicapa (pyGIMLi, Occam/L1, λ por Optuna) → DOI
→ 3. Dar Zarrouk de Zohdy y del multicapa (gana el que mejor ajusta)
→ 4. modelo de bloques (Optuna + LM con espesor mínimo, BIC) → IP (Seigel)
→ S, T, ρL, ρT, λ, ρm, corrosividad → unidades geoeléctricas (GMM)
→ perfiles 2D: pseudosecciones, secciones interpoladas e inversión LCI (pyGIMLi)
```

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Las credenciales se leen igual que en TlalliRes2D (variable `TRE_CREDENCIALES_YAML` o
`TRE_CREDENCIALES_RUTA`, secretos de Streamlit o `app/credenciales.yaml`). Se incluye una
copia del `credenciales.yaml` de TlalliRes2D, así que funcionan las mismas cuentas.

Pruebas: `python app/test_app.py` (21 pruebas, ~4 min) o `python app/test_app.py rapido`.

## Formato de datos

Texto con columnas `Sondeo, ID, a, AB/2, MN/2, K, I, V, R, ResApp, Direccion, IP`.

- El separador (tabulador, coma, punto y coma o espacios) y la coma decimal se detectan
  solos; los nombres de columna admiten sinónimos y acentos.
- `IP` es opcional (cero si falta); `Direccion` también (una sola dirección `U`).
- Un archivo puede traer varios sondeos y varias direcciones por sondeo.
- **Mínimo:** solo `AB_2, MN_2, AppRes` (o `AB/2, MN/2, ResApp`). En ese caso no se
  recalcula nada (sin K, I, V ni R no hay qué recalcular): se usa la ρa del archivo tal
  cual, sin corrección de empalmes, y solo se aplica el suavizado opcional y la inversión.
  Se asume Schlumberger salvo que se fuerce otro arreglo.
- Coordenadas aparte: `Sondeo, X, Y, Z` (o se escriben en la tabla de la pantalla 1).
- Modelo de referencia opcional por sondeo (pozo o sintético): `espesor, rho[, m_ip]`.

### Arreglo y convención de columnas

La columna `AB/2` no tiene un significado único en los arreglos dipolares. La
aplicación prueba todas las convenciones razonables y elige la que **reproduce la
columna K** (`geometria_sev.CONVENCIONES`):

| Arreglo | Convenciones probadas |
|---|---|
| Schlumberger | A,B = ∓AB/2; M,N = ∓MN/2 |
| Wenner | `a` de su columna, o a = 2/3·AB/2 |
| Dipolo-dipolo | AB/2 = distancia entre centros (n+1)a · = n·a · = n · = mitad entre centros |
| Polo-dipolo | AB/2 = A–centro de MN (n+½)a · = n·a · = n |
| Polo-polo | r = AB/2 · = a · = 2·AB/2 |

Si varias convenciones reproducen K por igual, gana la que lo hace en todas las filas.
Si siguen empatadas, gana el arreglo que indica la geometría.

Si la columna K no corresponde a ningún arreglo (error mediano > 25 %), no se usa:
- se prueba si la columna R es el factor geométrico (K y R intercambiadas al exportar);
- si tampoco, se usan reglas geométricas.

Si ResApp = K·V (y no K·V/I), se interpreta que la columna V trae la resistencia ΔV/I
en Ω, y se avisa. Sin columna K se usan reglas geométricas y la convención por defecto. El arreglo puede
forzarse en la pantalla 1.

## Decisiones técnicas

**Modelado directo.** Transformada de Koefoed con el filtro de Hankel J0 de 401 puntos de
Key (2009), vectorizada sobre todas las mediciones, y **jacobiano analítico** (derivada de
la recurrencia capa a capa). Error < 1e-5 frente a la solución analítica de dos capas y
coincidencia de 1e-7 con el jacobiano por diferencias de pyGIMLi. La inversión sigue
siendo `pg.Inversion` de pyGIMLi; solo cambia quién evalúa la física. El núcleo C++ de
pyGIMLi queda como motor alternativo seleccionable.

| | pyGIMLi C++ | Motor vectorizado |
|---|---|---|
| Jacobiano de 21 capas | 1.2 s | 0.034 s |
| Inversión Occam (28 capas) | 1.3 s | 0.18 s |

Nota: `pygimli.physics.ves.VESModelling` (1.5–1.6) pasa las distancias al núcleo C++ en
orden `(am, bm, an, bn)` en lugar de `(am, an, bm, bn)`. En Schlumberger/Wenner no se nota;
en dipolo-dipolo y polo-dipolo sí. Aquí se llama al núcleo con el orden correcto.

**Promedio de direcciones.** Media geométrica (log ρa) sobre la geometría de la dirección
con más datos; las demás se interpolan en log-log sin extrapolar. El ruido por dirección
se estima **agrupado** (mediana de varianzas corregida por el sesgo de χ²ν) y el error de
cada punto es el mayor entre σ_agrupada/√n y la dispersión de ese punto /√n.
- **Con 2 direcciones** no se puede saber cuál es la buena: siempre se usa la media
  geométrica y nunca se descarta una dirección. Si discrepan, solo aumenta el error del
  dato (pesa menos en la inversión).
- **Con 3 o más direcciones** hay mayoría: si una discrepa de la referencia y las demás
  no, se descarta la discrepante y se rehace el promedio con las coherentes.

**Empalmes de MN.** Se mide el salto sobre los residuales de un ajuste físico con la
geometría exacta de cada lectura. Así no se borra la parte física del salto (MN no
despreciable frente a AB). Solo se corrige si el salto es significativo (3σ) y si los
traslapes son coherentes entre sí.

**Detección de anómalos.** Se usa como referencia la curva de una inversión suave robusta
(IRLS ≈ L1): mide lo que ningún medio 1D explica. Un dato es anómalo si cumple:
- se sale de la tendencia **de su propio sondeo** (z robusto > 3 frente a la escala de
  residuales del sondeo) **y** de la **de la campaña** (z > 3 frente a la escala mediana
  de los demás sondeos) y su error es > 10 %; o
- dispersión entre direcciones > 20 % y > 2.5σ, solo con 3 o más direcciones; o
- pendiente ascendente > 45° (SCH/WE).

La escala de la campaña se obtiene en dos pasadas: al procesar un archivo, primero se
promedian y preprocesan todos los sondeos; después se repite el preprocesamiento de cada
uno con la escala mediana de los demás y se continúa con la inversión. Al reprocesar un
sondeo suelto se usa la escala guardada de los demás. Así no se descartan datos de un
sondeo ruidoso por compararlo solo consigo mismo, ni saltos que son normales en la campaña.

Tratamientos disponibles: suavizar (por defecto), ponderar o eliminar.

### Proceso de inversión (pantalla 3)

La inversión de cada sondeo sigue cuatro pasos. Cada paso puede ir en **automático** o
con **parámetros fijos**, y la configuración completa se guarda para toda la campaña
(ver "Parámetros de campaña"). La pantalla muestra, para cada sondeo, el texto de lo que
se hizo realmente en cada paso (n de Zohdy elegido, λ, fuente de Dar Zarrouk, pruebas de
Optuna, k elegido).

**Paso 1 — Zohdy (modelo inicial multicapa).** A cada dato se le asigna la base de su
capa en z = AB/n y su resistividad se corrige iterativamente hasta que la curva calculada
reproduce la observada. Se prueban varios n y cada uno se evalúa con Zohdy completo
(con las resistividades ya corregidas); gana el n de menor RMS.
- **Modo Edwards (por omisión):** n = n_Edwards ± 1 en pasos de 0.125. n_Edwards es la
  mediana de AB/z_e (≈ 5.3 en Schlumberger, ≈ 5.8 en Wenner). La ventana (±1) y el paso
  son configurables.
- **Modo rango:** n entre un mínimo y un máximo (p. ej. 5–7) con el paso elegido.
- **Modo factor (anterior):** factor sobre la profundidad de Edwards, elegido antes de
  corregir ρ.
- La app avisa si el óptimo queda en el borde del rango. En arreglos sin AB definido se
  usa una longitud equivalente de 5.3·z_e.

**Paso 2 — Multicapa con pyGIMLi.** Profundidades fijas; solo se invierte ρ, partiendo
del modelo de Zohdy.
- **Malla:** las interfaces de Zohdy subdivididas en log z (~50 celdas) hasta la
  **profundidad máxima viable** f·máx(z_e), donde f es el factor de profundidad que mejor
  ajusta. La malla logarítmica sigue disponible.
- **Regularización:** Occam suave (L2) por omisión; **robusta** (L1 en los datos, IRLS de
  pyGIMLi, resiste datos malos) seleccionable; nítida (gradiente mínimo) como opción.
- **λ por Optuna:** se minimiza |ln χ²| + 0.1·rugosidad + una penalización por
  residuales correlacionados (λ entre 0.1 y 1000). También puede fijarse a mano.

**Paso 3 — Dar Zarrouk: elegir la fuente del modelo de bloques.** Tanto Zohdy como el
multicapa se reducen a 2…k_max capas por Dar Zarrouk (segmentación óptima por programación
dinámica; variante clásica h = √(TS), ρ = √(T/S), y variante con espesor conservado).
Cada reducción se ajusta rápidamente y se comparan por BIC: la fuente que mejor ajusta
(Zohdy o multicapa) da los arranques principales; la otra queda como arranque
alternativo.

**Paso 4 — Inversión de bloques con Optuna.** Optuna (TPE) explora los hiperparámetros
viables:
- **k** (número de capas, 2…k_max, o fijo);
- **variante del arranque:** Dar Zarrouk clásico, Dar Zarrouk con espesor conservado,
  la otra fuente, división del mejor modelo de k−1 capas, o 10 arranques exploratorios
  por hipercubo latino (interfaces en log z; ρ = media del multicapa en cada capa);
- **escala de profundidades** (0.6–1.6, logarítmica) y **contraste γ** (0.5–1.5) aplicados
  al arranque.

Primero se encolan todos los arranques base de cada k; luego vienen las pruebas de
búsqueda (40 por omisión). Cada prueba es un Levenberg–Marquardt acotado (SciPy,
jacobiano analítico) con **espesor mínimo obligatorio**. Los 5 mejores de cada k se
refinan hasta converger. k se elige por BIC (o AIC); con ΔIC < 2 gana el más simple.
Entre ajustes equivalentes (Δφ ≤ 0.25) gana el de **mínima estructura**, Σ(Δ ln ρ)². El
elegido se pule con la inversión de bloques de pyGIMLi si mejora el ajuste, y se funden
capas contiguas casi iguales.

**Modo de bloques por profundidad (alternativo).** No usa Zohdy ni el multicapa para
arrancar: el intervalo de profundidades de investigación de Edwards [z_e mín, z_e máx]
se divide en las k capas que indica el usuario, con interfaces **log-uniformes**. La ρ inicial de cada capa es la
media (geométrica) de las ρa cuyos z_e caen en ese rango. Ese modelo se invierte con el
mismo Optuna + LM del paso 4 (k fijo).

**Espesor mínimo.** El modelo de bloques no acepta capas que el SEV no puede resolver:
- la primera capa mide al menos la profundidad de investigación de la apertura más
  corta;
- cada capa siguiente mide al menos una fracción (25 % por omisión, ajustable) de la
  profundidad de su techo.

La restricción va dentro del ajuste, no como filtro posterior. Las interfaces se
parametrizan como ln z₁ = ln h_min + w₁ y ln zᵢ = ln zᵢ₋₁ + ln(1 + f) + wᵢ, con w ≥ 0,
y el jacobiano se transforma analíticamente.

Las incertidumbres ±1σ salen de la covarianza linealizada e indican si la capa resuelve
h y ρ, solo S, solo T o nada.

**Colores en los gráficos.** Zohdy: amarillo, línea delgada. Multicapa: naranja, ancho
normal. Modelo invertido (bloques): azul (0, 0, 255), línea gruesa.

**Parámetros de campaña.** El botón "Guardar parámetros de la campaña" escribe
`parametros_campana.json` en la carpeta del proyecto (viaja dentro del `.isev`) y se
carga al abrir el proyecto. También se puede exportar/importar el JSON para usarlo en
otra campaña. "Automático" restablece los valores recomendados.

**Corrosividad.** Se estima a partir del modelo invertido (bloques), por capa, con las
clases de resistividad de NACE/Roberge (Ω·m): < 5 muy alta, 5–10 alta, 10–20 moderada,
20–100 baja, > 100 muy baja (umbrales editables). Se grafica debajo del último gráfico
de la inversión: el modelo sobre las bandas de clase, la clase de cada capa y la
resistividad longitudinal media ρL(0–z) con la profundidad de evaluación.

**DOI.** Índice de Oldenburg y Li con pesos en profundidad física (suavidad ∝ 1/Δz,
atracción a la referencia ∝ espesor). Sin esa ponderación, R sale ≈ 0 a cualquier
profundidad.

**Parámetros acumulados (interpretación).** Para el multicapa y el modelo de bloques se
calculan, por capa y acumulados desde la superficie:
- S, T, ρL = H/S, ρT = T/H, ρ promedio = √(ρL·ρT) y anisotropía λ;
- el contraste de cada capa contra el promedio de lo que tiene encima (≥ 2: aumento
  fuerte; ≤ 0.5: disminución fuerte).

La pantalla de inversión grafica T(z), S(z) y ρ promedio(z) = √(T/S), que es un
promedio corrido de 0 a z (sus picos y valles caen en las interfaces). Agrega dos paneles:
la pseudo-anisotropía λ(z) = √(T·S)/z (1 = homogéneo; crece con cualquier contraste) y
el contraste de cada capa contra el promedio de lo que tiene encima, con signo: más
resistiva a la derecha, más conductora a la izquierda. Son curvas independientes
de la discretización: donde el multicapa y el simplificado coinciden, son equivalentes
para los datos.

El multicapa se agrupa en capas equivalentes: se unen celdas contiguas mientras la
anisotropía del grupo λ = √(T·S)/H no pase de 1.02. Un grupo así es una sola capa con
ρ = √(T/S). La T de cada celda por separado no sirve para decidirlo, porque en la malla
logarítmica las celdas profundas son más gruesas.

**IP.** Relación de Seigel linealizada ma ≈ (∂ln ρa/∂ln ρ)·m con el jacobiano del modelo de
resistividad, mínimos cuadrados acotados (m ≥ 0) y λ por discrepancia. El ajuste se
verifica con la relación de Seigel no lineal. En el simplificado (pocas capas), la solución
lineal se refina con la relación de Seigel **no lineal** por mínimos cuadrados acotados.
Con curvas exactas del modelo B recupera 10/6/20/8/25/7 mV/V.

**Perfiles.** Proyección de los sondeos sobre una polilínea (vértices editables, ruta
automática por PCA), pseudosecciones observada, preprocesada y calculada, secciones
interpoladas (multicapa, simplificado, IP y modelo de referencia, todas con la misma escala
de color) e **inversión lateralmente restringida**: malla
2D de pyGIMLi (capa × sondeo) con jacobiano por bloques analítico y acoplamiento lateral
ajustable.

## Validación con datos sintéticos

`ejemplos/generar_sinteticos.py` genera 6 SEV (Schlumberger y Wenner alternados, dos
direcciones, IP) con AB/2 hasta 500 m, sobre el modelo A (por defecto) o el B
(`--modelo B --carpeta ejemplos/modelo_B`).

Métricas de error usadas en toda la validación: **χ²** (con el error de cada dato),
**MAPE** y **validación cruzada** dejando uno fuera (CV-MAPE: se quita un dato, se
reajusta el modelo de bloques y se predice el dato quitado).

### Modelo B (contrastes moderados), sondeo S1 Schlumberger

| Profundidad | ρ real | Curva exacta | 3 % de ruido, 2 direcciones |
|---|---|---|---|
| 0–5 m | 50 | 0–5 m · 50 | 0–4.7 m · 49.6 |
| 5–15 m | 100 | 5–15 m · 100 | 4.7–15 m · 92.8 |
| 15–30 m | 20 | 15–30 m · 20 | 15–170 m · 28.6 (agrupa 15–80 m) |
| 30–50 m | 60 | 30–50 m · 60 | |
| 50–80 m | 10 | 50–80 m · 10 | |
| > 80 m | 75 | > 80 m · 75 | > 170 m · 98 |
| χ² / MAPE / CV-MAPE | | 0.00 / 0.00 % / 0.00 % | 0.55 / 1.55 % / 2.10 % |

Con 3 % de ruido el error estimado por dato es ~2.1 % (3 %/√2) y las capas de 15–80 m
no se distinguen de una sola capa de 28.6 Ω·m: el modelo de 4 capas ajusta igual que el
verdadero (equivalencia).

### Modelo A (contrastes fuertes)

El modelo es:

| Profundidad | ρ (Ω·m) | m (mV/V) |
|---|---|---|
| 0–2 m | 10 | 25 |
| 2–10 m | 60 | 8 |
| 10–20 m | 10 | 25 |
| 20–40 m | 300 | 3 |
| 40–70 m | 10 | 25 |
| 70–90 m | 300 | 3 |
| > 90 m | 60 | 8 |

**Curva exacta (S1).** El preprocesamiento no toca ningún dato. El BIC elige 6 capas:
2 / 10.5 / 15.2 / 27.2 / 58.9 m con 10 / 59.7 / 4.7 / 473 / 14.9 / 64.7 Ω·m
(χ² = 0.04, MAPE = 0.01 %, CV-MAPE = 0.03 %).
- Las dos primeras interfaces salen exactas.
- El conductor de 10–20 m conserva su conductancia (S = 4.7/4.7 ≈ 1.0 S, real 1.0 S) y el
  resistivo de 20–40 m su resistencia transversal (T = 12.1 × 473 ≈ 5700 Ω·m², real 6000):
  equivalencia S y T.
- El candidato de 7 capas reproduce casi exactamente el modelo verdadero
  (10 / 60 / 10 / 305 / 9.6 / 147 / 60 Ω·m), pero con datos exactos y piso de error de
  0.1 % los dos tienen χ² < 1 y el BIC prefiere el más simple. Se puede revisar con
  "Capas fijas = 7".

**Con `--ruido 0.10 --aberrantes --desplazamiento`.**
- Detecta las lecturas aberrantes.
- El ruido estimado es de 7–11 % (real: 10 %).
- Con ese ruido, el modelo verdadero no ajusta mejor que uno de 3 capas (principio de
  equivalencia).

### Sondeo tipo IPI2Win (5 capas, 3 % de ruido, solo ResApp)

Real: 0.4 m · 46.7 / 0.6 m · 12.1 / 2.7 m · 787 / 7.9 m · 14.9 / 4812 Ω·m. Invertido:
0.4 · 43.9 / 0.75 · 14.6 / 1.6 · 1290 / 4.7 · 11.9 / 323 Ω·m (χ² = 1.18, MAPE = 0.91 %,
CV-MAPE = 2.3 %). El resistivo delgado conserva su T (2080 contra 2125 Ω·m²). Ninguna capa
es más delgada que el mínimo resoluble.

`ejemplos/construir_proyecto_demo.py` genera `Sinteticos_demo.isev` con todo procesado,
el modelo verdadero como referencia y el perfil P1 con LCI.

### Regularización del multicapa

El estándar es Occam suave (L2): el multicapa es un modelo continuo y conservador, que
sirve de referencia y de fuente para Dar Zarrouk. Con suavidad L2 los saltos se reparten
en varias celdas; por eso el modelo de bloques ya no sale de reducir solo el multicapa:
el paso 3 compara su reducción con la de Zohdy y el paso 4 explora con Optuna alrededor
de ambas. La regularización robusta (L1 en los datos) conviene con datos ruidosos o con
saltos de empalme; la nítida (gradiente mínimo) sigue disponible.

Además:
- Se desactivó el criterio de pyGIMLi que detiene la inversión al llegar a χ² < 1.
- El error de cada dato es el ruido medido en los datos (con piso de 0.1 %), no un 3 %
  fijo.

## Estructura

```
src/   lectura_sev · geometria_sev · forward_sev · filtro_hankel_sev · procesamiento_sev
       zohdy_sev · inversion_sev · parametros_sev · perfil_sev · graficos_sev
       pipeline_sev · almacenamiento_sev · paquete_sev · exportacion_sev
       licencias_tre · escalas_color_tre · generar_escala_tre   (compartidos con 2D)
app/   app.py · acciones_app · estado_app · auth_app · marca_sev · test_app
       pages/ 1 Proyecto y datos · 2 Preprocesamiento · 3 Inversión
              4 Parámetros y unidades · 5 Perfiles 2D · 6 Exportar datos
ejemplos/  generador de sintéticos, datos, coordenadas y modelo verdadero
```

Regla de estilo heredada: sin `elif`.
