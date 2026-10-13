# TLALLI-RES 1D — Sondeos Eléctricos Verticales e IP

Aplicación Streamlit para el flujo completo de SEV/IP, con la misma arquitectura que
TlalliRes2D (autenticación y licencias, proyecto `.isev` descargable, Parquet + SQLite +
DuckDB, lógica fuera de la interfaz y pruebas sin navegador):

```
lectura y QA/QC → empalmes → promedio de direcciones → preprocesamiento
→ Zohdy → multicapa (pyGIMLi, λ por Optuna) → DOI
→ reducción de Dar Zarrouk + BIC → simplificado (pyGIMLi) → IP (Seigel)
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
cada punto es σ/√n. Con dos direcciones, la desviación punto a punto es demasiado
inestable para usarla como peso.

**Empalmes de MN.** Se mide el salto sobre los residuales de un ajuste físico con la
geometría exacta de cada lectura. Así no se borra la parte física del salto (MN no
despreciable frente a AB). Solo se corrige si el salto es significativo (3σ) y si los
traslapes son coherentes entre sí.

**Detección de anómalos.** Se usa como referencia la curva de una inversión suave robusta
(IRLS ≈ L1): mide lo que ningún medio 1D explica. Un dato es anómalo si cumple:
- z robusto > 3 y error > 10 %, o
- dispersión entre direcciones > 20 % y > 2.5σ, o
- pendiente ascendente > 45° (SCH/WE).

Si una dirección discrepa y otra no, se descarta la discrepante y se rehace el promedio.
Tratamientos disponibles: suavizar (por defecto), ponderar o eliminar.

**Zohdy.** A cada dato se le asigna la base de su capa en z = AB/n. Se prueban n de 5 a 8
en pasos de 0.125 (configurable) y cada n se evalúa con Zohdy completo, es decir, con
las resistividades ya corregidas. Gana el n de menor RMS.
- La regla empírica AB/6 coincide con la profundidad de Edwards (AB/5.3 en Schlumberger,
  AB/5.8 en Wenner).
- En los sintéticos el óptimo cae entre AB/5.75 y AB/7.1 en el modelo A. En el modelo B
  cae en AB/4.25–4.75, fuera del rango; la app avisa cuando el óptimo queda en el borde.
- En arreglos sin AB definido se usa una longitud equivalente de 5.3·z_e.
- La versión anterior (factor sobre Edwards, elegido antes de corregir ρ) sigue disponible.

**Multicapa.** Profundidades fijas; solo se invierte ρ.

La malla por defecto se construye sobre Zohdy. El factor de profundidad f que mejor
ajusta la curva calibra la escala de profundidades con los datos.
- La **profundidad máxima viable** es f·máx(z_e). En los sintéticos da 164–228 m, contra
  433–478 m de la malla logarítmica anterior (2.5·máx z_e).
- Las celdas son las interfaces de Zohdy subdivididas en log z (~50 celdas), así que el
  modelo de Zohdy entra exacto como modelo inicial.
- La DOI y la búsqueda de interfaces del simplificado se limitan a la profundidad viable.
- La malla logarítmica sigue disponible como opción.

El multicapa se invierte desde ese modelo inicial. pyGIMLi resuelve con regularización nítida (soporte de gradiente
mínimo) o suave (Occam L2).
- Nítida: la reponderación se itera **hasta converger** (hasta 15 vueltas).
- λ se recorre por continuación de 100 a 0.3, y cada λ parte del modelo del anterior.
- λ se limita a 100. Por encima, en la parte profunda (poco sensible) manda la
  regularización, los contrastes se suavizan y se desplazan hacia abajo.
- Suave: λ se elige con Optuna, minimizando |ln χ²| + 0.1·rugosidad + una penalización
  por residuales correlacionados.

Zohdy sirve solo de modelo inicial: con un arranque homogéneo, el multicapa converge al
mismo resultado.

Invertir las capas de Zohdy (una por dato) con espesores y ρ libres no es viable: hay
2N−1 incógnitas para N datos.
- Sin regularización, el modelo acumula capas de menos de 0.5 m y resistividades
  extremas. Con 3 % de ruido ajusta el ruido: salen valores de 3 a 43 000 Ω·m.
- Con Marquardt amortiguado detenido en χ² ≤ 1, no se aleja de Zohdy.

pyGIMLi tampoco ofrece esa opción. Su inversión multicapa (`VESRhoModelling`) fija los
espesores e invierte solo ρ, igual que aquí. Su inversión por bloques (`VESModelling`)
libera h y ρ, pero en pocas capas: es el simplificado. La pantalla de inversión muestra el modelo de Zohdy propio (una capa
por dato) y la curva que genera.

**DOI.** Índice de Oldenburg y Li con pesos en profundidad física (suavidad ∝ 1/Δz,
atracción a la referencia ∝ espesor). Sin esa ponderación, R sale ≈ 0 a cualquier
profundidad.

**Simplificado.** Para k = 2…kmax se prueban cuatro familias de arranques:
- el modelo de Zohdy reducido a k capas por Dar Zarrouk;
- reducción de Dar Zarrouk del multicapa (segmentación óptima por programación dinámica;
  clásica h = √TS, ρ = √(T/S), y con espesor conservado);
- división del mejor modelo de k−1 capas (incluida la división neutra, que garantiza
  χ² no creciente);
- **búsqueda global**: hasta 24 posiciones de interfaces por hipercubo latino (log z, entre
  la menor y 1.5 veces la mayor profundidad de investigación). En cada capa, ρ inicial es
  la media del multicapa.

Cada arranque se ajusta con Levenberg–Marquardt acotado (SciPy, jacobiano analítico).
Se hace en dos etapas: todos los arranques con 8 evaluaciones, y los 6 mejores hasta
converger. Eso reduce el tiempo a la mitad (30–45 s por sondeo) sin cambiar el
resultado. El elegido se pule con la inversión de bloques de pyGIMLi.

**Espesor mínimo.** El simplificado no acepta capas que el SEV no puede resolver:
- la primera capa mide al menos la profundidad de investigación de la apertura más
  corta;
- cada capa siguiente mide al menos una fracción (25 % por omisión, ajustable) de la
  profundidad de su techo.

La restricción va dentro del ajuste, no como filtro posterior. Las interfaces se
parametrizan como ln z₁ = ln h_min + w₁ y ln zᵢ = ln zᵢ₋₁ + ln(1 + f) + wᵢ, con w ≥ 0,
y el jacobiano se transforma analíticamente. El pulido con pyGIMLi y la fusión de capas
solo se aceptan si cumplen la restricción.

**Equivalencia.** Entre los ajustes que explican los datos igual de bien (Δφ ≤ 0.25),
se queda el más parecido al multicapa. La comparación se pondera por la sensibilidad de
los datos a cada profundidad, para que la parte profunda (donde el multicapa lo decide
la regularización) no pese. Así se evita, por ejemplo, un
conductor delgado de 3 Ω·m cuando uno de 10 Ω·m y el triple de espesor da la misma
curva.

Se elige k por BIC; con ΔBIC < 2 gana el más simple. También se puede fijar k a mano
("Capas fijas") para revisar los candidatos de la tabla de selección. Las incertidumbres
±1σ salen de la covarianza linealizada e indican si la capa resuelve h y ρ, solo S,
solo T o nada.

**Parámetros acumulados (interpretación).** Para el multicapa y el simplificado se
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

### Modelo B (contrastes moderados)

| Profundidad | ρ real | Schlumberger (S1) | Wenner (S2) |
|---|---|---|---|
| 0–5 m | 50 | 0–5 m · 50 | 0–5 m · 50 |
| 5–15 m | 100 | 5–15 m · 100 | 5–15 m · 100 |
| 15–30 m | 20 | 15–30.2 m · 20.1 | 15–30.2 m · 20 |
| 30–50 m | 60 | 30.2–49.6 m · 61.2 | 30.2–48.9 m · 62.7 |
| 50–80 m | 10 | 49.6–80.1 m · 10.1 | 48.9–81.2 m · 10.7 |
| > 80 m | 75 | > 80.1 m · 75 | > 81.2 m · 75 |

Curvas exactas. El BIC elige 6 capas en los dos arreglos. El multicapa (λ = 100)
reproduce las capas hasta 50 m; el conductor de 50–80 m sale como ~13 Ω·m hasta 95–100 m
(misma conductancia), por equivalencia S.

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

**Curvas exactas (por defecto).**
- El preprocesamiento no toca ningún dato.
- El simplificado de Schlumberger (6 capas) recupera las interfaces de 2, 10.2, 18.6 y
  34 m, con 10, 60, 8.5 y 369 Ω·m. El resistivo conserva su T (5650 contra 6000 Ω·m²).
- El conductor de 40–70 m sale como 17 Ω·m hasta 70 m.
- La capa resistiva de 70–90 m está bajo un conductor de 30 m (S = 3 siemens) y apenas
  cambia la curva. El candidato de 7 capas la muestra (79–116 m, 175 Ω·m), pero su
  mejora de ajuste no basta para que el BIC lo elija. Se puede revisar con
  "Capas fijas = 7".

**Con `--ruido 0.10 --aberrantes --desplazamiento`.**
- Detecta 18 de 18 lecturas aberrantes.
- El ruido estimado es de 7–11 % (real: 10 %).
- El BIC elige 3 capas. Con ese ruido, el modelo verdadero no ajusta mejor que el de
  3 capas (principio de equivalencia).

`ejemplos/construir_proyecto_demo.py` genera `Sinteticos_demo.isev` con todo procesado,
el modelo verdadero como referencia y el perfil P1 con LCI.

### Por qué el multicapa usa regularización nítida

Con suavidad L2 (Occam clásico), cada salto de resistividad se reparte en varias capas.
Al reducir ese modelo, las interfaces quedan en lugares equivocados: con curvas exactas,
la capa de 300 Ω·m salía como 150 Ω·m de 25 a 50 m. Con soporte de gradiente mínimo, el
multicapa sale por bloques (10 / 61 / 10 / 274 Ω·m con interfaces en 2, 10, 20 y 39 m) y
Dar Zarrouk agrupa las capas reales.

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
