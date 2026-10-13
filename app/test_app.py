"""
test_app.py
==================================================================
Pruebas sin navegador de TLALLI-RES 1D: física, procesamiento,
inversión, almacenamiento y acciones de los botones.

Ejecutar::

    python app/test_app.py            # todas (~1–2 min)
    python app/test_app.py rapido     # omite las inversiones completas

Cada prueba es una función ``prueba_*``; el resumen final indica cuáles
fallaron.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import _ruta_src  # noqa: F401

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ejemplos"))

import acciones_app as acc  # noqa: E402
import almacenamiento_sev as alm  # noqa: E402
import forward_sev as fw  # noqa: E402
import geometria_sev as geo  # noqa: E402
import inversion_sev as inv  # noqa: E402
import lectura_sev as lec  # noqa: E402
import parametros_sev as pa  # noqa: E402
import perfil_sev as pf  # noqa: E402
import pipeline_sev as pl  # noqa: E402
import procesamiento_sev as pr  # noqa: E402

RAPIDO = "rapido" in sys.argv


def _sch(ab2, mn2):
    return fw.GeometriaSondeo(ab2 - mn2, ab2 + mn2, ab2 + mn2, ab2 - mn2)


def _tabla_sintetica(thk, rho, ruido=0.02, semilla=0, direcciones=("NS", "EW"), ip=None):
    rng = np.random.default_rng(semilla)
    ab2 = np.round(np.logspace(0, 2.5, 26), 2)
    mn2 = np.where(ab2 <= 12, 0.5, 5.0)
    extra = ab2[(ab2 > 8) & (ab2 <= 12)]
    ab2 = np.concatenate([ab2, extra])
    mn2 = np.concatenate([mn2, np.full(len(extra), 5.0)])
    g = _sch(ab2, mn2)
    verdad = fw.respuesta(g, thk, rho)
    ip_v = fw.cargabilidad_aparente(g, thk, rho, ip) if ip is not None else np.zeros(len(ab2))
    filas = []
    for d in direcciones:
        r = verdad * np.exp(rng.normal(0, ruido, len(ab2)))
        for i in range(len(ab2)):
            filas.append(dict(Sondeo="T1", ID=i + 1, a=np.nan, AB2=ab2[i], MN2=mn2[i], K=g.k[i], I=100.0,
                              V=r[i] * 100 / g.k[i], R=r[i] / g.k[i], ResApp=r[i], Direccion=d, IP=ip_v[i]))
    df = pd.DataFrame(filas)
    df["tiene_ip"] = ip is not None
    return df


# ----------------------------------------------------------------------
# Física
# ----------------------------------------------------------------------


def prueba_forward_analitico():
    assert fw.verificar_forward() < 1e-5


def prueba_motores_coinciden():
    rng = np.random.default_rng(0)
    thk = np.diff(np.r_[0, np.logspace(-0.3, 2, 15)])
    rho = np.exp(rng.normal(4, 1.5, 16))
    g = _sch(np.logspace(0, 2.5, 20), np.full(20, 0.5))
    a = fw.respuesta(g, thk, rho, motor="numpy")
    b = fw.respuesta(g, thk, rho, motor="pygimli")
    assert np.max(np.abs(a / b - 1)) < 1e-4


def prueba_jacobiano_analitico():
    rng = np.random.default_rng(1)
    thk = np.diff(np.r_[0, np.logspace(-0.3, 2, 10)])
    rho = np.exp(rng.normal(4, 1, 11))
    g = _sch(np.logspace(0, 2.5, 20), np.full(20, 0.5))
    ja = fw.jacobiano_log(g, thk, rho)
    fw.fijar_motor("pygimli")
    try:
        jn = fw.jacobiano_log(g, thk, rho)
    finally:
        fw.fijar_motor("numpy")
    assert np.max(np.abs(ja - jn)) < 1e-5


def prueba_seigel_lineal():
    g = _sch(np.logspace(0, 2.5, 20), np.full(20, 0.5))
    thk, rho, m = [3, 12], [50, 10, 300], np.array([0, 30, 0])
    lineal = fw.jacobiano_log(g, thk, rho, incluir_espesores=False) @ m
    exacta = fw.cargabilidad_aparente(g, thk, rho, m)
    assert np.max(np.abs(lineal - exacta)) < 1.0


def prueba_profundidad_edwards():
    df = pd.DataFrame({"a": [1.0], "AB2": [1.5], "MN2": [0.5], "K": [2 * np.pi]})
    z = geo.geometria(df, "WE", "a")["z_mediana"].iloc[0]
    assert abs(z - 0.519) < 0.002
    df = pd.DataFrame({"a": [1.0], "AB2": [2.0], "MN2": [0.5], "K": [6 * np.pi]})
    z = geo.geometria(df, "DD", "centros")["z_mediana"].iloc[0]
    assert abs(z - 0.416) < 0.002


def prueba_deteccion_arreglos():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    casos = {
        "WE": pd.DataFrame({"a": a, "AB2": 1.5 * a, "MN2": a / 2, "K": 2 * np.pi * a}),
        "DD": pd.DataFrame({"a": [1.0] * 4, "AB2": a + 1, "MN2": [0.5] * 4,
                            "K": np.pi * a * (a + 1) * (a + 2)}),
        "PD": pd.DataFrame({"a": [1.0] * 4, "AB2": a + 0.5, "MN2": [0.5] * 4, "K": 2 * np.pi * a * (a + 1)}),
        "PP": pd.DataFrame({"a": [np.nan] * 4, "AB2": a, "MN2": [np.nan] * 4, "K": 2 * np.pi * a}),
    }
    for esperado, df in casos.items():
        d = geo.detectar_arreglo(df)
        assert d.arreglo == esperado and d.confirmada, (esperado, d)


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------


def prueba_lectura_formatos():
    texto_tab = "Sondeo\tID\ta\tAB/2\tMN/2\tK\tI\tV\tR\tResApp\tDirección\tIP\n" + "\n".join(
        f"S1\t{i}\t\t{x}\t0.5\t{np.pi * (x * x - 0.25) / 1:.4f}\t100\t{50.0:.3f}\t0.5\t{x * 3:.3f}\tNS\t5"
        for i, x in enumerate([1, 2, 3, 5, 8, 13], start=1))
    df = lec.leer_sev(texto_tab.encode())
    assert len(df) == 6 and df["Direccion"].iloc[0] == "NS" and df["tiene_ip"].all()
    texto_coma = texto_tab.replace(".", ",").replace("\t", ";")
    df2 = lec.leer_sev(texto_coma.encode("latin-1"))
    assert np.allclose(df2["AB2"], df["AB2"])
    sin_encabezado = "\n".join(texto_tab.splitlines()[1:])
    df3 = lec.leer_sev(sin_encabezado.encode())
    assert len(df3) == 6 and np.allclose(df3["K"], df["K"])


def prueba_lectura_sin_ip_ni_direccion():
    texto = "Sondeo,AB2,MN2,ResApp\nA,1,0.5,10\nA,2,0.5,12\nA,3,0.5,14\nA,5,0.5,15\nA,8,0.5,16\n"
    df = lec.leer_sev(texto.encode())
    assert (df["IP"] == 0).all() and (df["Direccion"] == "U").all() and not df["tiene_ip"].any()


def prueba_correccion_unidades():
    df = _tabla_sintetica([3, 10], [50, 10, 200], direcciones=("NS",))
    df["V"] = df["V"] * 1000  # V en otra unidad
    tabla, _, _, adv = pr.preparar_crudo(df)
    assert any("10^" in a for a in adv)
    assert np.allclose(tabla["rhoa"], tabla["ResApp"], rtol=1e-6)


# ----------------------------------------------------------------------
# Procesamiento
# ----------------------------------------------------------------------


def prueba_promedio_geometrico():
    df = _tabla_sintetica([3, 10], [50, 10, 200], ruido=0.0)
    df.loc[df["Direccion"] == "EW", "ResApp"] *= 1.21
    df.loc[df["Direccion"] == "EW", "V"] *= 1.21
    tabla, _, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla)
    prom, met, _ = pr.promediar_direcciones(tabla)
    ns = tabla[tabla["Direccion"] == "NS"].sort_values(["AB2", "MN2"])["rhoa"].to_numpy()
    assert np.allclose(prom["rhoa_obs"].to_numpy(), ns * 1.1, rtol=1e-6)
    assert abs(met["anisotropia_mediana"] - 1.21) < 1e-6


def prueba_detecta_aberrantes():
    df = _tabla_sintetica([3, 10], [50, 10, 200], ruido=0.03, semilla=4)
    malos = [5, 15]
    for i in malos:
        fila = df.index[(df["Direccion"] == "NS")][i]
        df.loc[fila, ["ResApp", "V"]] *= 2.5
    tabla, det, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    prom, _, _ = pr.promediar_direcciones(tabla)
    proc, met, _ = pr.preprocesar(prom, arreglo=det.arreglo)
    detectados = set(np.round(proc.loc[proc["anomalo"], "AB2"], 2))
    esperados = set(np.round(df[df["Direccion"] == "NS"]["AB2"].iloc[malos], 2))
    assert esperados <= detectados, (esperados, detectados)
    assert len(detectados - esperados) <= 1


def prueba_curva_limpia_no_se_toca():
    df = _tabla_sintetica([3, 10], [50, 10, 200], ruido=0.01, semilla=2)
    tabla, det, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    prom, _, _ = pr.promediar_direcciones(tabla)
    proc, met, _ = pr.preprocesar(prom, arreglo=det.arreglo)
    assert met["n_anomalos"] == 0 and np.allclose(proc["rhoa_proc"], proc["rhoa_obs"])


def prueba_empalme_significativo():
    df = _tabla_sintetica([3, 10], [50, 10, 200], ruido=0.005, direcciones=("NS",))
    tramo = df["MN2"] == 5.0
    df.loc[tramo, ["ResApp", "V"]] *= 1.25
    tabla, det, _, _ = pr.preparar_crudo(df)
    _, informe = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    assert informe and informe[0]["aplicado"] and abs(informe[0]["factor"] - 0.8) < 0.1


# ----------------------------------------------------------------------
# Inversión
# ----------------------------------------------------------------------


def prueba_dar_zarrouk_conserva_s_t():
    h = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    rho = np.array([20.0, 5.0, 8.0, 100.0, 150.0, 40.0])
    nh, nr, tramos = inv.reducir_dar_zarrouk(h, rho, 3)
    for indice, (a, b) in enumerate(tramos):
        if b == len(rho):
            continue
        else:
            pass
        original = inv.parametros_dar_zarrouk(h[a:b], rho[a:b])
        reducido = inv.parametros_dar_zarrouk(nh[indice:indice + 1], nr[indice:indice + 1])
        assert abs(reducido["S"] / original["S"] - 1) < 1e-9
        assert abs(reducido["T"] / original["T"] - 1) < 1e-9


def prueba_segmentacion_optima():
    valores = np.array([1, 1, 1, 3, 3, 2, 2, 2.0])
    tramos = inv.segmentar(valores, np.ones(8), 3)
    assert tramos == [(0, 3), (3, 5), (5, 8)]


def prueba_inversion_tres_capas():
    if RAPIDO:
        return
    else:
        pass
    df = _tabla_sintetica([3, 10], [50, 10, 200], ruido=0.02, semilla=3, ip=[0, 30, 2])
    tabla, det, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    prom, _, _ = pr.promediar_direcciones(tabla)
    proc, _, _ = pr.preprocesar(prom, arreglo=det.arreglo)
    res = inv.invertir_sondeo(proc, inv.ConfigInversion(n_trials=8))
    s = res.simplificado
    assert len(s) == 3, s
    assert abs(s["rho"].iloc[0] / 50 - 1) < 0.15
    assert abs(s["S"].iloc[1] / (10 / 10) - 1) < 0.25          # conductancia de la capa 2
    assert 0.5 < res.metricas["multicapa_chi2"] < 2.0
    assert s["m_ip"].iloc[1] > 15                                # capa cargable detectada
    dz = pa.dar_zarrouk(res.multicapa, 13.0)
    assert dz["S"] > 0 and dz["lambda"] >= 1.0


def prueba_sin_ruido_detecta_capas():
    """Curva exacta del modelo de 7 capas: el simplificado debe encontrar
    las interfaces resolubles (2, 10, 20 y 40 m) y sus resistividades."""
    if RAPIDO:
        return
    else:
        pass
    import generar_sinteticos as gs

    datos, _, _ = gs.generar()
    df = datos[datos["Sondeo"] == "S1"].rename(columns={"AB/2": "AB2", "MN/2": "MN2"})
    df["tiene_ip"] = True
    tabla, det, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    prom, _, _ = pr.promediar_direcciones(tabla)
    proc, met, _ = pr.preprocesar(prom, arreglo=det.arreglo)
    assert met["n_anomalos"] == 0
    s = inv.invertir_sondeo(proc).simplificado
    assert 5 <= len(s) <= 8, s
    interfaces = s["z_base"].to_numpy()[:3]
    assert np.allclose(interfaces, [2, 10, 20], rtol=0.15), interfaces
    # Resistivo de 20–40 m: los datos fijan su resistencia transversal
    # (T = 20 m × 300 Ω·m = 6000 Ω·m²), no h y ρ por separado.
    assert abs(s["espesor"].iloc[3] * s["rho"].iloc[3] / 6000.0 - 1.0) < 0.15, s
    assert np.allclose(s["rho"].to_numpy()[:2], [10, 60], rtol=0.2), s["rho"]
    # Conductor de 10–20 m: lo que los datos fijan es su conductancia
    # (S = 10 m / 10 Ω·m = 1 siemens), no h y ρ por separado.
    assert abs(s["espesor"].iloc[2] / s["rho"].iloc[2] - 1.0) < 0.15, s
    assert s["rho"].iloc[3] > 150


def prueba_sin_ruido_modelo_b():
    """Modelo B (contrastes moderados), curva exacta: Schlumberger debe
    recuperar las 6 capas con interfaces 5, 15, 30, 50 y 80 m."""
    if RAPIDO:
        return
    else:
        pass
    import generar_sinteticos as gs

    datos, _, _ = gs.generar(modelo="B")
    df = datos[datos["Sondeo"] == "S1"].rename(columns={"AB/2": "AB2", "MN/2": "MN2"})
    df["tiene_ip"] = True
    tabla, det, _, _ = pr.preparar_crudo(df)
    tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
    prom, _, _ = pr.promediar_direcciones(tabla)
    proc, _, _ = pr.preprocesar(prom, arreglo=det.arreglo)
    s = inv.invertir_sondeo(proc).simplificado
    assert len(s) == 6, s
    assert np.allclose(s["z_base"].to_numpy()[:5], [5, 15, 30, 50, 80], rtol=0.1), s
    rho = s["rho"].to_numpy()
    assert np.allclose(rho[[0, 1, 2, 3, 5]], [50, 100, 20, 60, 75], rtol=0.15), s
    # Conductor de 50–80 m: los datos fijan su conductancia (30 m / 10 Ω·m = 3 S).
    assert abs(s["espesor"].iloc[4] / s["rho"].iloc[4] / 3.0 - 1.0) < 0.1, s
    assert np.allclose(s["m_ip"].to_numpy(), [10, 6, 20, 8, 25, 7], atol=2.0), s


def prueba_columnas_intercambiadas():
    """Archivo real de usuario: la columna K trae I/V, la columna R trae el
    factor geométrico y la columna V la resistencia ΔV/I. La app debe
    detectarlo, usar la K de la posición real de los electrodos y
    reproducir el ResApp del archivo."""
    d = pd.DataFrame(dict(
        Sondeo="U", Direccion="E", ID=range(1, 9), a=[0.6, 1, 1.4, 2, 3, 4, 5, 7],
        AB2=[0.8, 1.5, 2.2, 3, 4.5, 6, 7.5, 10.5], MN2=[0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3.5],
        K=[42.81, 42.41, 45.78, 47.74, 51.59, 52.66, 49.9, 48.88],
        I=[317.28, 199.71, 192.86, 187.72, 183.13, 175.38, 158.11, 135.58],
        V=[7.4114, 4.709, 4.2128, 3.9321, 3.5497, 3.3304, 3.1685, 2.7737],
        R=[4.7124, 6.2832, 8.2467, 12.5664, 18.8496, 25.1327, 31.4159, 43.9823],
        ResApp=[34.9252, 29.5877, 34.7413, 49.4126, 66.9106, 83.7026, 99.5425, 121.9951], IP=0.0))
    d["tiene_ip"] = False
    tabla, det, _, adv = pr.preparar_crudo(d)
    assert det.columna_k == "R", det
    assert np.allclose(tabla["K_geom"], d["R"], rtol=1e-3), tabla["K_geom"]
    assert np.allclose(tabla["rhoa"], d["ResApp"], rtol=1e-3), tabla["rhoa"]
    assert any("la columna V contiene la resistencia" in a for a in adv), adv


def prueba_lci():
    if RAPIDO:
        return
    else:
        pass
    datos, iniciales, coords = {}, {}, []
    for i, h2 in enumerate([6, 10, 14]):
        df = _tabla_sintetica([2, h2], [50, 8, 150], ruido=0.03, semilla=10 + i, direcciones=("U",))
        df["Sondeo"] = f"L{i}"
        tabla, det, _, _ = pr.preparar_crudo(df)
        tabla, _ = pr.corregir_empalmes(tabla, arreglo=det.arreglo)
        prom, _, _ = pr.promediar_direcciones(tabla)
        proc, _, _ = pr.preprocesar(prom, arreglo=det.arreglo)
        datos[f"L{i}"] = proc
        iniciales[f"L{i}"] = inv.invertir_sondeo(proc, inv.ConfigInversion(metodo_lambda="manual", calcular_doi=False,
                                                                           k_max=3)).multicapa
        coords.append({"Sondeo": f"L{i}", "X": 100.0 * i, "Y": 5.0, "Z": 0.0})
    proy = pf.proyectar(pd.DataFrame(coords), [[0, 0], [200, 0]], buffer=20)
    res = pf.invertir_lci(proy, datos, iniciales, lam=20)
    assert res["chi2"] < 3.0 and len(res["modelo"]) == 3 * 25


# ----------------------------------------------------------------------
# Almacenamiento, paquete y acciones
# ----------------------------------------------------------------------


def prueba_pipeline_y_paquete():
    import generar_sinteticos as gs

    base = Path(tempfile.mkdtemp())
    try:
        datos, coords, _ = gs.generar()
        datos = datos[datos["Sondeo"].isin(["S1", "S2"])]
        ruta = base / "sev.txt"
        datos.to_csv(ruta, sep="\t", index=False, na_rep="")
        rc = base / "coords.txt"
        coords.to_csv(rc, sep="\t", index=False)
        resultado = acc.accion_crear_proyecto(base, "Prueba")
        proyecto = resultado.datos["proyecto"]
        cfg = pl.ConfigPipeline()
        cfg.inversion.n_trials = 6
        hasta = "procesado" if RAPIDO else "parametros"
        r = acc.accion_ejecutar_archivo(proyecto, ruta, cfg, hasta_etapa=hasta)
        assert r, r.mensaje
        assert acc.accion_cargar_coordenadas(proyecto, rc)
        estado = pl.estado_sondeo(proyecto, "S2")
        assert estado["procesado"]
        assert alm.listar_sondeos(proyecto).set_index("sondeo").loc["S2", "arreglo"] == "WE"
        if not RAPIDO:
            assert all(estado.values())
            assert acc.accion_unidades(proyecto)
            assert acc.accion_perfil(proyecto, "P1", [[499990, 2150000], [500110, 2150003]], 50, True, 20, 1)
        else:
            pass
        paquete = acc.accion_empaquetar_proyecto(proyecto)
        assert paquete
        restaurado = acc.accion_restaurar_proyecto(paquete.datos["contenido"], base / "otro")
        assert restaurado and len(alm.listar_sondeos(restaurado.datos["proyecto"])) == 2
    finally:
        shutil.rmtree(base, ignore_errors=True)


def prueba_rechaza_proyecto_2d():
    import io
    import json
    import zipfile

    import paquete_sev as pk

    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w") as z:
        z.writestr(pk.NOMBRE_MANIFIESTO, json.dumps({"formato": "iprt", "version_formato": 1}))
    try:
        pk.restaurar_proyecto(memoria.getvalue(), tempfile.mkdtemp())
        raise AssertionError("Debió rechazar un .iprt")
    except pk.PaqueteIncompatibleError:
        pass


def main() -> int:
    pruebas = [(n, f) for n, f in globals().items() if n.startswith("prueba_") and callable(f)]
    fallos = []
    inicio = time.time()
    for nombre, funcion in pruebas:
        t = time.time()
        try:
            funcion()
            print(f"  OK   {nombre} ({time.time() - t:.1f}s)")
        except Exception:
            fallos.append(nombre)
            print(f"  FALLA {nombre}")
            traceback.print_exc()
    print(f"\n{len(pruebas) - len(fallos)}/{len(pruebas)} pruebas correctas en {time.time() - inicio:.0f}s"
          + (" (modo rápido)" if RAPIDO else ""))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
