"""
auth_app.py
==================================================================
Capa de autenticación de la aplicación: pantalla de acceso, guardia de
páginas y panel de administración de licencias.

Toda la lógica (validación de vigencia, registro de sesiones, hashes)
vive en :mod:`licencias_tre`; aquí solo se dibujan widgets y se
conecta con ``streamlit-authenticator``.

Uso desde cualquier pantalla::

    import auth_app
    sesion = auth_app.exigir_acceso()   # detiene la página si no procede

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

import _ruta_src  # noqa: F401  (registra src en sys.path)
import licencias_tre as lic
import marca_sev as marca_tre

logger = logging.getLogger("tre.auth")

#: Ubicación del archivo de credenciales, junto a la aplicación.
RUTA_CREDENCIALES = Path(__file__).resolve().parent / "credenciales.yaml"

CLAVE_ID_SESION = "id_sesion_local"
CLAVE_ESTADO_LICENCIA = "estado_licencia"


def _id_sesion() -> str:
    """Identificador único de esta pestaña del navegador.

    Se genera una vez por sesión de Streamlit y persiste en
    ``session_state``: es lo que distingue "el mismo usuario recargando"
    de "otra persona entrando con la misma licencia".
    """
    if CLAVE_ID_SESION not in st.session_state:
        st.session_state[CLAVE_ID_SESION] = uuid.uuid4().hex
    else:
        pass
    return st.session_state[CLAVE_ID_SESION]


@st.cache_resource
def _registro_sesiones() -> lic.RegistroSesiones:
    """Registro de sesiones compartido por todos los usuarios.

    ``cache_resource`` es lo que lo hace compartido: devuelve el mismo
    objeto a todas las sesiones del proceso, que es justo lo que se
    necesita para detectar que dos personas usan la misma licencia.
    """
    return lic.RegistroSesiones()


@st.cache_data(ttl=30)
def _configuracion_cacheada(ruta: str, marca_tiempo: float) -> dict[str, Any]:
    """Credenciales cacheadas, con relectura automática al editar el
    archivo (la marca de tiempo forma parte de la clave de caché).

    El TTL corto es deliberado: permite suspender o renovar una licencia
    editando el YAML y que surta efecto en menos de un minuto, sin
    reiniciar la aplicación.
    """
    return lic.cargar_configuracion(Path(ruta))


#: Variable de entorno que puede contener el YAML de credenciales completo.
VARIABLE_ENTORNO_CREDENCIALES = "TRE_CREDENCIALES_YAML"

#: Variable de entorno que puede apuntar a un archivo de credenciales.
VARIABLE_ENTORNO_RUTA = "TRE_CREDENCIALES_RUTA"


def _configuracion_desde_entorno() -> Optional[dict[str, Any]]:
    """Lee las credenciales desde variables de entorno.

    Es la vía **portable**: ``st.secrets`` solo existe en Streamlit
    Cloud, mientras que las variables de entorno funcionan en cualquier
    servidor, contenedor o plataforma (VPS, Docker, Railway, Render,
    Fly.io, systemd…). Tenerla evita quedar atado a un proveedor: si
    mañana se cambia de hosting, la aplicación sigue leyendo sus
    credenciales sin tocar código.

    Se admiten dos formas:

    - ``TRE_CREDENCIALES_YAML``: el YAML completo como texto.
    - ``TRE_CREDENCIALES_RUTA``: ruta a un archivo de credenciales
      fuera del repositorio (p. ej. ``/etc/tre/credenciales.yaml``),
      que es lo habitual en un servidor propio.

    Returns
    -------
    dict or None
        ``None`` si ninguna variable está definida.
    """
    import os

    import yaml

    contenido = os.getenv(VARIABLE_ENTORNO_CREDENCIALES)
    if contenido:
        try:
            configuracion = yaml.safe_load(contenido)
        except yaml.YAMLError as exc:
            st.error(
                f"La variable {VARIABLE_ENTORNO_CREDENCIALES} no contiene YAML válido: {exc}"
            )
            return None
        if isinstance(configuracion, dict) and "credentials" in configuracion:
            configuracion.setdefault("licencias", {})
            return configuracion
        else:
            st.error(
                f"La variable {VARIABLE_ENTORNO_CREDENCIALES} no contiene la sección "
                "'credentials'."
            )
            return None
    else:
        pass

    ruta_externa = os.getenv(VARIABLE_ENTORNO_RUTA)
    if ruta_externa:
        try:
            return lic.cargar_configuracion(Path(ruta_externa))
        except lic.ConfiguracionLicenciasError as exc:
            st.error(f"No se pudieron leer las credenciales indicadas por "
                      f"{VARIABLE_ENTORNO_RUTA}: {exc}")
            return None
    else:
        pass

    return None


def _configuracion_desde_secrets() -> Optional[dict[str, Any]]:
    """Lee las credenciales de ``st.secrets`` si están definidas.

    Es la vía para despliegues alojados: el archivo ``credenciales.yaml``
    **no se versiona** (contiene los hashes y la clave que firma las
    cookies), así que en la nube no existe en el repositorio. En su
    lugar se pega su contenido en la configuración de secretos del
    host, donde queda cifrado y fuera del control de versiones.

    Se admiten dos formas de escribirlo en los secretos:

    - Secciones TOML nativas (``[credenciales.credentials.usernames.admin]``).
    - Una única clave ``credenciales_yaml`` con el YAML completo como
      texto, que es lo más cómodo: se copia y pega el archivo tal cual.

    Returns
    -------
    dict or None
        ``None`` si no hay credenciales en los secretos, para que el
        llamador caiga al archivo local.
    """
    # Streamlit no lanza al acceder a ``st.secrets``, sino al consultarlo:
    # sin archivo de secretos, cualquier ``in`` o indexación levanta
    # StreamlitSecretNotFoundError. Por eso la comprobación completa va
    # dentro del try, y no solo el acceso al objeto.
    try:
        secretos = st.secrets
        tiene_yaml = "credenciales_yaml" in secretos
        tiene_seccion = "credenciales" in secretos
    except Exception:
        # Ejecución local sin secretos configurados: es el caso normal en
        # desarrollo, no un error. Se cae al archivo local.
        return None

    if tiene_yaml:
        import yaml

        try:
            configuracion = yaml.safe_load(str(secretos["credenciales_yaml"]))
        except yaml.YAMLError as exc:
            st.error(f"El secreto 'credenciales_yaml' no es YAML válido: {exc}")
            return None
        if isinstance(configuracion, dict) and "credentials" in configuracion:
            configuracion.setdefault("licencias", {})
            return configuracion
        else:
            st.error("El secreto 'credenciales_yaml' no contiene la sección 'credentials'.")
            return None
    else:
        pass

    if tiene_seccion:
        # st.secrets devuelve objetos de solo lectura; se convierten a dict
        # normal porque el resto del código los modifica (setdefault).
        import json

        configuracion = json.loads(json.dumps(secretos["credenciales"], default=dict))
        if "credentials" in configuracion:
            configuracion.setdefault("licencias", {})
            return configuracion
        else:
            st.error("La sección 'credenciales' de los secretos no contiene 'credentials'.")
            return None
    else:
        pass

    return None


def cargar_configuracion() -> Optional[dict[str, Any]]:
    """Configuración de credenciales, o ``None`` si no se pudo cargar
    (en cuyo caso ya se mostró el error en pantalla).

    Orden de búsqueda: **variables de entorno**, luego secretos de
    Streamlit, luego archivo local. Así el mismo código funciona en desarrollo (con
    ``credenciales.yaml``) y en producción (con secretos), sin ramas
    condicionales en el despliegue ni riesgo de subir el archivo por
    descuido.
    """
    # Orden de búsqueda: entorno (portable) -> secretos de Streamlit ->
    # archivo local. El entorno va primero porque es el mecanismo que
    # funciona en cualquier plataforma; los secretos de Streamlit quedan
    # como comodidad para ese host concreto.
    desde_entorno = _configuracion_desde_entorno()
    if desde_entorno is not None:
        return desde_entorno
    else:
        pass

    desde_secretos = _configuracion_desde_secrets()
    if desde_secretos is not None:
        return desde_secretos
    else:
        pass

    if not RUTA_CREDENCIALES.exists():
        st.error(
            "**No hay credenciales configuradas.**\n\n"
            "En un despliegue alojado, defina el secreto `credenciales_yaml` "
            "con el contenido del archivo de credenciales.\n\n"
            "En local, créelo con:\n\n"
            "```\npython src/licencias_tre.py crear app/credenciales.yaml admin SuClaveSegura\n```"
        )
        return None
    else:
        pass

    try:
        return _configuracion_cacheada(
            str(RUTA_CREDENCIALES), RUTA_CREDENCIALES.stat().st_mtime
        )
    except lic.ConfiguracionLicenciasError as exc:
        st.error(f"Error en el archivo de credenciales: {exc}")
        return None


def _autenticador(configuracion: dict[str, Any]) -> stauth.Authenticate:
    """Instancia de ``streamlit-authenticator`` con estas credenciales."""
    cookie = configuracion.get("cookie", {})
    return stauth.Authenticate(
        configuracion["credentials"],
        cookie.get("name", "tre_app_auth"),
        cookie.get("key", "clave_por_defecto"),
        cookie.get("expiry_days", 7),
    )


def exigir_acceso() -> Optional[dict[str, Any]]:
    """Guardia de acceso: muestra el login si hace falta y detiene la
    página cuando el usuario no puede pasar.

    Comprueba, en orden: credenciales válidas → licencia vigente →
    sesión no duplicada. Los tres son motivos distintos de rechazo y se
    informan por separado, porque la acción del usuario es diferente en
    cada caso (reintentar, renovar, o cerrar la otra sesión).

    Returns
    -------
    dict or None
        ``{'usuario', 'nombre', 'estado', 'autenticador'}`` si el acceso
        procede. Si no, la función llama a ``st.stop()`` y no retorna.
    """
    configuracion = cargar_configuracion()
    if configuracion is None:
        st.stop()
    else:
        pass

    autenticador = _autenticador(configuracion)

    # La portada se dibuja solo mientras no hay sesión: una vez dentro, el
    # espacio vertical vale más para el trabajo que para la identidad, que
    # ya queda presente en el encabezado de cada pantalla.
    if st.session_state.get("authentication_status") is not True:
        marca_tre.portada_login()
    else:
        pass

    autenticador.login(location="main", key="login_tre")

    autenticado = st.session_state.get("authentication_status")
    usuario = st.session_state.get("username")
    nombre = st.session_state.get("name")

    if autenticado is False:
        st.error("Usuario o contraseña incorrectos.")
        st.stop()
    else:
        pass

    if autenticado is None:
        st.info("Ingrese sus credenciales para acceder a la aplicación.")
        st.stop()
    else:
        pass

    # --- Licencia ---
    estado = lic.evaluar_licencia(configuracion, usuario)
    if not estado:
        st.error(f"**Acceso denegado.** {estado.motivo}")
        autenticador.logout("Cerrar sesión", "main", key="logout_denegado")
        st.stop()
    else:
        pass

    # --- Sesión única ---
    registro = _registro_sesiones()
    id_sesion = _id_sesion()
    forzar = bool(st.session_state.get("forzar_sesion", False))
    concedida, mensaje = registro.intentar_ocupar(usuario, id_sesion, forzar=forzar)
    st.session_state["forzar_sesion"] = False

    if not concedida:
        st.warning(f"**Licencia en uso.** {mensaje}")
        columnas = st.columns(2)
        if columnas[0].button("Forzar acceso aquí", type="primary", key="btn_forzar"):
            st.session_state["forzar_sesion"] = True
            st.rerun()
        else:
            pass
        if columnas[1].button("Reintentar", key="btn_reintentar"):
            st.rerun()
        else:
            pass
        st.stop()
    else:
        pass

    # Si la sesión fue expulsada por otro acceso, el latido lo detecta aquí.
    if not registro.latido(usuario, id_sesion):
        st.warning(
            "Su sesión se cerró porque esta licencia se usó desde otro equipo. "
            "Vuelva a iniciar sesión si desea continuar aquí."
        )
        if st.button("Reconectar", key="btn_reconectar"):
            st.session_state["forzar_sesion"] = True
            st.rerun()
        else:
            pass
        st.stop()
    else:
        pass

    st.session_state[CLAVE_ESTADO_LICENCIA] = estado
    return {
        "usuario": usuario, "nombre": nombre,
        "estado": estado, "autenticador": autenticador,
    }


def barra_lateral_sesion(sesion: dict[str, Any]) -> None:
    """Dibuja en la barra lateral la identidad, la vigencia de la
    licencia y el botón de cierre de sesión."""
    estado: lic.EstadoLicencia = sesion["estado"]

    with st.sidebar:
        st.divider()
        st.markdown(f"👤 **{sesion['nombre']}**")
        st.caption(f"Usuario: `{sesion['usuario']}`")

        if estado.es_administrador:
            st.caption("🔑 Administrador")
        else:
            pass

        if estado.expira is not None:
            if estado.dias_restantes is not None and estado.dias_restantes <= 15:
                st.warning(f"Licencia vence en {estado.dias_restantes} días ({estado.expira}).")
            else:
                st.caption(f"Licencia vigente hasta {estado.expira}.")
        else:
            st.caption("Licencia sin caducidad.")

        if st.button("Cerrar sesión", use_container_width=True, key="btn_logout"):
            # El espacio temporal se elimina aquí: es el compromiso de no
            # conservar datos del usuario tras la sesión.
            import estado_app

            estado_app.liberar_espacio_trabajo()
            _registro_sesiones().liberar(sesion["usuario"], _id_sesion())
            sesion["autenticador"].logout(location="unrendered")
            st.rerun()
        else:
            pass


def panel_administracion(sesion: dict[str, Any]) -> None:
    """Panel de licencias, visible solo para administradores.

    Permite ver el estado de todas las cuentas, quién está conectado y
    liberar sesiones bloqueadas — las tres cosas que se necesitan para
    dar soporte sin entrar por SSH al servidor.
    """
    estado: lic.EstadoLicencia = sesion["estado"]
    if not estado.es_administrador:
        return None
    else:
        pass

    configuracion = cargar_configuracion()
    if configuracion is None:
        return None
    else:
        pass

    with st.expander("🔑 Administración de licencias", expanded=False):
        st.markdown("**Cuentas registradas**")
        st.dataframe(
            pd.DataFrame(lic.listar_licencias(configuracion)),
            use_container_width=True, hide_index=True,
        )

        registro = _registro_sesiones()
        activas = registro.sesiones_activas()
        st.markdown("**Sesiones activas**")
        if activas:
            st.dataframe(
                pd.DataFrame(
                    [{"Usuario": u, "Inactivo (s)": s} for u, s in activas.items()]
                ),
                use_container_width=True, hide_index=True,
            )
            usuario_liberar = st.selectbox(
                "Liberar sesión de", ["—"] + list(activas), key="sel_liberar_sesion"
            )
            if usuario_liberar != "—":
                if st.button(f"Liberar sesión de '{usuario_liberar}'", key="btn_liberar_sesion"):
                    registro.liberar(usuario_liberar)
                    st.success(f"Sesión de '{usuario_liberar}' liberada.")
                    st.rerun()
                else:
                    pass
            else:
                pass
        else:
            st.caption("No hay sesiones activas en este momento.")

        import paquete_sev as pk

        uso = pk.uso_disco_espacios()
        st.markdown("**Espacios de trabajo temporales**")
        st.caption(
            f"{uso['n_espacios']} espacio(s) activos · {uso['total_mb']} MB en disco. "
            f"Se eliminan solos tras {pk.HORAS_CADUCIDAD_ESPACIO:.0f} h sin actividad."
        )
        if st.button("🧹 Limpiar espacios caducados ahora", key="btn_limpiar_espacios"):
            eliminados = pk.limpiar_espacios_caducados()
            st.success(f"{eliminados} espacio(s) eliminado(s).")
            st.rerun()
        else:
            pass

        st.divider()
        st.caption(
            "Para altas, bajas y renovaciones edite `credenciales.yaml` o use la "
            "utilidad de terminal:\n\n"
            "```\npython src/licencias_tre.py agregar app/credenciales.yaml "
            'usuario Clave "Nombre" 2027-03-31\n```\n'
            "Los cambios se detectan automáticamente en menos de un minuto."
        )
    return None
