"""
licencias_tre.py
==================================================================
Gestión de licencias y control de sesión única.

Contiene **lógica pura**: valida vigencias, administra el registro de
sesiones activas y genera hashes de contraseña. No importa Streamlit,
de modo que todo aquí sea probable desde código sin abrir un navegador.

Modelo de licencias
----------------------
Las credenciales viven en un YAML junto a la aplicación, con la
estructura que espera ``streamlit-authenticator`` más una sección
propia de licencias::

    credentials:
      usernames:
        jperez:
          email: jperez@empresa.com
          name: Juan Pérez
          password: $2b$12$...        # hash bcrypt
    cookie:
      name: tre_app_auth
      key: <clave aleatoria>
      expiry_days: 7
    licencias:
      jperez:
        rol: usuario                  # 'usuario' | 'administrador'
        expira: '2027-03-31'          # null = sin caducidad
        activa: true

Un YAML es deliberado para 10-20 licencias: administrar un archivo de
texto es más rápido que montar y mantener una base de datos, y el
archivo se versiona y respalda con el resto del proyecto. Si la
cartera creciera a cientos de usuarios, este módulo es el punto por
donde se sustituiría el respaldo sin tocar la interfaz.

Sesión única
---------------
El registro de sesiones vive en memoria del proceso y **expira por
inactividad**. La expiración no es un adorno: sin ella, un usuario que
cierra el navegador sin desconectarse —lo habitual— dejaría su licencia
bloqueada indefinidamente y habría que liberarla a mano.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("tre.licencias")

#: Segundos sin latido tras los que una sesión se considera abandonada.
#:
#: 15 minutos es un equilibrio deliberado. El valor NO expulsa a quien está
#: trabajando —cada interacción renueva el latido— sino que define cuánto
#: espera una segunda persona antes de poder tomar la licencia de una sesión
#: que parece abandonada. Con un umbral corto (2 min), un usuario que deja la
#: pestaña abierta mientras levanta topografía en campo o revisa datos en otro
#: programa podía perder su sesión ante otro acceso. Con un umbral muy largo,
#: una licencia queda bloqueada demasiado tiempo tras un cierre accidental del
#: navegador. Quince minutos cubre las pausas normales de trabajo sin volver
#: molesta la recuperación.
TIEMPO_EXPIRACION_SESION = 900.0

#: Rol con acceso irrestricto.
ROL_ADMINISTRADOR = "administrador"


# ======================================================================
# Excepciones
# ======================================================================


class LicenciaError(Exception):
    """Error base de la gestión de licencias."""


class ConfiguracionLicenciasError(LicenciaError):
    """El archivo de credenciales falta o está mal formado."""


# ======================================================================
# Estado de una licencia
# ======================================================================


@dataclass
class EstadoLicencia:
    """Resultado de evaluar la licencia de un usuario.

    Es booleano-evaluable: ``bool(estado)`` equivale a ``estado.valida``,
    para que la interfaz escriba ``if estado:`` directamente.
    """

    valida: bool
    usuario: str
    rol: str = "usuario"
    motivo: str = ""
    expira: Optional[date] = None
    dias_restantes: Optional[int] = None

    def __bool__(self) -> bool:
        return self.valida

    @property
    def es_administrador(self) -> bool:
        """``True`` si la cuenta tiene rol de administrador."""
        return self.rol == ROL_ADMINISTRADOR


# ======================================================================
# Carga y validación de licencias
# ======================================================================


def cargar_configuracion(ruta: Path) -> dict[str, Any]:
    """Lee el YAML de credenciales y licencias.

    Raises
    ------
    ConfiguracionLicenciasError
        Si el archivo no existe, no es YAML válido o le falta la
        sección de credenciales.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        # El mensaje incluye la ruta ABSOLUTA que se intentó abrir: el error más
        # común no es que falte el archivo, sino que la ruta relativa se resuelve
        # desde el directorio donde se ejecuta el comando, no desde el del módulo.
        raise ConfiguracionLicenciasError(
            f"No se encontró el archivo de credenciales.\n"
            f"  Ruta buscada: {ruta.resolve()}\n"
            f"  Directorio actual: {Path.cwd()}\n\n"
            "Si el archivo existe, indique su ruta completa. Si aún no lo ha creado:\n"
            "  python src/licencias_tre.py crear app/credenciales.yaml admin SuClave"
        )
    else:
        pass

    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            configuracion = yaml.safe_load(archivo)
    except yaml.YAMLError as exc:
        raise ConfiguracionLicenciasError(f"El archivo '{ruta}' no es YAML válido: {exc}") from exc

    if not isinstance(configuracion, dict) or "credentials" not in configuracion:
        raise ConfiguracionLicenciasError(
            f"'{ruta}' no tiene la sección 'credentials' que requiere la autenticación."
        )
    else:
        pass

    configuracion.setdefault("licencias", {})
    configuracion.setdefault(
        "cookie", {"name": "tre_app_auth", "key": "cambie_esta_clave", "expiry_days": 7}
    )
    return configuracion


def guardar_configuracion(configuracion: dict[str, Any], ruta: Path) -> None:
    """Escribe el YAML de credenciales, creando la carpeta si falta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(configuracion, archivo, allow_unicode=True, sort_keys=False)


def _a_fecha(valor: Any) -> Optional[date]:
    """Normaliza a ``date`` un valor de fecha del YAML (que puede venir
    como ``date``, ``datetime`` o texto ISO)."""
    if valor is None:
        return None
    else:
        pass
    if isinstance(valor, datetime):
        return valor.date()
    else:
        pass
    if isinstance(valor, date):
        return valor
    else:
        pass
    try:
        return datetime.fromisoformat(str(valor)).date()
    except ValueError:
        logger.warning("Fecha de expiración no interpretable: %r; se trata como sin caducidad.", valor)
        return None


def evaluar_licencia(
    configuracion: dict[str, Any], usuario: str, hoy: Optional[date] = None
) -> EstadoLicencia:
    """Evalúa si un usuario autenticado tiene licencia vigente.

    Un usuario puede autenticarse correctamente (contraseña válida) y
    aun así no tener acceso: la autenticación y la licencia son cosas
    distintas, y separarlas permite suspender una cuenta sin borrarla,
    conservando su historial.

    Parameters
    ----------
    hoy : datetime.date, optional
        Fecha de referencia (inyectable para pruebas).

    Returns
    -------
    EstadoLicencia
        Con ``valida=False`` y un ``motivo`` legible si no procede.
    """
    hoy = hoy or date.today()
    licencias = configuracion.get("licencias", {}) or {}
    licencia = licencias.get(usuario)

    if licencia is None:
        # Sin entrada explícita se asume usuario estándar sin caducidad: así
        # una cuenta recién creada en 'credentials' funciona sin obligar a
        # duplicar el registro en dos sitios.
        return EstadoLicencia(valida=True, usuario=usuario, rol="usuario")
    else:
        pass

    rol = str(licencia.get("rol", "usuario"))

    if not licencia.get("activa", True):
        return EstadoLicencia(
            valida=False, usuario=usuario, rol=rol,
            motivo="Su licencia está suspendida. Contacte al administrador.",
        )
    else:
        pass

    expira = _a_fecha(licencia.get("expira"))
    if expira is None:
        return EstadoLicencia(valida=True, usuario=usuario, rol=rol)
    else:
        pass

    dias_restantes = (expira - hoy).days
    if dias_restantes < 0:
        return EstadoLicencia(
            valida=False, usuario=usuario, rol=rol, expira=expira,
            dias_restantes=dias_restantes,
            motivo=(f"Su licencia venció el {expira.isoformat()}. "
                     "Contacte al administrador para renovarla."),
        )
    else:
        pass

    return EstadoLicencia(
        valida=True, usuario=usuario, rol=rol, expira=expira, dias_restantes=dias_restantes
    )


# ======================================================================
# Registro de sesiones activas (sesión única por licencia)
# ======================================================================


def _formato_espera(segundos: float) -> str:
    """Formatea una espera en la unidad que el usuario entiende de un
    vistazo: segundos por debajo del minuto, minutos por encima."""
    if segundos < 60:
        return f"~{int(segundos)} segundos"
    else:
        minutos = int(round(segundos / 60))
        return f"~{minutos} minuto{'s' if minutos != 1 else ''}"


@dataclass
class _SesionActiva:
    """Sesión viva de un usuario."""

    id_sesion: str
    ultimo_latido: float = field(default_factory=time.time)


class RegistroSesiones:
    """Registro en memoria de qué licencia está en uso y por quién.

    Impide que dos personas usen la misma licencia a la vez. El control
    es **cooperativo con expiración**: cada interacción del usuario
    renueva su latido, y una sesión sin latidos por más de
    ``tiempo_expiracion`` segundos se considera abandonada y se libera
    automáticamente.

    Notes
    -----
    El registro vive en la memoria del proceso, que es suficiente para
    el despliegue previsto (una instancia de la aplicación). Si algún
    día se corren varias réplicas detrás de un balanceador, este
    registro debe mudarse a un almacén compartido (SQLite o Redis) para
    que todas vean las mismas sesiones; el resto de la interfaz no
    cambiaría, porque solo consume estos métodos.
    """

    def __init__(self, tiempo_expiracion: float = TIEMPO_EXPIRACION_SESION) -> None:
        self._sesiones: dict[str, _SesionActiva] = {}
        self._candado = threading.Lock()
        self.tiempo_expiracion = tiempo_expiracion

    def _esta_viva(self, sesion: _SesionActiva, ahora: float) -> bool:
        return (ahora - sesion.ultimo_latido) < self.tiempo_expiracion

    def intentar_ocupar(
        self, usuario: str, id_sesion: str, forzar: bool = False
    ) -> tuple[bool, str]:
        """Intenta registrar una sesión para un usuario.

        Parameters
        ----------
        forzar : bool
            Expulsa la sesión previa aunque siga viva. Es la salida para
            el usuario que cambió de equipo y no quiere esperar a que
            expire la anterior.

        Returns
        -------
        tuple[bool, str]
            ``(concedida, mensaje)``.
        """
        ahora = time.time()
        with self._candado:
            actual = self._sesiones.get(usuario)

            if actual is None:
                self._sesiones[usuario] = _SesionActiva(id_sesion, ahora)
                return True, "Sesión iniciada."
            else:
                pass

            if actual.id_sesion == id_sesion:
                actual.ultimo_latido = ahora
                return True, "Sesión ya activa."
            else:
                pass

            if not self._esta_viva(actual, ahora):
                logger.info("Sesión abandonada de '%s' liberada por inactividad.", usuario)
                self._sesiones[usuario] = _SesionActiva(id_sesion, ahora)
                return True, "Se liberó una sesión previa inactiva y se inició la suya."
            else:
                pass

            if forzar:
                logger.info("Sesión previa de '%s' expulsada a petición del usuario.", usuario)
                self._sesiones[usuario] = _SesionActiva(id_sesion, ahora)
                return True, "Se cerró la sesión anterior y se inició la suya."
            else:
                pass

            inactividad = ahora - actual.ultimo_latido
            espera_segundos = max(0.0, self.tiempo_expiracion - inactividad)
            return False, (
                "Esta licencia ya está en uso en otra sesión. Si es usted en otro "
                f"equipo o navegador, puede forzar el acceso o esperar "
                f"{_formato_espera(espera_segundos)} a que la sesión anterior "
                "expire por inactividad."
            )

    def latido(self, usuario: str, id_sesion: str) -> bool:
        """Renueva la marca de actividad. Devuelve ``False`` si la
        sesión ya no es la titular (fue expulsada por otro acceso)."""
        ahora = time.time()
        with self._candado:
            actual = self._sesiones.get(usuario)
            if actual is None or actual.id_sesion != id_sesion:
                return False
            else:
                pass
            actual.ultimo_latido = ahora
            return True

    def liberar(self, usuario: str, id_sesion: Optional[str] = None) -> None:
        """Libera la sesión de un usuario (cierre de sesión explícito).

        Con ``id_sesion`` solo libera si esa es la sesión titular, para
        que un cierre tardío de una sesión ya expulsada no eche a quien
        entró después.
        """
        with self._candado:
            actual = self._sesiones.get(usuario)
            if actual is None:
                return None
            else:
                pass
            if id_sesion is not None and actual.id_sesion != id_sesion:
                return None
            else:
                pass
            del self._sesiones[usuario]
        logger.info("Sesión de '%s' liberada.", usuario)
        return None

    def sesiones_activas(self) -> dict[str, float]:
        """Usuarios con sesión viva y sus segundos de inactividad."""
        ahora = time.time()
        with self._candado:
            return {
                usuario: round(ahora - sesion.ultimo_latido, 1)
                for usuario, sesion in self._sesiones.items()
                if self._esta_viva(sesion, ahora)
            }


# ======================================================================
# Administración de usuarios
# ======================================================================


def generar_hash(contrasena: str) -> str:
    """Genera el hash bcrypt de una contraseña."""
    import streamlit_authenticator as stauth

    return stauth.Hasher.hash(contrasena)


def crear_configuracion_inicial(
    usuario_admin: str, contrasena_admin: str, nombre_admin: str = "Administrador",
    correo_admin: str = "admin@local", clave_cookie: Optional[str] = None,
) -> dict[str, Any]:
    """Construye una configuración nueva con una única cuenta de
    administrador **sin caducidad**.

    La clave de la cookie se genera aleatoriamente si no se indica: es
    lo que firma la sesión, y dejar un valor por defecto compartido
    permitiría falsificar cookies entre despliegues.
    """
    import secrets

    return {
        "credentials": {
            "usernames": {
                usuario_admin: {
                    "email": correo_admin,
                    "name": nombre_admin,
                    "password": generar_hash(contrasena_admin),
                }
            }
        },
        "cookie": {
            "name": "tre_app_auth",
            "key": clave_cookie or secrets.token_urlsafe(32),
            "expiry_days": 7,
        },
        "licencias": {
            usuario_admin: {"rol": ROL_ADMINISTRADOR, "expira": None, "activa": True}
        },
    }


def agregar_usuario(
    configuracion: dict[str, Any], usuario: str, contrasena: str, nombre: str,
    correo: str = "", expira: Optional[str] = None, rol: str = "usuario",
) -> dict[str, Any]:
    """Añade o reemplaza un usuario con su licencia. Devuelve la
    configuración modificada (no la escribe a disco)."""
    configuracion.setdefault("credentials", {}).setdefault("usernames", {})[usuario] = {
        "email": correo or f"{usuario}@local",
        "name": nombre,
        "password": generar_hash(contrasena),
    }
    configuracion.setdefault("licencias", {})[usuario] = {
        "rol": rol, "expira": expira, "activa": True,
    }
    return configuracion


def listar_licencias(
    configuracion: dict[str, Any], hoy: Optional[date] = None
) -> list[dict[str, Any]]:
    """Resumen de todas las cuentas y el estado de su licencia, para el
    panel de administración."""
    hoy = hoy or date.today()
    usuarios = (configuracion.get("credentials", {}) or {}).get("usernames", {}) or {}

    filas: list[dict[str, Any]] = []
    for usuario, datos in usuarios.items():
        estado = evaluar_licencia(configuracion, usuario, hoy=hoy)
        filas.append({
            "Usuario": usuario,
            "Nombre": datos.get("name", ""),
            "Correo": datos.get("email", ""),
            "Rol": estado.rol,
            "Expira": estado.expira.isoformat() if estado.expira else "sin caducidad",
            "Días restantes": estado.dias_restantes if estado.dias_restantes is not None else "—",
            "Vigente": "sí" if estado else "no",
            "Motivo": estado.motivo,
        })
    return filas


# ======================================================================
# Utilidad de línea de comandos
# ======================================================================


def generar_secreto_streamlit(configuracion: dict[str, Any]) -> str:
    """Convierte la configuración en el bloque TOML listo para pegar en
    los secretos del host (``credenciales_yaml``).

    Se emite como una sola clave con el YAML embebido, en vez de
    traducir la estructura a secciones TOML: es menos propenso a
    errores de transcripción y permite copiar el archivo tal cual.
    """
    cuerpo = yaml.safe_dump(configuracion, allow_unicode=True, sort_keys=False)
    return 'credenciales_yaml = """\n' + cuerpo + '"""\n'


def _cli() -> None:
    """Administración desde la terminal, sin abrir la aplicación::

        python licencias_tre.py crear   credenciales.yaml admin MiClave123
        python licencias_tre.py agregar credenciales.yaml jperez Clave456 "Juan Pérez" 2027-03-31
        python licencias_tre.py listar  credenciales.yaml
        python licencias_tre.py hash    MiClave123
        python licencias_tre.py secreto credenciales.yaml   # para pegar en el host
    """
    import argparse

    analizador = argparse.ArgumentParser(description="Administración de licencias TRE/IP")
    subcomandos = analizador.add_subparsers(dest="comando", required=True)

    crear = subcomandos.add_parser("crear", help="crea el archivo con una cuenta de administrador")
    crear.add_argument("ruta")
    crear.add_argument("usuario")
    crear.add_argument("contrasena")
    crear.add_argument("--nombre", default="Administrador")

    agregar = subcomandos.add_parser("agregar", help="añade un usuario con licencia")
    agregar.add_argument("ruta")
    agregar.add_argument("usuario")
    agregar.add_argument("contrasena")
    agregar.add_argument("nombre")
    agregar.add_argument("expira", nargs="?", default=None, help="AAAA-MM-DD o vacío")
    agregar.add_argument("--rol", default="usuario")

    listar = subcomandos.add_parser("listar", help="muestra el estado de las licencias")
    listar.add_argument("ruta")

    hash_cmd = subcomandos.add_parser("hash", help="genera el hash de una contraseña")
    hash_cmd.add_argument("contrasena")

    secreto = subcomandos.add_parser(
        "secreto", help="imprime el bloque TOML para los secretos del host"
    )
    secreto.add_argument("ruta")

    argumentos = analizador.parse_args()

    if argumentos.comando == "hash":
        print(generar_hash(argumentos.contrasena))
        return None
    else:
        pass

    if argumentos.comando == "crear":
        configuracion = crear_configuracion_inicial(
            argumentos.usuario, argumentos.contrasena, nombre_admin=argumentos.nombre
        )
        guardar_configuracion(configuracion, Path(argumentos.ruta))
        print(f"Creado '{argumentos.ruta}' con el administrador '{argumentos.usuario}'.")
        return None
    else:
        pass

    configuracion = cargar_configuracion(Path(argumentos.ruta))

    if argumentos.comando == "secreto":
        print(generar_secreto_streamlit(configuracion))
        return None
    else:
        pass

    if argumentos.comando == "agregar":
        agregar_usuario(
            configuracion, argumentos.usuario, argumentos.contrasena,
            argumentos.nombre, expira=argumentos.expira, rol=argumentos.rol,
        )
        guardar_configuracion(configuracion, Path(argumentos.ruta))
        print(f"Usuario '{argumentos.usuario}' agregado (expira: {argumentos.expira or 'nunca'}).")
        return None
    else:
        pass

    for fila in listar_licencias(configuracion):
        print(fila)
    return None


if __name__ == "__main__":
    _cli()
