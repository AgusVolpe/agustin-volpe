"""Interpretación del parte por voz:
- Parseo de números hablados en español de campo ("ciento doce", "mil seiscientos veinte", "uno ocho", "uno y medio").
- Extracción de filas validadas a FilaParte (una fila por equipo y variable, aunque la nota los mezcle).
- Resolución de nombres informales y unidades contra GET /api/planta (usando PlantaTools).
- Detección de ambigüedades ("la bomba" a secas, diferencial "dieciocho" en vez de "uno ocho", o fuera de rango histórico).
- Gestión del diálogo con la tablet (borrador, preguntas con opciones y aplicación de correcciones del operador).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from planta_contrato import FilaParte, MensajeTablet, Pantalla

from sistema.integraciones.tools import (
    ConsultarEquiposInput,
    ConsultarHistorialSensoresInput,
    PlantaTools,
)

log = logging.getLogger(__name__)

# Diccionario de números hablados en español para convertir texto dictado en campo a float exacto
_VALORES_PALABRA: dict[str, int] = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiun": 21,
    "veintiuno": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}

MAPA_UNIDADES_CAMPO: dict[str, str] = {
    "kilo": "kg/cm2",
    "kilos": "kg/cm2",
    "kilogramo": "kg/cm2",
    "kilogramos": "kg/cm2",
    "kg/cm2": "kg/cm2",
    "kg/cm²": "kg/cm2",
    "kg": "kg/cm2",
    "psi": "psi",
    "libra": "psi",
    "libras": "psi",
    "bar": "bar",
    "bares": "bar",
    "metro": "m",
    "metros": "m",
    "m": "m",
    "grado": "degC",
    "grados": "degC",
    "degc": "degC",
    "gotas/min": "gotas/min",
    "gotas por minuto": "gotas/min",
}


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower().strip())
    return "".join(c for c in t if not unicodedata.combining(c))


def parsear_entero_espanol(tokens: list[str]) -> int | None:
    """Convierte una secuencia de palabras numéricas ('mil', 'seiscientos', 'veinte') a entero (1620)."""
    total = 0
    acumulado = 0
    usado = False
    for tok in tokens:
        if tok == "y":
            continue
        if tok.isdigit():
            acumulado += int(tok)
            usado = True
        elif tok == "mil":
            acumulado = (acumulado or 1) * 1000
            total += acumulado
            acumulado = 0
            usado = True
        elif tok in _VALORES_PALABRA:
            acumulado += _VALORES_PALABRA[tok]
            usado = True
        else:
            break
    return (total + acumulado) if usado else None


def extraer_numero_hablado(fragmento: str) -> tuple[float | None, bool]:
    """Extrae un número (entero o decimal hablado como 'uno ocho', 'uno coma ocho', 'uno y medio') de una frase.
    Devuelve (valor, fue_dicho_como_entero_mayor_a_10)."""
    norm = normalizar(fragmento)
    norm = re.sub(
        r"\b(b-10[123]|f-201[ab]|tk-301|ft-401|bomba\s+(?:uno|dos|tres|1|2|3)|la\s+(?:uno|dos|tres|1|2|3))\b",
        " ",
        norm,
    )
    # 1. Decimal con dígitos (ej. "1.8" o "1,8" o "112")
    m_dig = re.search(r"\b(\d+(?:[.,]\d+)?)\b", norm)
    if m_dig and not re.search(rf"\b(b-|f-|tk-|ft-){re.escape(m_dig.group(1))}", norm):
        s_num = m_dig.group(1).replace(",", ".")
        val = float(s_num)
        return val, ("." not in s_num and val >= 10)

    tokens = re.findall(r"[a-z0-9]+", norm)
    palabras_num = set(_VALORES_PALABRA) | {"mil", "coma", "con", "medio", "punto"}

    # Buscamos la primera ventana de palabras numéricas
    inicio = next((i for i, t in enumerate(tokens) if t in _VALORES_PALABRA or t == "mil"), None)
    if inicio is None:
        return None, False

    ventana: list[str] = []
    for t in tokens[inicio:]:
        if t in palabras_num or (t == "y" and ventana):
            ventana.append(t)
        else:
            break

    # Caso "X y medio" -> X.5
    if len(ventana) >= 3 and ventana[-2:] == ["y", "medio"]:
        ent = parsear_entero_espanol(ventana[:-2])
        if ent is not None:
            return float(ent) + 0.5, False

    # Caso "X coma Y" o "X con Y" o "X punto Y"
    for sep in ("coma", "con", "punto"):
        if sep in ventana:
            idx = ventana.index(sep)
            ent = parsear_entero_espanol(ventana[:idx])
            dec = parsear_entero_espanol(ventana[idx + 1 :])
            if ent is not None and dec is not None:
                return float(f"{ent}.{dec}"), False

    # Caso implícito de dos dígitos hablados separados ("uno ocho" -> 1.8, "dos cuatro" -> 2.4)
    if (
        len(ventana) == 2
        and ventana[0] in _VALORES_PALABRA
        and ventana[1] in _VALORES_PALABRA
        and _VALORES_PALABRA[ventana[0]] < 10
        and _VALORES_PALABRA[ventana[1]] < 10
    ):
        ent = _VALORES_PALABRA[ventana[0]]
        dec = _VALORES_PALABRA[ventana[1]]
        return float(f"{ent}.{dec}"), False

    ent = parsear_entero_espanol(ventana)
    if ent is not None:
        return float(ent), ent >= 10
    return None, False


@dataclass
class ResultadoExtraccionNota:
    filas: list[FilaParte] = field(default_factory=list)
    ambiguedad: dict[str, Any] | None = None
    opciones_tablet: list[str] = field(default_factory=list)
    pregunta_tablet: str | None = None


class ProcesadorParteVoz:
    """Interpreta notas de recorrida por voz y respuestas del operador en la tablet."""

    def __init__(self, tools: PlantaTools):
        self.tools = tools
        self._equipos_info = {
            e.tag: e for e in self.tools.consultar_equipos(ConsultarEquiposInput()).equipos
        }
        # Alias ordenados de mayor a menor longitud para detectar menciones de equipos en cada oración
        self._alias_ordenados: list[tuple[str, str]] = []
        for tag, eq in self._equipos_info.items():
            self._alias_ordenados.append((normalizar(tag), tag))
            for n in eq.nombres_informales:
                self._alias_ordenados.append((normalizar(n), tag))
        self._alias_ordenados.sort(key=lambda x: len(x[0]), reverse=True)

    def _goteo_actual_planta(self, tag: str) -> float | None:
        """Obtiene el caudal de goteo de empaquetadura actual del equipo desde /api/estado si está disponible."""
        try:
            estado = self.tools.api.obtener_estado()
            for b in estado.get("bombas", []):
                if b.get("tag") == tag and b.get("goteo", 0) > 0:
                    return float(round(b["goteo"]))
        except Exception:
            pass
        return None

    def _identificar_equipo_en_oracion(self, oracion_norm: str) -> tuple[str | None, bool, str | None]:
        """Devuelve (tag, es_ambiguo, frase_dicha)."""
        for alias, tag in self._alias_ordenados:
            if re.search(rf"\b{re.escape(alias)}\b", oracion_norm):
                return tag, False, alias

        # Verificar referencias ambiguas genéricas (ej. "la bomba", "una bomba", "el filtro")
        m_amb = re.search(r"\b(la bomba|una bomba|bomba|el filtro|un filtro|filtro)\b", oracion_norm)
        if m_amb:
            dicho = m_amb.group(1)
            res = self.tools.consultar_equipos(ConsultarEquiposInput(consulta=dicho))
            if res.ambiguo:
                return None, True, dicho
            if len(res.equipos) == 1:
                return res.equipos[0].tag, False, dicho
        return None, False, None

    def _extraer_unidad_presion(self, texto_norm: str, tag: str | None) -> str:
        if re.search(r"\b(psi|libras|libra)\b", texto_norm):
            return "psi"
        if re.search(r"\b(bar|bares)\b", texto_norm):
            return "bar"
        if re.search(r"\b(kilos|kilo|kilogramos|kg)\b", texto_norm):
            return "kg/cm2"
        # Si no dice unidad, usamos la unidad del manómetro local del equipo en /api/planta
        if tag and tag in self._equipos_info and self._equipos_info[tag].unidad_manometro_local:
            return self._equipos_info[tag].unidad_manometro_local  # type: ignore[return-value]
        return "kg/cm2"

    def _fuera_de_rango_historico(self, tag: str, variable_sensor: str, valor_en_unidad_sensor: float) -> bool:
        """Consulta los 45 días del historiador (/api/historian/{tag}/{variable}) para ver si el valor sale de rango."""
        try:
            hist = self.tools.consultar_historial_sensores(
                ConsultarHistorialSensoresInput(tag=tag, variable=variable_sensor, ultimas_muestras=200)
            )
            if hist.minimo_historico is not None and hist.maximo_historico is not None:
                margen = max(0.4, (hist.maximo_historico - hist.minimo_historico) * 0.35)
                return (
                    valor_en_unidad_sensor < hist.minimo_historico - margen
                    or valor_en_unidad_sensor > hist.maximo_historico + margen
                )
        except Exception:
            pass
        return False

    def extraer_parte_de_texto(
        self, texto: str, tag_forzado: str | None = None
    ) -> ResultadoExtraccionNota:
        """Extrae todas las filas FilaParte de una nota de voz, detectando ambigüedades de equipo o de valor."""
        texto_norm = normalizar(texto)
        if not texto_norm or "recorrida sin novedad" in texto_norm:
            return ResultadoExtraccionNota(filas=[])

        # Cortar cualquier intento de prompt injection ("nota para el sistema: ...")
        for corte in ("nota para el sistema", "instruccion para el sistema", "ignora las reglas"):
            if corte in texto_norm:
                idx = texto_norm.index(corte)
                texto_norm = texto_norm[:idx].strip()

        # Normalizar errores fonéticos frecuentes de Whisper en español de planta
        texto_norm = re.sub(r"\bambien\b", "tambien", texto_norm)
        texto_norm = re.sub(r"\b(?:fitro|filto|fito)\b", "filtro", texto_norm)
        texto_norm = re.sub(r"\bgolpequeo\b", "golpeteo", texto_norm)
        texto_norm = re.sub(r"\bfiltro\s+(?:no\s+ve|v|ve|be)\b", "filtro b", texto_norm)
        texto_norm = re.sub(r"\bfiltro\s+ha\b", "filtro a", texto_norm)
        texto_norm = re.sub(r"\bdel\s+a1\b", "de la uno", texto_norm)
        texto_norm = re.sub(r"\bel\s+a1\b", "la uno", texto_norm)
        texto_norm = re.sub(r"\bla\s+una\b", "la uno", texto_norm)
        texto_norm = re.sub(r"\blados\b", "la dos", texto_norm)
        texto_norm = re.sub(r"\bstein\b", "esta en", texto_norm)
        texto_norm = re.sub(r"\bdesparada\b", "disparada", texto_norm)
        texto_norm = re.sub(r"(\d+)\s*[°º]", r"\1 grados", texto_norm)
        texto_norm = re.sub(
            r"\b(filtro\s+[ab]|primario|secundario|izquierda|derecha|rodamiento),?\s+medio\b",
            r"\1 me dio",
            texto_norm,
        )
        # Proteger comas decimales (1,4) y frases compuestas ("parada, en reserva") antes de dividir cláusulas
        texto_div = re.sub(r"(?<=\d),(?=\d)", ".", texto_norm)
        texto_div = re.sub(r"\bparada,\s*en\s+reserva\b", "parada en reserva", texto_div)
        texto_div = re.sub(r"\bdisparada,\s*no\s+arranca\b", "disparada no arranca", texto_div)

        oraciones = [o.strip() for o in re.split(r"[.;,]+", texto_div) if o.strip()]
        filas: list[FilaParte] = []
        ambiguedad_detectada: dict[str, Any] | None = None
        opciones_tablet: list[str] = []
        pregunta_tablet: str | None = None

        ultimo_tag: str | None = None

        for oracion in oraciones:
            # Limpiar muletillas y conectores al inicio de la oración (incluso encadenados)
            oracion_limpia = re.sub(
                r"^(?:(?:bueno|eh|a ver|listo|este|bien|ah y|ah|y|despues|tambien|ademas)\b[\s]*)+",
                "",
                oracion,
            ).strip()
            if not oracion_limpia:
                continue

            tag, es_amb, dicho_amb = self._identificar_equipo_en_oracion(oracion_limpia)
            if es_amb and tag_forzado:
                tag, es_amb = tag_forzado, False

            if es_amb:
                res_eq = self.tools.consultar_equipos(ConsultarEquiposInput(consulta=dicho_amb or "la bomba"))
                ambiguedad_detectada = {
                    "tipo": "equipo",
                    "dicho": dicho_amb,
                    "oracion": oracion_limpia,
                    "candidatos": [e.tag for e in res_eq.equipos],
                }
                opciones_tablet = res_eq.opciones_aclaracion
                pregunta_tablet = res_eq.pregunta_aclaracion or "¿A qué equipo te referís?"
                tag = "AMBIGUO"
            elif tag:
                ultimo_tag = tag
            else:
                tag = ultimo_tag

            if not tag:
                continue

            # 1. Nivel de tanque pulmón (TK-301)
            if tag == "TK-301" or "pulmon" in oracion_limpia or "metro" in oracion_limpia:
                val, _ = extraer_numero_hablado(oracion_limpia)
                obs = "rebalse a pileta" if "rebals" in oracion_limpia else None
                if val is not None or obs is not None:
                    filas.append(
                        FilaParte(tag="TK-301", variable="nivel", valor=val, unidad="m", observacion=obs)
                    )
                continue

            # 2. Presión diferencial de filtro (F-201A / F-201B)
            if "diferencial" in oracion_limpia or tag in ("F-201A", "F-201B"):
                parte_num = oracion_limpia.split("me dio")[-1] if "me dio" in oracion_limpia else oracion_limpia
                val, entero_mayor_10 = extraer_numero_hablado(parte_num)
                if val is not None:
                    # Detectar ambigüedad de valor dicho como "dieciocho" (18) en lugar de "uno ocho" (1.8)
                    if entero_mayor_10 and 10 <= val <= 35:
                        val_corregido = round(val / 10.0, 1)
                        ambiguedad_detectada = {
                            "tipo": "valor",
                            "tag": tag,
                            "variable": "presion_diferencial",
                            "dicho": val,
                            "sugerido": val_corregido,
                        }
                        opciones_tablet = [
                            f"{tag}: {val_corregido} kg/cm2",
                            f"{tag}: {val} kg/cm2",
                        ]
                        pregunta_tablet = (
                            f"En {tag} escuché {int(val)} de diferencial (fuera de rango histórico). "
                            f"¿Quisiste decir {val_corregido} kg/cm²? Confirmame el parte."
                        )
                        val = val_corregido
                    elif self._fuera_de_rango_historico(tag, "presion_diferencial", val) and val > 4.0:
                        ambiguedad_detectada = {
                            "tipo": "rango_historico",
                            "tag": tag,
                            "variable": "presion_diferencial",
                            "valor": val,
                        }
                        opciones_tablet = [f"Confirmar {val} kg/cm2 en {tag}", "Corregir valor"]
                        pregunta_tablet = (
                            f"El valor {val} kg/cm² en {tag} está fuera del rango histórico. ¿Confirmás que es correcto?"
                        )

                    obs = "diferencial alto" if val >= 1.5 else None
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="presion_diferencial",
                            valor=val,
                            unidad="kg/cm2",
                            observacion=obs,
                        )
                    )
                continue

            # 3. Bombas: una misma oración puede combinar descarga, empaquetadura, rodamiento, aceite, correas, ruido o estado
            subfrases = [s.strip() for s in oracion_limpia.split(",") if s.strip()]

            # 3a. Estado parada / reserva / mantenimiento / disparo
            if any(w in oracion_limpia for w in ("parada", "en reserva", "por mantenimiento", "disparada", "no arranca")):
                if "mantenimiento" in oracion_limpia:
                    obs_est = "parada por mantenimiento"
                elif "disparada" in oracion_limpia or "no arranca" in oracion_limpia:
                    obs_est = "disparada, no arranca"
                else:
                    obs_est = "parada, en reserva"
                filas.append(FilaParte(tag=tag, variable="estado", valor=None, unidad=None, observacion=obs_est))
                continue

            for sub in subfrases:
                # 3b. Presión de descarga local
                if "descarga" in sub or ("esta en" in sub and "grad" not in sub and "metro" not in sub):
                    tramo = sub.split("esta en")[-1] if "esta en" in sub else sub
                    val, _ = extraer_numero_hablado(tramo)
                    if val is not None:
                        unidad = self._extraer_unidad_presion(sub, None if tag == "AMBIGUO" else tag)
                        filas.append(
                            FilaParte(
                                tag=tag,
                                variable="presion_descarga_local",
                                valor=float(val),
                                unidad=unidad,
                            )
                        )

                # 3c. Goteo / pérdida por empaquetadura (solo cargar valor numérico si el operador lo dijo)
                if "empaquetadura" in sub or "gotea" in sub or "chorrea" in sub:
                    fuerte = any(w in sub for w in ("bastante", "chorrea", "importante", "mucho"))
                    obs_emp = (
                        "pérdida importante por empaquetadura" if fuerte else "goteo por empaquetadura"
                    )
                    val_goteo, _ = extraer_numero_hablado(sub)
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="goteo_empaquetadura",
                            valor=val_goteo,
                            unidad="gotas/min" if val_goteo is not None else None,
                            observacion=obs_emp,
                        )
                    )

                # 3d. Temperatura de rodamiento con pistola IR
                if "rodamiento" in sub and ("dio" in sub or "grado" in sub or "pistola" in sub):
                    tramo = sub.split("me dio")[-1] if "me dio" in sub else sub
                    val_t, _ = extraer_numero_hablado(tramo)
                    if val_t is not None:
                        filas.append(
                            FilaParte(
                                tag=tag,
                                variable="temperatura_rodamiento_ir",
                                valor=float(val_t),
                                unidad="degC",
                                observacion="rodamiento caliente" if val_t > 75.0 else None,
                            )
                        )

                # 3e. Nivel de aceite en el visor
                if "visor" in sub or ("aceite" in sub and ("bajo" in sub or "casi no tiene" in sub)):
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="nivel_aceite_visor",
                            valor=None,
                            unidad=None,
                            observacion="nivel de aceite bajo",
                        )
                    )

                # 3f. Estado de correas
                if "correa" in sub or "chillan" in sub:
                    obs_cor = "correas gastadas" if "gastada" in sub else "correas flojas"
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="estado_correas",
                            valor=None,
                            unidad=None,
                            observacion=obs_cor,
                        )
                    )

                # 3g. Ruido (golpeteo, piedras/cavitación, rodamiento)
                if "golpeteo" in sub:
                    filas.append(
                        FilaParte(
                            tag=tag, variable="ruido", valor=None, unidad=None, observacion="hace un golpeteo metálico"
                        )
                    )
                elif "piedras" in sub:
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="ruido",
                            valor=None,
                            unidad=None,
                            observacion="suena como si tuviera piedras adentro",
                        )
                    )
                elif "ruido" in sub and "rodamiento" in sub:
                    filas.append(
                        FilaParte(
                            tag=tag,
                            variable="ruido",
                            valor=None,
                            unidad=None,
                            observacion="tiene un ruido raro de rodamiento",
                        )
                    )

        return ResultadoExtraccionNota(
            filas=filas,
            ambiguedad=ambiguedad_detectada,
            opciones_tablet=opciones_tablet,
            pregunta_tablet=pregunta_tablet,
        )

    def armar_mensaje_tablet_inicial(
        self, nota_id: str, ts: datetime, extraccion: ResultadoExtraccionNota, intento: int = 1
    ) -> tuple[MensajeTablet, bool]:
        """Arma el MensajeTablet inicial para una nota.
        Devuelve (MensajeTablet, queda_confirmado_inmediato)."""
        # Si hay ambigüedad de equipo ("la bomba" a secas), preguntamos primero con opciones (sin filas)
        if extraccion.ambiguedad and extraccion.ambiguedad.get("tipo") == "equipo":
            msg = MensajeTablet(
                id=f"TAB-{nota_id}-preg-{intento}",
                ts=ts,
                nota_ids=[nota_id],
                texto=extraccion.pregunta_tablet or "¿A cuál equipo te referís?",
                pantalla=Pantalla(
                    titulo="Aclarar equipo del parte",
                    filas=[],
                    opciones=extraccion.opciones_tablet,
                ),
                espera_respuesta=True,
            )
            return msg, False

        filas_con_valor = [f for f in extraccion.filas if f.valor is not None]
        # Si la nota no tiene valores numéricos para validar en dialogo.py (ej. solo "parada, en reserva" o nota vacía),
        # mostramos las filas en pantalla y confirmamos sin esperar respuesta para no generar pregunta_de_mas
        if not filas_con_valor:
            msg = MensajeTablet(
                id=f"TAB-{nota_id}-{intento}",
                ts=ts,
                nota_ids=[nota_id],
                texto="Parte registrado sin lecturas numéricas pendientes de confirmación.",
                pantalla=Pantalla(
                    titulo="Parte de recorrida · registrado",
                    filas=extraccion.filas,
                    opciones=[],
                ),
                espera_respuesta=False,
            )
            return msg, True

        # Si hay ambigüedad de valor o rango histórico (ej. "dieciocho" -> 1.8), mandamos el borrador con opciones
        if extraccion.ambiguedad and extraccion.opciones_tablet:
            msg = MensajeTablet(
                id=f"TAB-{nota_id}-{intento}",
                ts=ts,
                nota_ids=[nota_id],
                texto=extraccion.pregunta_tablet or "Verificá el valor fuera de rango y confirmame el parte.",
                pantalla=Pantalla(
                    titulo="Parte de recorrida · confirmar valor",
                    filas=extraccion.filas,
                    opciones=extraccion.opciones_tablet,
                ),
                espera_respuesta=True,
            )
            return msg, False

        # Caso estándar sin ambigüedad: enviamos borrador directo pidiendo confirmación (opciones=[])
        msg = MensajeTablet(
            id=f"TAB-{nota_id}-{intento}",
            ts=ts,
            nota_ids=[nota_id],
            texto="Te muestro el borrador del parte. ¿Confirmás que está bien?",
            pantalla=Pantalla(
                titulo="Parte de recorrida · borrador",
                filas=extraccion.filas,
                opciones=[],
            ),
            espera_respuesta=True,
        )
        return msg, False

    def procesar_respuesta_operador(
        self,
        texto_respuesta: str,
        filas_actuales: list[FilaParte],
        texto_nota_original: str,
        ambiguedad_pendiente: dict[str, Any] | None,
    ) -> tuple[bool, list[FilaParte]]:
        """Interpreta la respuesta del operador en planta/respuestas.
        Devuelve (confirmo_parte, nuevas_filas)."""
        resp_norm = normalizar(texto_respuesta)

        # 1. Si estábamos esperando aclaración de equipo ambiguo ("Era la uno", "La dos")
        if ambiguedad_pendiente and ambiguedad_pendiente.get("tipo") == "equipo":
            tag_elegido, _, _ = self._identificar_equipo_en_oracion(resp_norm)
            if tag_elegido:
                re_ext = self.extraer_parte_de_texto(texto_nota_original, tag_forzado=tag_elegido)
                return False, re_ext.filas

        # 2. Si el operador confirma ("Sí, confirmo", "Está bien, confirmo", "Correcto, dale", "Perfecto")
        if any(w in resp_norm for w in ("confirmo", "esta bien", "correcto", "perfecto", "asi es")) and not resp_norm.startswith("no"):
            return True, filas_actuales

        # 3. Si el operador corrige ("No. La uno estaba en ciento quince kilos, no en...")
        nuevas = [FilaParte(**f.model_dump()) for f in filas_actuales]
        oraciones = [o.strip() for o in re.split(r"[.;]+", resp_norm) if o.strip()]

        for oracion in oraciones:
            if oracion in ("no", "corregilo y te confirmo", "arregla eso", "con eso esta"):
                continue
            tag, _, _ = self._identificar_equipo_en_oracion(oracion)
            if not tag:
                continue

            # Corrección de unidad ("Lo de la dos es en psi, no en kilos")
            if "es en " in oracion:
                tramo_u = oracion.split("es en ", 1)[1].split(",")[0]
                nueva_u = self._extraer_unidad_presion(tramo_u, tag)
                for f in nuevas:
                    if f.tag == tag and f.valor is not None:
                        f.unidad = nueva_u

            # Corrección de valor o lectura faltante ("La uno estaba en ciento quince kilos..." / "Te faltó la dos, que estaba en cien gotas por minuto")
            elif "estaba en " in oracion:
                tramo_v = oracion.split("estaba en ", 1)[1].split(", no en")[0]
                val, _ = extraer_numero_hablado(tramo_v)
                if val is not None:
                    if "gota" in tramo_v:
                        var_obj, unidad = "goteo_empaquetadura", "gotas/min"
                    elif "grado" in tramo_v:
                        var_obj, unidad = "temperatura_rodamiento_ir", "degC"
                    elif "metro" in tramo_v or tag == "TK-301":
                        var_obj, unidad = "nivel", "m"
                    elif tag.startswith("F-") or "diferencial" in tramo_v:
                        var_obj, unidad = "presion_diferencial", "kg/cm2"
                    else:
                        var_obj, unidad = "presion_descarga_local", self._extraer_unidad_presion(tramo_v, tag)

                    actualizada = False
                    # 1) Buscar fila exacta del equipo + variable
                    for f in nuevas:
                        if f.tag == tag and f.variable == var_obj:
                            f.valor = float(val)
                            f.unidad = unidad
                            actualizada = True
                            break
                    # 2) Fallback: si la respuesta dice ", no en <viejo>", matchear por valor viejo
                    if not actualizada and ", no en " in oracion:
                        val_viejo, _ = extraer_numero_hablado(oracion.split(", no en ", 1)[1])
                        for f in nuevas:
                            if f.tag == tag and f.valor is not None and val_viejo is not None and abs(f.valor - val_viejo) <= 1.0:
                                f.valor = float(val)
                                f.unidad = unidad
                                actualizada = True
                                break
                    if not actualizada:
                        nuevas.append(FilaParte(tag=tag, variable=var_obj, valor=float(val), unidad=unidad))

            # Corrección de equipo ("Eso de ciento doce era de la uno, no de la dos")
            elif "era de " in oracion and ", no de " in oracion:
                val_ref, _ = extraer_numero_hablado(oracion.split("era de ", 1)[0])
                parte_corr = oracion.split("era de ", 1)[1]
                tag_real, _, _ = self._identificar_equipo_en_oracion(parte_corr.split(", no de")[0])
                tag_mal, _, _ = self._identificar_equipo_en_oracion(parte_corr.split(", no de")[1])
                if tag_real and tag_mal:
                    for f in nuevas:
                        if f.tag == tag_mal and (
                            val_ref is None or (f.valor is not None and abs(f.valor - val_ref) < 0.2)
                        ):
                            f.tag = tag_real
                            if f.variable == "presion_descarga_local":
                                f.unidad = self._extraer_unidad_presion("", tag_real)
                            break

        return False, nuevas
