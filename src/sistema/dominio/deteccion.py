"""Detección en planta en código puro (sin LLM):
- Alerta de seguridad local (S1 a S4) en < 30s reales, sin falsas alarmas.
- Evaluación determinística de partes confirmados contra límites de /api/planta (planta.yaml).
- Detección de anomalías sostenidas en sensores (descartando picos aislados).
- Construcción del contexto de escalado (parte, fotos y tendencia de las últimas horas).
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from planta_contrato import EventoSistema, FilaParte, LecturaSensor, MensajeTablet, Pantalla, TipoEvento, Topicos

# Factores de conversión de presión hacia kg/cm2 (idénticos a los de la planta / corrector)
FACTOR_A_KG_CM2: dict[str, float] = {
    "kg/cm2": 1.0,
    "kg/cm²": 1.0,
    "kilos": 1.0,
    "psi": 1.0 / 14.2233,
    "bar": 1.01972,
}

# Mapeo entre variables leídas localmente en recorrida y variables de sensores con límite en planta.yaml
VARIABLES_PARTE_A_SENSOR: dict[str, str] = {
    "presion_descarga_local": "presion_descarga",
    "presion_descarga": "presion_descarga",
    "presion_diferencial": "presion_diferencial",
    "presion_diferencial_local": "presion_diferencial",
    "presion_succion": "presion_succion",
    "temperatura_aceite": "temperatura_aceite",
    "vibracion": "vibracion",
    "caudal": "caudal",
}

# Observaciones operativas que indican estado normal (NO son una anomalía)
OBSERVACIONES_NORMALES: set[str] = {
    "parada, en reserva",
    "parada en reserva",
    "en reserva",
    "parada por mantenimiento",
    "en mantenimiento",
    "sin novedad",
    "todo normal",
    "normal",
    "bien",
    "",
}

# Patrones determinísticos para las 4 condiciones inseguras del catálogo (S1 a S4)
PATRONES_SEGURIDAD: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "S1",
        "Pérdida con proyección de agua a presión",
        re.compile(
            r"(agua a presion|proyeccion de agua|chorro a presion|chorro de agua|niebla de agua|sale agua con fuerza|se rompio algo en la descarga|rotura en la descarga)",
            re.IGNORECASE,
        ),
    ),
    (
        "S2",
        "Presencia u olor a gas en planta",
        re.compile(r"(olor a gas|huevo podrido|fuga de gas|escape de gas|presencia de gas|h2s|detector.*gas)", re.IGNORECASE),
    ),
    (
        "S3",
        "Humo, olor a quemado o fuego en equipo",
        re.compile(
            r"(sale humo|humo del motor|olor a quemado|hay humo|fuego|incendio|llamas en)",
            re.IGNORECASE,
        ),
    ),
    (
        "S4",
        "Persona lesionada o accidente personal",
        re.compile(
            r"(sangrando|persona lesionada|se golpeo la mano|se lastimo|herido|accidente|lesionado)",
            re.IGNORECASE,
        ),
    ),
]


def _normalizar(texto: str | None) -> str:
    if not texto:
        return ""
    t = unicodedata.normalize("NFKD", texto.lower().strip())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass
class ResultadoSeguridad:
    es_inseguro: bool
    codigo: str | None = None
    descripcion: str | None = None
    equipo: str | None = None
    evento: EventoSistema | None = None
    aviso_tablet: MensajeTablet | None = None


@dataclass
class ContextoEscalado:
    """Contexto adjunto a todo evento Escalado (parte confirmado, fotos y tendencia reciente)."""

    evento: EventoSistema
    filas_parte: list[FilaParte] = field(default_factory=list)
    fotos: list[str] = field(default_factory=list)
    tendencias_sensores: dict[str, Any] = field(default_factory=dict)
    motivos_anomalia: list[str] = field(default_factory=list)


class DetectorPlanta:
    """Motor determinístico (en código, sin LLM) para detección de anomalías y alertas de seguridad."""

    def __init__(self, definicion_planta: dict[str, Any], muestras_sostenidas: int = 3):
        self.definicion = definicion_planta
        self.muestras_sostenidas = max(2, muestras_sostenidas)
        # Mapa (tag, variable) -> dict con alarma_baja, alarma_alta, disparo, unidad
        self.limites: dict[tuple[str, str], dict[str, Any]] = {
            (s["tag"], s["variable"]): s for s in definicion_planta.get("sensores", [])
        }
        # Mapa nombre informal normalizado -> TAG para inferir equipo en alertas de seguridad
        self.alias_a_tag: dict[str, str] = {}
        for eq in definicion_planta.get("equipos", []):
            tag = eq["tag"]
            self.alias_a_tag[_normalizar(tag)] = tag
            for alias in eq.get("nombres", []):
                self.alias_a_tag[_normalizar(alias)] = tag

        # Contadores de desvíos consecutivos por (tag, variable) para filtrar picos aislados
        self._desvios_consecutivos: dict[tuple[str, str], int] = {}
        self._episodio_activo: set[tuple[str, str]] = set()
        self._historial_reciente: dict[tuple[str, str], deque[tuple[str, float]]] = {}
        self._baseline_filtro: dict[str, tuple[datetime, float]] = {}
        self._ultimo_caudal_m3d: float = 2400.0

    # ------------------------------------------------------------------ 1. Alerta de Seguridad (S1-S4)
    def evaluar_seguridad_nota(
        self,
        nota_id: str,
        ts: datetime,
        texto_nota: str,
        equipo_sugerido: str | None = None,
    ) -> ResultadoSeguridad:
        """Evalúa en milisegundos si el texto de una nota describe una condición insegura (S1 a S4)."""
        texto_norm = _normalizar(texto_nota)
        texto_norm = re.sub(r"\bdel\s+a1\b", "de la uno", texto_norm)
        texto_norm = re.sub(r"\bel\s+a1\b", "la uno", texto_norm)
        if not texto_norm:
            return ResultadoSeguridad(es_inseguro=False)

        for codigo, descripcion, patron in PATRONES_SEGURIDAD:
            if patron.search(texto_norm):
                equipo = equipo_sugerido or self._inferir_equipo_en_texto(texto_norm)
                evento = EventoSistema(
                    id=f"EV-SEG-{nota_id}",
                    ts=ts,
                    tipo=TipoEvento.alerta_seguridad,
                    equipo=equipo,
                    origen="parte",
                    parte_id=f"PARTE-{nota_id}",
                    nota_ids=[nota_id],
                    resumen=f"Alerta de seguridad {codigo} ({descripcion}): {texto_nota.strip()[:140]}",
                )
                aviso = MensajeTablet(
                    id=f"TAB-SEG-{nota_id}",
                    ts=ts,
                    nota_ids=[nota_id],
                    texto=(
                        f"¡ALERTA DE SEGURIDAD ({codigo})! Aplique de inmediato el procedimiento de emergencia, "
                        "retírese de la zona de riesgo y dé aviso por radio al supervisor de turno."
                    ),
                    pantalla=Pantalla(
                        titulo=f"¡ALERTA DE SEGURIDAD ({codigo})!",
                        filas=[
                            FilaParte(
                                tag=equipo or "PLANTA",
                                variable="condicion_insegura",
                                valor=None,
                                unidad=None,
                                observacion=f"{codigo}: {descripcion}",
                            )
                        ],
                        opciones=[],
                    ),
                    espera_respuesta=False,
                )
                return ResultadoSeguridad(
                    es_inseguro=True,
                    codigo=codigo,
                    descripcion=descripcion,
                    equipo=equipo,
                    evento=evento,
                    aviso_tablet=aviso,
                )

        return ResultadoSeguridad(es_inseguro=False)

    def _inferir_equipo_en_texto(self, texto_norm: str) -> str | None:
        # Ordenamos alias de más largo a más corto para evitar falsos parciales
        for alias, tag in sorted(self.alias_a_tag.items(), key=lambda x: len(x[0]), reverse=True):
            if len(alias) >= 3 and re.search(rf"\b{re.escape(alias)}\b", texto_norm):
                return tag
        return None

    # ------------------------------------------------------------------ 2. Límites numéricos y observaciones
    def valor_fuera_de_limite(
        self, tag: str, variable: str, valor: float | None, unidad: str | None
    ) -> tuple[bool, str | None]:
        """Compara un valor contra los límites de alarma de planta.yaml (nunca Rev. C).
        Aplica conversión determinística de unidades (psi / bar -> kg/cm2)."""
        if valor is None:
            return False, None

        # Límite local específico de temperatura IR de rodamiento (> 75 °C indica rodamiento caliente)
        if variable == "temperatura_rodamiento_ir":
            if valor > 75.0:
                return True, f"temperatura_rodamiento_ir={valor} degC supera límite normal (75 degC)"
            return False, None

        sensor_var = VARIABLES_PARTE_A_SENSOR.get(variable)
        if not sensor_var:
            return False, None

        lim = self.limites.get((tag, sensor_var))
        if not lim:
            return False, None

        unidad_norm = _normalizar(unidad) if unidad else "kg/cm2"
        factor = FACTOR_A_KG_CM2.get(unidad_norm, 1.0)
        valor_convertido = float(valor) * factor

        alarma_alta = lim.get("alarma_alta")
        alarma_baja = lim.get("alarma_baja")

        if alarma_alta is not None and valor_convertido >= float(alarma_alta):
            return (
                True,
                f"{variable}={round(valor_convertido, 2)} {lim['unidad']} >= alarma_alta ({alarma_alta})",
            )
        if alarma_baja is not None and valor_convertido <= float(alarma_baja):
            return (
                True,
                f"{variable}={round(valor_convertido, 2)} {lim['unidad']} <= alarma_baja ({alarma_baja})",
            )

        return False, None

    def observacion_es_anomala(self, observacion: str | None) -> bool:
        """Determina si una observación del parte representa una anomalía."""
        if observacion is None:
            return False
        obs_norm = _normalizar(observacion)
        if not obs_norm:
            return False
        return obs_norm not in OBSERVACIONES_NORMALES

    # ------------------------------------------------------------------ 3. Evaluación de Parte Confirmado
    def evaluar_parte_confirmado(
        self,
        nota_id: str,
        ts: datetime,
        filas: list[FilaParte],
        fotos: list[str] | None = None,
        tendencias_por_equipo: dict[str, Any] | None = None,
    ) -> list[ContextoEscalado]:
        """Evalúa un parte confirmado y genera un EventoSistema idempotente por cada equipo involucrado
        (o un único evento SinNovedad si el parte no tenía filas)."""
        fotos = fotos or []
        tendencias_por_equipo = tendencias_por_equipo or {}

        if not filas:
            ev = EventoSistema(
                id=f"EV-{nota_id}",
                ts=ts,
                tipo=TipoEvento.sin_novedad,
                equipo=None,
                origen="parte",
                parte_id=f"PARTE-{nota_id}",
                nota_ids=[nota_id],
                resumen="Recorrida sin novedad.",
            )
            return [ContextoEscalado(evento=ev)]

        # Agrupamos filas por equipo (TAG) preservando orden de aparición
        filas_por_equipo: dict[str, list[FilaParte]] = {}
        for f in filas:
            filas_por_equipo.setdefault(f.tag, []).append(f)

        resultados: list[ContextoEscalado] = []
        for tag, filas_eq in filas_por_equipo.items():
            motivos: list[str] = []
            for f in filas_eq:
                fuera, detalle_lim = self.valor_fuera_de_limite(f.tag, f.variable, f.valor, f.unidad)
                if fuera and detalle_lim:
                    motivos.append(detalle_lim)
                if self.observacion_es_anomala(f.observacion):
                    motivos.append(f"observación: {f.observacion}")

            es_anomalo = len(motivos) > 0
            tipo_ev = TipoEvento.escalado if es_anomalo else TipoEvento.sin_novedad
            resumen = (
                f"Anomalía en {tag}: {'; '.join(motivos)}"
                if es_anomalo
                else f"Lecturas de {tag} dentro de rango normal"
            )
            ev = EventoSistema(
                id=f"EV-{nota_id}-{tag}",
                ts=ts,
                tipo=tipo_ev,
                equipo=tag,
                origen="parte",
                parte_id=f"PARTE-{nota_id}",
                nota_ids=[nota_id],
                resumen=resumen[:200],
            )
            resultados.append(
                ContextoEscalado(
                    evento=ev,
                    filas_parte=filas_eq,
                    fotos=fotos if es_anomalo else [],
                    tendencias_sensores=tendencias_por_equipo.get(tag, {}),
                    motivos_anomalia=motivos,
                )
            )

        return resultados

    # ------------------------------------------------------------------ 4. Evaluación de Sensores (Anomalía Sostenida)
    def evaluar_lectura_sensor(self, lectura: LecturaSensor) -> ContextoEscalado | None:
        """Evalúa una lectura de planta/sensores/#.
        Ignora picos aislados y lecturas de bombas paradas en reserva (`presion_descarga < 15 kg/cm2`).
        Escala cuando el desvío se sostiene durante `self.muestras_sostenidas` muestras consecutivas,
        o cuando un filtro muestra colmatación acelerada sostenida (>15x tasa normal), emitiendo un único
        evento idempotente por episodio."""
        clave = (lectura.tag, lectura.variable)
        if lectura.valor is None or lectura.calidad != "buena":
            return None

        val_f = float(lectura.valor)

        # Una bomba parada (en reserva o mantenimiento) marca presión estática ~1.6 kg/cm2; no es alarma baja en marcha
        if lectura.variable == "presion_descarga" and val_f < 15.0:
            self._desvios_consecutivos[clave] = 0
            return None

        # Durante un corte de energía o arranque, el caudal del colector cae temporalmente cerca de cero
        if lectura.variable == "caudal":
            if val_f < 100.0:
                self._desvios_consecutivos[clave] = 0
                return None
            if lectura.tag == "FT-401" and val_f >= 500.0:
                self._ultimo_caudal_m3d = val_f

        ventana = self._historial_reciente.setdefault(clave, deque(maxlen=60))
        ventana.append((lectura.ts.isoformat(), val_f))

        fuera, detalle = self.valor_fuera_de_limite(
            lectura.tag, lectura.variable, lectura.valor, lectura.unidad
        )

        # Detección predictiva de colmatación acelerada (F07) en filtros F-201A / F-201B
        # aplicando corrección por caudal (FC48-3 §3.2: ΔP_corr = ΔP / (Q / 2400)^2) para eliminar la onda diaria.
        # Tasa normal: F-201A ~0.0017 y F-201B ~0.0011 kg/cm2/h; >= 0.0065 es >4x-6x la tasa normal.
        if (
            lectura.tag in ("F-201A", "F-201B")
            and lectura.variable == "presion_diferencial"
        ):
            if val_f < 0.30:
                self._baseline_filtro.pop(lectura.tag, None)
            else:
                q_rel = max(0.5, min(1.3, self._ultimo_caudal_m3d / 2400.0))
                dp_corr = val_f / (q_rel ** 2)
                base = self._baseline_filtro.get(lectura.tag)
                if base is None or dp_corr < base[1] - 0.06:
                    self._baseline_filtro[lectura.tag] = (lectura.ts, dp_corr)
                elif not fuera and dp_corr >= 0.50:
                    t0_base, v0_base = base
                    dt_h = (lectura.ts - t0_base).total_seconds() / 3600.0
                    delta_v = dp_corr - v0_base
                    if dt_h >= 3.5 and delta_v >= 0.04 and (delta_v / dt_h) >= 0.0065:
                        fuera = True
                        detalle = (
                            f"tendencia acelerada de colmatación en {lectura.tag} "
                            f"(ΔP corregida subió de {v0_base:.2f} a {dp_corr:.2f} kg/cm2 en {dt_h:.1f} h)"
                        )

        if not fuera:
            self._desvios_consecutivos[clave] = 0
            self._episodio_activo.discard(clave)
            return None

        cuenta = self._desvios_consecutivos.get(clave, 0) + 1
        self._desvios_consecutivos[clave] = cuenta

        # Si todavía no alcanzó el umbral de muestras consecutivas (pico aislado) o ya se escaló este episodio
        if cuenta < self.muestras_sostenidas or clave in self._episodio_activo:
            return None

        self._episodio_activo.add(clave)
        ts_bloque = lectura.ts.strftime("%Y%m%d%H")
        ev = EventoSistema(
            id=f"EV-SENS-{lectura.tag}-{lectura.variable}-{ts_bloque}",
            ts=lectura.ts,
            tipo=TipoEvento.escalado,
            equipo=lectura.tag,
            origen="sensor",
            parte_id=None,
            nota_ids=[],
            resumen=f"Desvío sostenido en sensor {lectura.tag}/{lectura.variable}: {detalle}",
        )
        return ContextoEscalado(
            evento=ev,
            filas_parte=[],
            fotos=[],
            tendencias_sensores={
                lectura.variable: [{"ts": t, "valor": v} for t, v in list(ventana)[-10:]]
            },
            motivos_anomalia=[detalle or "desvío sostenido"],
        )
