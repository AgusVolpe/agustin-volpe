"""Router explícito por tipo de mensaje entrante desde la planta (Requisito 3.2).

Clasifica cada mensaje MQTT en una de las rutas operativas del sistema:
- `seguridad`          : NotaRecorrida con condición insegura S1-S4 (vía rápida <30s, sin LLM).
- `parte`              : NotaRecorrida normal (flujo de voz + tablet).
- `sensor`             : LecturaSensor en `planta/sensores/{tag}/{variable}`.
- `respuesta_operador` : RespuestaOperador en `planta/respuestas`.
- `cierre`             : NotaCierre en `planta/cierres` (Bloque G).
- `energia`            : EstadoEnergia en `planta/energia` (corte/restablecimiento de luz).
- `desconocido`        : Cualquier otro tópico no contemplado.
"""

from __future__ import annotations

from enum import Enum

from planta_contrato import Topicos
from pydantic import BaseModel, ConfigDict


class RutaMensaje(str, Enum):
    seguridad = "seguridad"
    parte = "parte"
    sensor = "sensor"
    respuesta_operador = "respuesta_operador"
    cierre = "cierre"
    energia = "energia"
    desconocido = "desconocido"


class DecisionRuteo(BaseModel):
    """Resultado auditable del router de mensajes."""

    model_config = ConfigDict(extra="forbid")

    topico: str
    ruta: RutaMensaje
    motivo: str


def rutear_mensaje(topico: str, es_condicion_insegura: bool = False) -> DecisionRuteo:
    """Determina determinísticamente la ruta que debe atender un mensaje de la planta."""
    if topico == Topicos.RECORRIDAS:
        if es_condicion_insegura:
            return DecisionRuteo(
                topico=topico,
                ruta=RutaMensaje.seguridad,
                motivo="Nota de recorrida clasificada como condición insegura S1-S4 (vía rápida local).",
            )
        return DecisionRuteo(
            topico=topico,
            ruta=RutaMensaje.parte,
            motivo="Nota de recorrida normal dirigida al procesador de parte por voz.",
        )

    if topico == Topicos.RESPUESTAS:
        return DecisionRuteo(
            topico=topico,
            ruta=RutaMensaje.respuesta_operador,
            motivo="Respuesta del operador desde la tablet.",
        )

    if topico == Topicos.CIERRES:
        return DecisionRuteo(
            topico=topico,
            ruta=RutaMensaje.cierre,
            motivo="Nota de cierre de orden de trabajo (técnico propio o contratista).",
        )

    if topico == Topicos.ENERGIA:
        return DecisionRuteo(
            topico=topico,
            ruta=RutaMensaje.energia,
            motivo="Evento de corte o restablecimiento de energía eléctrica en planta.",
        )

    if topico.startswith("planta/sensores/"):
        return DecisionRuteo(
            topico=topico,
            ruta=RutaMensaje.sensor,
            motivo="Muestra periódica de sensor conectado.",
        )

    return DecisionRuteo(
        topico=topico,
        ruta=RutaMensaje.desconocido,
        motivo=f"Tópico sin ruta registrada: {topico}",
    )
