"""Cliente MQTT para suscribirse a planta/# y publicar en sistema/#."""

import json
import logging
from typing import Callable

import paho.mqtt.client as mqtt
from pydantic import BaseModel

from sistema.config import settings

log = logging.getLogger(__name__)


class MqttCliente:
    def __init__(self, on_mensaje_callback: Callable[[str, dict], None]):
        self.on_mensaje_callback = on_mensaje_callback
        self.cliente = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"agente-{settings.sistema_nombre}",
        )
        self.cliente.on_connect = self._on_connect
        self.cliente.on_message = self._on_message

    def iniciar(self) -> None:
        self.cliente.reconnect_delay_set(min_delay=1, max_delay=10)
        self.cliente.connect_async(settings.mqtt_host, settings.mqtt_port, keepalive=30)
        self.cliente.loop_start()

    def detener(self) -> None:
        self.cliente.loop_stop()
        self.cliente.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        log.info("Conectado a MQTT (%s:%s)", settings.mqtt_host, settings.mqtt_port)
        topicos = [
            ("planta/recorridas", 1),
            ("planta/respuestas", 1),
            ("planta/cierres", 1),
            ("planta/energia", 1),
            ("planta/sensores/#", 0),
        ]
        client.subscribe(topicos)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.on_mensaje_callback(msg.topic, payload)
        except Exception as e:
            log.error("Error procesando mensaje en %s: %s", msg.topic, e)

    def publicar(self, topico: str, modelo: BaseModel, retain: bool = False) -> None:
        """Valida y publica un modelo de planta_contrato en formato JSON."""
        if self.cliente.is_connected():
            payload_json = modelo.model_dump_json()
            self.cliente.publish(topico, payload_json, qos=1, retain=retain)