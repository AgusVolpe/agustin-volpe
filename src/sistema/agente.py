"""Punto de entrada y coordinador principal:
- Recepción MQTT (planta/recorridas, planta/respuestas, planta/sensores/#, planta/energia).
- Latido cada 25s en sistema/estado con pendientes_sincronizar real.
- Flujo completo del Parte por Voz (A) con presupuesto de pasos y timeout por nota (1.4).
- Detección en planta en código (B): AlertaSeguridad (<30s) y clasificación SinNovedad / Escalado.
- Persistencia del estado de conversación por nota en PostgreSQL (2.4).
"""

from __future__ import annotations

import logging
import queue
import time
from datetime import datetime, timezone

import httpx
from planta_contrato import (
    EstadoSistema,
    FilaParte,
    LecturaSensor,
    MensajeTablet,
    NotaRecorrida,
    Pantalla,
    RespuestaOperador,
    TipoEvento,
    Topicos,
)

from sistema.config import settings
from sistema.dominio.deteccion import DetectorPlanta
from sistema.integraciones.mqtt_cliente import MqttCliente
from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import PlantaTools
from sistema.persistencia.repositorio import EstadoConversacionNota, RepositorioPlanta
from sistema.procesamiento.analisis import ContextBuilderAnomalias
from sistema.procesamiento.parte_voz import ProcesadorParteVoz
from sistema.procesamiento.transcripcion import TranscriptorLocal
from sistema.rag.buscador import BuscadorRAG
from sistema.rag.indexador import IndexadorRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sistema.agente")

MAX_INTENTOS_TABLET = 3  # Condición de corte / presupuesto de pasos por nota (1.4)


class AgentePlanta:
    def __init__(self):
        self.cola_mensajes: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.mqtt = MqttCliente(on_mensaje_callback=self.encolar_mensaje)
        self.http_interno = httpx.Client(timeout=3.0, trust_env=False)
        self.ultimo_ts_planta: datetime = datetime.now(timezone.utc)

        self.api = PlantaAPIClient()
        self.tools = PlantaTools(api_client=self.api)
        self.repo = RepositorioPlanta()
        self.transcriptor = TranscriptorLocal()
        self.detector: DetectorPlanta | None = None
        self.procesador_voz: ProcesadorParteVoz | None = None
        self.indexador_rag: IndexadorRAG | None = None
        self.buscador_rag: BuscadorRAG | None = None
        self.context_builder: ContextBuilderAnomalias | None = None

    def inicializar_componentes(self) -> None:
        self.repo.inicializar_esquema()
        definicion_planta = self.api.obtener_planta()
        self.detector = DetectorPlanta(definicion_planta=definicion_planta, muestras_sostenidas=3)
        self.procesador_voz = ProcesadorParteVoz(tools=self.tools)
        self.indexador_rag = IndexadorRAG()
        self.indexador_rag.indexar_en_postgres()
        self.buscador_rag = BuscadorRAG(indexador=self.indexador_rag)
        self.context_builder = ContextBuilderAnomalias(tools=self.tools, buscador_rag=self.buscador_rag)
        self.transcriptor.precargar()
        log.info("Componentes de dominio, voz, persistencia, RAG y STT inicializados.")

    def encolar_mensaje(self, topico: str, payload: dict) -> None:
        if "ts" in payload and payload["ts"]:
            try:
                self.ultimo_ts_planta = datetime.fromisoformat(payload["ts"])
            except ValueError:
                pass
        if not topico.startswith("planta/sensores/"):
            log.info("Mensaje recibido en %s: %s", topico, payload.get("id", ""))
        self.cola_mensajes.put((topico, payload))

    def obtener_ts_planta(self) -> datetime:
        """Devuelve la hora simulada actual de la planta."""
        try:
            r = self.http_interno.get(f"{settings.sim_url}/api/estado")
            if r.status_code == 200:
                ts_str = r.json().get("ts")
                if ts_str:
                    self.ultimo_ts_planta = datetime.fromisoformat(ts_str)
        except Exception:
            pass
        return self.ultimo_ts_planta

    def verificar_enlace(self) -> bool:
        """Consulta si el radioenlace de la planta está arriba o caído."""
        try:
            r = self.http_interno.get(f"{settings.enlace_control_url}/estado")
            if r.status_code == 200:
                return bool(r.json().get("arriba", True))
        except Exception:
            pass
        return False

    def publicar_latido(self) -> None:
        enlace_ok = self.verificar_enlace()
        if enlace_ok:
            sincronizados = self.repo.marcar_pendientes_sincronizados()
            if sincronizados > 0:
                log.info("Enlace restablecido: %d eventos pendientes sincronizados.", sincronizados)

        pendientes = self.repo.contar_pendientes_sincronizar()
        estado = EstadoSistema(
            ts=self.obtener_ts_planta(),
            sistema=settings.sistema_nombre,
            enlace_ok=enlace_ok,
            pendientes_sincronizar=pendientes,
        )
        self.mqtt.publicar(Topicos.ESTADO, estado)
        log.info(
            "Latido enviado -> sistema=%s, enlace_ok=%s, pendientes=%d",
            estado.sistema,
            estado.enlace_ok,
            estado.pendientes_sincronizar,
        )

    # ------------------------------------------------------------------ Flujo A y B: Nota de Recorrida
    def manejar_nota_recorrida(self, payload: dict) -> None:
        t0 = time.monotonic()
        nota = NotaRecorrida.model_validate(payload)
        assert self.detector is not None and self.procesador_voz is not None

        # 1. Descargar audio WAV y transcribir localmente
        texto = ""
        if nota.audio_url:
            try:
                wav_bytes = self.api.descargar_bytes(nota.audio_url)
                texto = self.transcriptor.transcribir_wav_bytes(wav_bytes)
            except Exception as e:
                log.error("Error transcribiendo nota %s: %s", nota.id, e)

        # 2. Evaluar inmediatamente Condición Insegura (S1 a S4) en código puro (< 30s reales, sin nube)
        res_seg = self.detector.evaluar_seguridad_nota(
            nota_id=nota.id,
            ts=nota.ts,
            texto_nota=texto,
        )
        if res_seg.es_inseguro and res_seg.evento and res_seg.aviso_tablet:
            enlace_ok = self.verificar_enlace()
            self.mqtt.publicar(Topicos.TABLET, res_seg.aviso_tablet)
            self.mqtt.publicar(Topicos.EVENTOS, res_seg.evento)
            self.repo.guardar_evento(res_seg.evento, sincronizado=enlace_ok)
            self.repo.guardar_conversacion(
                EstadoConversacionNota(
                    nota_id=nota.id,
                    recorrida_id=nota.recorrida_id,
                    ts_nota=nota.ts.isoformat(),
                    operador=nota.operador,
                    audio_url=nota.audio_url,
                    fotos=nota.fotos,
                    texto_transcripto=texto,
                    estado="Confirmado",
                    intento=1,
                    ultimo_mensaje_tablet_id=res_seg.aviso_tablet.id,
                    filas=[],
                )
            )
            log.warning(
                "AlertaSeguridad (%s) emitida para %s en %.2f s",
                res_seg.codigo,
                nota.id,
                time.monotonic() - t0,
            )
            return

        # 3. Extraer filas del parte y detectar si hay ambigüedad de equipo o de valor
        extraccion = self.procesador_voz.extraer_parte_de_texto(texto)
        msg_tablet, confirmado_inmediato = self.procesador_voz.armar_mensaje_tablet_inicial(
            nota_id=nota.id,
            ts=nota.ts,
            extraccion=extraccion,
            intento=1,
        )

        conv = EstadoConversacionNota(
            nota_id=nota.id,
            recorrida_id=nota.recorrida_id,
            ts_nota=nota.ts.isoformat(),
            operador=nota.operador,
            audio_url=nota.audio_url,
            fotos=nota.fotos,
            texto_transcripto=texto,
            estado="Confirmado" if confirmado_inmediato else "Borrador",
            intento=1,
            ultimo_mensaje_tablet_id=msg_tablet.id,
            filas=[f.model_dump() for f in extraccion.filas],
            ambiguedad_pendiente=extraccion.ambiguedad,
        )
        self.repo.guardar_conversacion(conv)
        self.mqtt.publicar(Topicos.TABLET, msg_tablet)

        # Si era una nota sin valores numéricos (ej. solo "parada, en reserva" o VIVO vacía), ya quedó confirmada
        if confirmado_inmediato:
            self._emitir_eventos_de_parte_confirmado(conv)

    # ------------------------------------------------------------------ Flujo A: Respuesta en Tablet
    def manejar_respuesta_operador(self, payload: dict) -> None:
        resp = RespuestaOperador.model_validate(payload)
        assert self.procesador_voz is not None

        conv = self.repo.obtener_conversacion_por_mensaje_tablet(resp.mensaje_id)
        if not conv:
            log.warning("Respuesta %s para mensaje desconocido %s", resp.id, resp.mensaje_id)
            return

        filas_actuales = [FilaParte.model_validate(f) for f in conv.filas]
        confirmo, nuevas_filas = self.procesador_voz.procesar_respuesta_operador(
            texto_respuesta=resp.texto,
            filas_actuales=filas_actuales,
            texto_nota_original=conv.texto_transcripto,
            ambiguedad_pendiente=conv.ambiguedad_pendiente,
        )

        # Condición de corte: si el operador confirmó o se alcanzó el presupuesto máximo de intentos
        if confirmo or conv.intento >= MAX_INTENTOS_TABLET:
            conv.estado = "Confirmado"
            conv.filas = [f.model_dump() for f in nuevas_filas]
            conv.ambiguedad_pendiente = None
            msg_ok = MensajeTablet(
                id=f"TAB-{conv.nota_id}-ok",
                ts=resp.ts,
                nota_ids=[conv.nota_id],
                texto="Listo, parte confirmado y guardado.",
                pantalla=None,
                espera_respuesta=False,
            )
            conv.ultimo_mensaje_tablet_id = msg_ok.id
            self.repo.guardar_conversacion(conv)
            self.mqtt.publicar(Topicos.TABLET, msg_ok)
            self._emitir_eventos_de_parte_confirmado(conv)
            return

        # Si el operador aclaró una ambigüedad de equipo o corrigió un dato, enviamos el borrador actualizado
        conv.intento += 1
        conv.filas = [f.model_dump() for f in nuevas_filas]
        conv.ambiguedad_pendiente = None

        filas_con_valor = [f for f in nuevas_filas if f.valor is not None]
        if not filas_con_valor:
            conv.estado = "Confirmado"
            msg_ok = MensajeTablet(
                id=f"TAB-{conv.nota_id}-ok",
                ts=resp.ts,
                nota_ids=[conv.nota_id],
                texto="Listo, parte guardado.",
                pantalla=Pantalla(titulo="Parte de recorrida · confirmado", filas=nuevas_filas, opciones=[]),
                espera_respuesta=False,
            )
            conv.ultimo_mensaje_tablet_id = msg_ok.id
            self.repo.guardar_conversacion(conv)
            self.mqtt.publicar(Topicos.TABLET, msg_ok)
            self._emitir_eventos_de_parte_confirmado(conv)
            return

        msg_nuevo = MensajeTablet(
            id=f"TAB-{conv.nota_id}-{conv.intento}",
            ts=resp.ts,
            nota_ids=[conv.nota_id],
            texto="Corregido. ¿Confirmás ahora el parte?",
            pantalla=Pantalla(
                titulo="Parte de recorrida · borrador corregido",
                filas=nuevas_filas,
                opciones=[],
            ),
            espera_respuesta=True,
        )
        conv.ultimo_mensaje_tablet_id = msg_nuevo.id
        self.repo.guardar_conversacion(conv)
        self.mqtt.publicar(Topicos.TABLET, msg_nuevo)

    def _emitir_eventos_de_parte_confirmado(self, conv: EstadoConversacionNota) -> None:
        """Una vez que el parte pasa a Confirmado, evalúa las reglas en código y publica EventoSistema."""
        assert self.detector is not None
        # Notas en vivo sin ítems para reportar no obligan a generar anomalías
        if conv.nota_id.startswith("VIVO-") and not conv.filas:
            return

        filas = [FilaParte.model_validate(f) for f in conv.filas]
        ts_nota = datetime.fromisoformat(conv.ts_nota)
        contextos = self.detector.evaluar_parte_confirmado(
            nota_id=conv.nota_id,
            ts=ts_nota,
            filas=filas,
            fotos=conv.fotos,
        )
        enlace_ok = self.verificar_enlace()
        for ctx in contextos:
            ev = ctx.evento
            # Un evento Escalado necesita salir a análisis (queda pendiente de sincronizar si el enlace está caído)
            sincronizado = True if ev.tipo == TipoEvento.sin_novedad else enlace_ok
            self.repo.guardar_evento(ev, sincronizado=sincronizado)
            self.mqtt.publicar(Topicos.EVENTOS, ev)
            log.info(
                "Evento emitido: id=%s tipo=%s equipo=%s (sincronizado=%s)",
                ev.id,
                ev.tipo.value,
                ev.equipo,
                sincronizado,
            )

    def manejar_lectura_sensor(self, payload: dict) -> None:
        if self.detector is None:
            return
        try:
            lectura = LecturaSensor.model_validate(payload)
            ctx = self.detector.evaluar_lectura_sensor(lectura)
            if ctx is not None:
                enlace_ok = self.verificar_enlace()
                self.repo.guardar_evento(ctx.evento, sincronizado=enlace_ok)
                # Solo publicamos eventos de sensor si son requeridos o los persistimos para análisis
                log.info("Desvío sostenido en sensor detectado: %s", ctx.evento.id)
        except Exception as e:
            log.debug("Error evaluando sensor: %s", e)

    def procesar_cola(self) -> None:
        while not self.cola_mensajes.empty():
            topico, payload = self.cola_mensajes.get_nowait()
            try:
                if topico == Topicos.RECORRIDAS:
                    self.manejar_nota_recorrida(payload)
                elif topico == Topicos.RESPUESTAS:
                    self.manejar_respuesta_operador(payload)
                elif topico.startswith("planta/sensores/"):
                    self.manejar_lectura_sensor(payload)
            except Exception as e:
                log.exception("Error procesando tópico %s: %s", topico, e)
            finally:
                self.cola_mensajes.task_done()

    def correr(self) -> None:
        log.info("Iniciando Agente de Planta (%s)...", settings.sistema_nombre)
        self.inicializar_componentes()
        self.mqtt.iniciar()
        time.sleep(2)

        ultimo_latido = 0.0
        try:
            while True:
                ahora = time.time()
                if ahora - ultimo_latido >= settings.latido_intervalo_s:
                    self.publicar_latido()
                    ultimo_latido = ahora

                self.procesar_cola()
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("Deteniendo agente...")
        finally:
            self.mqtt.detener()


if __name__ == "__main__":
    AgentePlanta().correr()
