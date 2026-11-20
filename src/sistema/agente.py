"""Punto de entrada y coordinador principal (Entrega 1 + Entrega 2):
- Router explícito por tópico (seguridad, parte, sensor, respuesta_operador, cierre, energia).
- Latido cada 25s en sistema/estado con pendientes_sincronizar real.
- Flujo completo del Parte por Voz (A) con presupuesto de pasos por nota (1.4).
- Detección en planta en código (B): AlertaSeguridad (<30s, offline) y SinNovedad / Escalado.
- Orquestación LangGraph (C + D): WorkflowOrdenTrabajo con Ollama local + VLM NVIDIA NIM + Revisor + Reglas en código.
- Durabilidad (3.3 / F3): checkpoints por `correlation_id`, cola de análisis resiliente a cortes de enlace y apagón (`sin-luz`),
  anti-duplicación (`E3`) y trazabilidad LLM/VLM en PostgreSQL 16.
- Interfaz Humana propia (FastAPI + Panel Web en puerto 8090) para Planificador, Supervisor e Inspector.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import uvicorn
from planta_contrato import (
    EstadoEnergia,
    EstadoOrden,
    EstadoSistema,
    EventoSistema,
    FilaParte,
    LecturaSensor,
    MensajeTablet,
    NotaCierre,
    NotaRecorrida,
    OrdenSistema,
    Pantalla,
    RespuestaOperador,
    TipoEvento,
    Topicos,
)

from sistema.config import settings
from sistema.dominio.deteccion import ContextoEscalado, DetectorPlanta
from sistema.integraciones.mqtt_cliente import MqttCliente
from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import PlantaTools
from sistema.interfaz.api_humana import crear_app_humana
from sistema.orquestacion.router import RutaMensaje, rutear_mensaje
from sistema.orquestacion.workflow_orden import ResultadoWorkflowOrden, WorkflowOrdenTrabajo
from sistema.persistencia.repositorio import (
    CheckpointWorkflow,
    EstadoConversacionNota,
    ItemColaAnalisis,
    RegistroOrdenTrabajo,
    RepositorioPlanta,
)
from sistema.procesamiento.analisis import ContextBuilderAnomalias
from sistema.procesamiento.cierres import evaluar_cierre_y_certificacion
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


def serializar_contexto_escalado(ctx: ContextoEscalado) -> dict[str, Any]:
    """Serializa un ContextoEscalado a dict JSON-safe para persistencia durable en PostgreSQL."""
    return {
        "evento": ctx.evento.model_dump(mode="json"),
        "filas_parte": [f.model_dump(mode="json") for f in ctx.filas_parte],
        "fotos": list(ctx.fotos),
        "tendencias_sensores": dict(ctx.tendencias_sensores),
        "motivos_anomalia": list(ctx.motivos_anomalia),
    }


def deserializar_contexto_escalado(data: dict[str, Any]) -> ContextoEscalado:
    """Reconstruye un ContextoEscalado desde JSON persistido en `cola_eventos_analisis`."""
    return ContextoEscalado(
        evento=EventoSistema.model_validate(data["evento"]),
        filas_parte=[FilaParte.model_validate(f) for f in data.get("filas_parte", [])],
        fotos=list(data.get("fotos", [])),
        tendencias_sensores=dict(data.get("tendencias_sensores", {})),
        motivos_anomalia=list(data.get("motivos_anomalia", [])),
    )


class AgentePlanta:
    def __init__(self, iniciar_ui: bool = True, iniciar_worker_analisis: bool = True):
        self.cola_mensajes: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.mqtt = MqttCliente(on_mensaje_callback=self.encolar_mensaje)
        self.http_interno = httpx.Client(timeout=3.0, trust_env=False)
        self.ultimo_ts_planta: datetime = datetime.now(timezone.utc)
        self.iniciar_ui = iniciar_ui
        self.iniciar_worker_analisis = iniciar_worker_analisis

        self.api = PlantaAPIClient()
        self.tools = PlantaTools(api_client=self.api)
        self.repo = RepositorioPlanta()
        self.transcriptor = TranscriptorLocal()
        self.detector: DetectorPlanta | None = None
        self.procesador_voz: ProcesadorParteVoz | None = None
        self.indexador_rag: IndexadorRAG | None = None
        self.buscador_rag: BuscadorRAG | None = None
        self.context_builder: ContextBuilderAnomalias | None = None
        self.workflow_orden: WorkflowOrdenTrabajo | None = None

        self._lock_analisis = threading.Lock()
        self._evento_nuevo_analisis = threading.Event()
        self._detener_event = threading.Event()
        self._hilo_analisis: threading.Thread | None = None
        self._hilo_ui: threading.Thread | None = None
        self._hay_energia: bool = True

    def inicializar_componentes(self) -> None:
        self.repo.inicializar_esquema()
        definicion_planta = self.api.obtener_planta()
        self.detector = DetectorPlanta(definicion_planta=definicion_planta, muestras_sostenidas=2)
        self.procesador_voz = ProcesadorParteVoz(tools=self.tools)
        self.indexador_rag = IndexadorRAG()
        self.indexador_rag.indexar_en_postgres()
        self.buscador_rag = BuscadorRAG(indexador=self.indexador_rag)
        self.context_builder = ContextBuilderAnomalias(tools=self.tools, buscador_rag=self.buscador_rag)
        self.workflow_orden = WorkflowOrdenTrabajo(tools=self.tools, buscador_rag=self.buscador_rag)
        self.transcriptor.precargar()
        # Precargar Ollama en hilo en segundo plano para no demorar la reconexión MQTT tras un corte de luz (sin-luz)
        threading.Thread(target=self.workflow_orden.llm.precargar, name="preload-ollama", daemon=True).start()

        self.recuperar_checkpoints_al_arrancar()

        if self.iniciar_ui:
            self._iniciar_servidor_ui()
        if self.iniciar_worker_analisis:
            self._iniciar_hilo_analisis()

        log.info("Componentes de dominio, voz, persistencia, RAG, STT, LangGraph y UI inicializados.")

    def _iniciar_servidor_ui(self) -> None:
        app = crear_app_humana(
            repo=self.repo,
            publicar_orden_fn=self.publicar_orden,
            obtener_ts_fn=self.obtener_ts_planta,
        )

        def _run_uvicorn() -> None:
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=settings.api_ui_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            server.run()

        self._hilo_ui = threading.Thread(target=_run_uvicorn, name="ui-humana-fastapi", daemon=True)
        self._hilo_ui.start()
        log.info("Interfaz Humana FastAPI iniciada en http://0.0.0.0:%d", settings.api_ui_port)

    def _iniciar_hilo_analisis(self) -> None:
        def _loop_analisis() -> None:
            while not self._detener_event.is_set():
                self._evento_nuevo_analisis.wait(timeout=5.0)
                self._evento_nuevo_analisis.clear()
                if self._detener_event.is_set():
                    break
                try:
                    enlace_ok = self.verificar_enlace()
                    self.drenar_cola_analisis(enlace_ok=enlace_ok)
                except Exception as exc:
                    log.exception("Error en hilo de análisis de órdenes: %s", exc)

        self._hilo_analisis = threading.Thread(target=_loop_analisis, name="worker-analisis", daemon=True)
        self._hilo_analisis.start()

    def recuperar_checkpoints_al_arrancar(self) -> dict[str, int]:
        """Recuperación durable tras reinicio o apagón (`sin-luz`, criterio F3).

        - Conserva intactas las órdenes en `PendienteAprobacion` (nunca retroceden a `Borrador` ni se duplican).
        - Señala al worker que reanude cualquier ítem pendiente en `cola_eventos_analisis`.
        """
        ordenes = self.repo.listar_ordenes()
        pendientes_aprob = [r for r in ordenes if r.orden.estado == EstadoOrden.pendiente]
        ckpts_abiertos = self.repo.listar_checkpoints(solo_no_completados=True)
        cola_pend = self.repo.obtener_eventos_analisis_pendientes(enlace_ok=True)

        if pendientes_aprob or ckpts_abiertos or cola_pend:
            log.info(
                "Recuperación durable tras reinicio: %d órdenes totales (%d en PendienteAprobacion), "
                "%d checkpoints activos/suspendidos, %d eventos en cola de análisis.",
                len(ordenes),
                len(pendientes_aprob),
                len(ckpts_abiertos),
                len(cola_pend),
            )
            for reg in pendientes_aprob:
                log.info(
                    "Orden sobreviviente preservada -> id=%s estado=%s equipo=%s tareas=%s correlation_id=%s",
                    reg.orden.id,
                    reg.orden.estado.value,
                    reg.orden.equipo,
                    reg.orden.tareas,
                    reg.correlation_id,
                )
            if cola_pend:
                self._evento_nuevo_analisis.set()

        return {
            "ordenes_totales": len(ordenes),
            "pendientes_aprobacion": len(pendientes_aprob),
            "checkpoints_abiertos": len(ckpts_abiertos),
            "cola_analisis_pendiente": len(cola_pend),
        }

    def publicar_orden(self, orden: OrdenSistema) -> None:
        """Publica una `OrdenSistema` validada por contrato en `sistema/ordenes`."""
        self.mqtt.publicar(Topicos.ORDENES, orden)
        log.info(
            "Orden publicada -> id=%s estado=%s equipo=%s tareas=%s ejecutor=%s",
            orden.id,
            orden.estado.value,
            orden.equipo,
            orden.tareas,
            orden.ejecutor,
        )

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
            # Al restablecerse el enlace, despertamos el drenaje de fotos/análisis pendientes antes de cerrar la cola
            if self.repo.obtener_eventos_analisis_pendientes(enlace_ok=True):
                if self.iniciar_worker_analisis:
                    self._evento_nuevo_analisis.set()
                else:
                    self.drenar_cola_analisis(enlace_ok=True)

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

    # ------------------------------------------------------------------ Cola durable y ejecución del Workflow (C + D)
    def encolar_escalado_para_analisis(self, ctx: ContextoEscalado, enlace_ok: bool) -> str:
        """Persiste el evento escalado en la cola durable + checkpoint inicial por `correlation_id`."""
        corr_id = ctx.evento.id
        # Si ya existe una orden abierta para este equipo y llega un re-disparo sin fotos de sensor, evitamos saturar la cola
        if not ctx.fotos and ctx.evento.origen == "sensor" and ctx.evento.equipo:
            abiertas_eq = [
                r
                for r in self.repo.listar_ordenes()
                if r.orden.equipo.upper() == ctx.evento.equipo.upper()
                and r.orden.estado
                not in (EstadoOrden.rechazada, EstadoOrden.cerrada, EstadoOrden.certificada)
            ]
            if abiertas_eq:
                log.info(
                    "Evento sensor %s omitido de cola LLM: ya existe orden abierta %s (%s) en %s.",
                    ctx.evento.id,
                    abiertas_eq[0].orden.id,
                    abiertas_eq[0].orden.estado.value,
                    ctx.evento.equipo,
                )
                return corr_id

        requiere_enlace = bool(ctx.fotos)
        item = ItemColaAnalisis(
            evento_id=ctx.evento.id,
            correlation_id=corr_id,
            contexto_json=serializar_contexto_escalado(ctx),
            procesado=False,
            requiere_enlace=requiere_enlace,
        )
        self.repo.encolar_evento_analisis(item)

        etapa_inicial = "esperando_enlace" if (requiere_enlace and not enlace_ok) else "iniciado"
        estado_wf = "suspendido" if etapa_inicial == "esperando_enlace" else "activo"
        self.repo.guardar_checkpoint(
            CheckpointWorkflow(
                correlation_id=corr_id,
                evento_id=ctx.evento.id,
                etapa_actual=etapa_inicial,
                estado_workflow=estado_wf,
                payload_estado={
                    "equipo": ctx.evento.equipo,
                    "requiere_enlace": requiere_enlace,
                    "fotos": list(ctx.fotos),
                },
            )
        )

        if enlace_ok or not requiere_enlace:
            if self.iniciar_worker_analisis:
                self._evento_nuevo_analisis.set()
            else:
                self.drenar_cola_analisis(enlace_ok=enlace_ok)
        else:
            log.info(
                "Evento %s con %d foto(s) encolado hasta restablecer enlace (correlation_id=%s).",
                ctx.evento.id,
                len(ctx.fotos),
                corr_id,
            )
            # Notificamos de inmediato el incremento de pendientes durante el corte de enlace (E2)
            self.publicar_latido()
        return corr_id

    def drenar_cola_analisis(self, enlace_ok: bool) -> list[ResultadoWorkflowOrden]:
        """Procesa los ítems pendientes de `cola_eventos_analisis` con anti-duplicación y checkpoints."""
        if self.workflow_orden is None:
            return []

        resultados: list[ResultadoWorkflowOrden] = []
        with self._lock_analisis:
            pendientes = self.repo.obtener_eventos_analisis_pendientes(enlace_ok=enlace_ok)
            for item in pendientes:
                try:
                    ctx = deserializar_contexto_escalado(item.contexto_json)
                    # Si es un evento de sensor sin fotos y ya existe orden abierta para ese equipo, marcar procesado directo
                    if not ctx.fotos and ctx.evento.origen == "sensor" and ctx.evento.equipo:
                        ya_abierta = any(
                            r.orden.equipo.upper() == ctx.evento.equipo.upper()
                            and r.orden.estado
                            not in (EstadoOrden.rechazada, EstadoOrden.cerrada, EstadoOrden.certificada)
                            for r in self.repo.listar_ordenes()
                        )
                        if ya_abierta:
                            self.repo.marcar_evento_analisis(item.evento_id, procesado=True, error=None)
                            continue
                    res = self.workflow_orden.ejecutar(
                        contexto_escalado=ctx,
                        ts_decision=self.obtener_ts_planta(),
                        enlace_ok=enlace_ok,
                        correlation_id=item.correlation_id,
                    )
                    resultados.append(res)
                    self._persistir_resultado_workflow(ctx=ctx, res=res)
                    self.repo.marcar_evento_analisis(item.evento_id, procesado=True, error=None)
                except Exception as exc:
                    log.exception("Error procesando evento en cola %s: %s", item.evento_id, exc)
                    self.repo.marcar_evento_analisis(item.evento_id, procesado=False, error=str(exc)[:300])
        return resultados

    def _persistir_resultado_workflow(self, ctx: ContextoEscalado, res: ResultadoWorkflowOrden) -> None:
        """Guarda trazas LLM/VLM, aplica deduplicación E3/F3, persiste orden + checkpoint y publica MQTT."""
        orden_id_traza = res.orden_final.id if res.orden_final else None

        # 1. Trazabilidad auditable de cada llamada LLM y VLM
        for traza in res.trazas_llm:
            self.repo.registrar_traza_llm(
                correlation_id=res.correlation_id,
                orden_id=orden_id_traza,
                paso=traza.paso,
                modelo=traza.modelo,
                proveedor=traza.proveedor,
                herramientas=traza.herramientas_usadas,
                tokens_entrada=traza.tokens_entrada,
                tokens_salida=traza.tokens_salida,
                costo_usd=traza.costo_usd,
                latencia_ms=traza.latencia_ms,
                exito=traza.exito,
                error=traza.error,
            )
        for foto in res.fotos_analizadas:
            self.repo.registrar_traza_llm(
                correlation_id=res.correlation_id,
                orden_id=orden_id_traza,
                paso="analisis_foto_vlm",
                modelo=foto.modelo_usado,
                proveedor="nvidia_nim" if foto.modelo_usado not in ("ninguno", "heuristica_local") else "local",
                herramientas=["analizar_foto"],
                tokens_entrada=foto.tokens_entrada,
                tokens_salida=foto.tokens_salida,
                costo_usd=foto.costo_usd,
                latencia_ms=foto.latencia_ms,
                exito=(foto.estado_analisis == "analizada"),
                error=(foto.descripcion if foto.estado_analisis == "error" else None),
            )

        # 2. Casos sin orden (ej. F10 sensor congelado o con deriva)
        if not res.requiere_orden or res.orden_final is None:
            self.repo.guardar_checkpoint(
                CheckpointWorkflow(
                    correlation_id=res.correlation_id,
                    evento_id=ctx.evento.id,
                    orden_id=None,
                    etapa_actual="sin_orden",
                    estado_workflow="completado",
                    payload_estado={
                        "falla_principal": res.analisis.falla_principal if res.analisis else None,
                        "resumen": res.analisis.resumen_tecnico if res.analisis else "",
                    },
                )
            )
            log.info("Workflow %s finalizado sin generar orden (F10/no requiere orden).", res.correlation_id)
            return

        orden = res.orden_final

        # 3. Anti-duplicación (E3) y supervivencia a corte (F3): nunca duplicar ni retroceder una orden abierta
        existente = self.repo.buscar_orden_abierta(orden.equipo, orden.tareas)
        if existente is not None:
            nuevos_evs = list(existente.orden.evento_ids)
            if ctx.evento.id not in nuevos_evs:
                nuevos_evs.append(ctx.evento.id)
                existente.orden = existente.orden.model_copy(update={"evento_ids": nuevos_evs})
                self.repo.guardar_orden(existente)
            self.repo.guardar_checkpoint(
                CheckpointWorkflow(
                    correlation_id=res.correlation_id,
                    evento_id=ctx.evento.id,
                    orden_id=existente.orden.id,
                    etapa_actual=existente.orden.estado.value.lower(),
                    estado_workflow="suspendido"
                    if existente.orden.estado == EstadoOrden.pendiente
                    else "activo",
                    payload_estado={"deduplicada_con": existente.orden.id},
                )
            )
            log.info(
                "Orden ya abierta encontrada (%s, estado=%s) para %s %s: se evita duplicado.",
                existente.orden.id,
                existente.orden.estado.value,
                orden.equipo,
                orden.tareas,
            )
            return

        # 4. Persistir orden nueva + checkpoint durable y publicar en sistema/ordenes
        reg = RegistroOrdenTrabajo(
            orden=orden,
            correlation_id=res.correlation_id,
            citas_validadas=res.citas_validadas,
            dano_estructural_grave=res.dano_estructural_grave,
            analisis_resumen=res.analisis.model_dump(mode="json") if res.analisis else {},
            fotos_analizadas=[f.model_dump(mode="json") for f in res.fotos_analizadas],
        )
        self.repo.guardar_orden(reg)

        es_pendiente = orden.estado == EstadoOrden.pendiente
        self.repo.guardar_checkpoint(
            CheckpointWorkflow(
                correlation_id=res.correlation_id,
                evento_id=ctx.evento.id,
                orden_id=orden.id,
                etapa_actual="pendiente_aprobacion" if es_pendiente else "aprobada",
                estado_workflow="suspendido" if es_pendiente else "activo",
                payload_estado={
                    "equipo": orden.equipo,
                    "tareas": list(orden.tareas),
                    "estado_orden": orden.estado.value,
                    "citas_validadas": list(res.citas_validadas),
                },
            )
        )
        self.publicar_orden(orden)

    # ------------------------------------------------------------------ Flujo A y B: Nota de Recorrida
    def manejar_nota_recorrida(self, payload: dict) -> None:
        t0 = time.monotonic()
        nota = NotaRecorrida.model_validate(payload)
        assert self.detector is not None and self.procesador_voz is not None

        # 1. Descargar audio WAV y transcribir localmente (100% offline)
        texto = ""
        if nota.audio_url:
            try:
                wav_bytes = self.api.descargar_bytes(nota.audio_url)
                texto = self.transcriptor.transcribir_wav_bytes(wav_bytes)
                log.info("Nota %s transcrita: %r", nota.id, texto)
            except Exception as e:
                log.error("Error transcribiendo nota %s: %s", nota.id, e)

        # 2. Evaluar inmediatamente Condición Insegura (S1 a S4) en código puro (< 30s reales, sin nube)
        res_seg = self.detector.evaluar_seguridad_nota(
            nota_id=nota.id,
            ts=nota.ts,
            texto_nota=texto,
        )
        if res_seg.es_inseguro and res_seg.evento and res_seg.aviso_tablet:
            _ = rutear_mensaje(Topicos.RECORRIDAS, es_condicion_insegura=True)
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
            # Si la alerta de seguridad adjunta foto (ej. S1 rotura de descarga en fuga-sin-enlace),
            # encolamos análisis multimodal + orden (siempre PendienteAprobacion) para procesar cuando haya enlace
            if nota.fotos and res_seg.codigo == "S1":
                ctx_seg = ContextoEscalado(
                    evento=res_seg.evento,
                    filas_parte=[],
                    fotos=list(nota.fotos),
                    motivos_anomalia=[res_seg.descripcion or "rotura con proyección de agua a presión"],
                )
                self.encolar_escalado_para_analisis(ctx_seg, enlace_ok=enlace_ok)
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
        """Una vez que el parte pasa a Confirmado, evalúa las reglas en código, publica EventoSistema
        y dispara el workflow de análisis + orden para todo evento `Escalado`."""
        assert self.detector is not None
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
            if ev.tipo == TipoEvento.escalado:
                if not enlace_ok:
                    # Publicar estado inmediato para reflejar pendientes_sincronizar > 0 durante cortes breves (E2)
                    self.publicar_latido()
                self.encolar_escalado_para_analisis(ctx, enlace_ok=enlace_ok)

    # ------------------------------------------------------------------ Flujo B/C: Sensores, Energía y Cierres
    def manejar_lectura_sensor(self, payload: dict) -> None:
        if self.detector is None:
            return
        try:
            lectura = LecturaSensor.model_validate(payload)
            ctx = self.detector.evaluar_lectura_sensor(lectura)
            if ctx is not None:
                enlace_ok = self.verificar_enlace()
                self.repo.guardar_evento(ctx.evento, sincronizado=enlace_ok)
                self.mqtt.publicar(Topicos.EVENTOS, ctx.evento)
                log.info("Desvío sostenido en sensor detectado: %s", ctx.evento.id)
                self.encolar_escalado_para_analisis(ctx, enlace_ok=enlace_ok)
        except Exception as e:
            log.debug("Error evaluando sensor: %s", e)

    def manejar_estado_energia(self, payload: dict) -> None:
        try:
            ev_en = EstadoEnergia.model_validate(payload)
            anterior = self._hay_energia
            self._hay_energia = ev_en.hay_energia
            if not ev_en.hay_energia:
                log.warning("Corte de energía reportado en planta (%s).", ev_en.ts.isoformat())
            elif not anterior and ev_en.hay_energia:
                log.info("Energía restablecida en planta (%s). Verificando colas pendientes.", ev_en.ts.isoformat())
                self._evento_nuevo_analisis.set()
        except Exception as e:
            log.debug("Error leyendo planta/energia: %s", e)

    def manejar_nota_cierre(self, payload: dict) -> None:
        """Bloque G: transcribe `NotaCierre`, compara contra la orden aprobada, persiste
        diferencias/certificación sugerida y pasa la orden a `Ejecutada` (criterio G1)."""
        cierre = NotaCierre.model_validate(payload)
        reg = self.repo.obtener_orden(cierre.orden_id)
        if reg is None:
            log.warning("NotaCierre %s para orden desconocida %s.", cierre.id, cierre.orden_id)
            return

        # 1. Transcribir audio de cierre localmente (100% offline con faster-whisper)
        texto_cierre = ""
        if cierre.audio_url:
            try:
                wav_bytes = self.api.descargar_bytes(cierre.audio_url)
                texto_cierre = self.transcriptor.transcribir_wav_bytes(wav_bytes)
            except Exception as exc:
                log.error("Error transcribiendo cierre %s: %s", cierre.id, exc)

        # 2. Comparar cierre contra la orden y calcular certificación contractual (CT-SVP-2025 / CT-SVP-2022)
        eval_cierre = evaluar_cierre_y_certificacion(
            orden=reg.orden,
            texto_cierre=texto_cierre,
            fecha_inicio=reg.orden.ts,
        )

        # 3. Pasar orden a Ejecutada (G1) y dejar lista para Supervisor (propio) o Inspector (contratista) en UI :8090
        reg.orden = reg.orden.model_copy(update={"estado": EstadoOrden.ejecutada, "ts": cierre.ts})
        reg.cierre_datos = {
            **cierre.model_dump(mode="json"),
            "texto_transcripto": texto_cierre,
            "evaluacion": eval_cierre.model_dump(mode="json"),
        }
        reg.diferencias_cierre = list(eval_cierre.diferencias)
        self.repo.guardar_orden(reg)

        ckpt = self.repo.obtener_checkpoint(reg.correlation_id)
        if ckpt:
            ckpt.cierre_id = cierre.id
            ckpt.etapa_actual = "ejecutada"
            ckpt.estado_workflow = "suspendido"
            ckpt.payload_estado = {
                **ckpt.payload_estado,
                "cierre_id": cierre.id,
                "estado_sugerido": eval_cierre.estado_sugerido,
                "monto_habilita_sugerido_usd": eval_cierre.monto_habilita_usd,
                "diferencias": eval_cierre.diferencias,
            }
            self.repo.guardar_checkpoint(ckpt)

        self.publicar_orden(reg.orden)
        log.info(
            "Cierre %s procesado -> orden=%s estado=Ejecutada sugerido=%s diferencias=%d",
            cierre.id,
            reg.orden.id,
            eval_cierre.estado_sugerido,
            len(eval_cierre.diferencias),
        )

    def procesar_cola(self) -> None:
        while not self.cola_mensajes.empty():
            topico, payload = self.cola_mensajes.get_nowait()
            try:
                decision = rutear_mensaje(topico)
                if decision.ruta == RutaMensaje.parte:
                    self.manejar_nota_recorrida(payload)
                elif decision.ruta == RutaMensaje.respuesta_operador:
                    self.manejar_respuesta_operador(payload)
                elif decision.ruta == RutaMensaje.sensor:
                    self.manejar_lectura_sensor(payload)
                elif decision.ruta == RutaMensaje.cierre:
                    self.manejar_nota_cierre(payload)
                elif decision.ruta == RutaMensaje.energia:
                    self.manejar_estado_energia(payload)
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
            self._detener_event.set()
            self._evento_nuevo_analisis.set()
            self.mqtt.detener()


if __name__ == "__main__":
    AgentePlanta().correr()
