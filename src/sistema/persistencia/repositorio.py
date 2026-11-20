"""Esquema SQL y operaciones de lectura/escritura persistentes en PostgreSQL 16 (con fallback en memoria para tests).

Persiste de forma durable (sobrevive al apagón de `sin-luz`, criterio F3):
- `conversaciones_notas`: estado de la conversación de cada nota de recorrida en tablet.
- `eventos_planta`: eventos generados (SinNovedad, Escalado, AlertaSeguridad) y estado de sincronización.
- `ordenes_trabajo`: órdenes de trabajo con su `correlation_id`, citas validadas, análisis, cierre y diferencias.
- `checkpoints_workflow`: checkpoint por `correlation_id` uniendo `evento -> orden -> cierre`.
- `cola_eventos_analisis`: eventos pendientes de análisis (o con fotos encoladas durante corte de enlace).
- `trazas_llm`: trazabilidad auditable de modelos, herramientas, tokens, costo y latencia (requisito F).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from planta_contrato import EstadoOrden, EventoSistema, OrdenSistema

from sistema.config import settings

log = logging.getLogger(__name__)

ESTADOS_ORDEN_TERMINALES = {
    EstadoOrden.rechazada.value,
    EstadoOrden.cerrada.value,
    EstadoOrden.certificada.value,
}


@dataclass
class EstadoConversacionNota:
    nota_id: str
    recorrida_id: str
    ts_nota: str
    operador: str
    audio_url: str | None = None
    fotos: list[str] = field(default_factory=list)
    texto_transcripto: str = ""
    estado: str = "Borrador"  # "Borrador" | "Confirmado"
    intento: int = 0
    ultimo_mensaje_tablet_id: str | None = None
    filas: list[dict[str, Any]] = field(default_factory=list)
    ambiguedad_pendiente: dict[str, Any] | None = None


@dataclass
class RegistroOrdenTrabajo:
    """Registro completo de una orden persistida con trazabilidad y datos para la UI humana."""

    orden: OrdenSistema
    correlation_id: str
    citas_validadas: list[str] = field(default_factory=list)
    dano_estructural_grave: bool = False
    analisis_resumen: dict[str, Any] = field(default_factory=dict)
    fotos_analizadas: list[dict[str, Any]] = field(default_factory=list)
    cierre_datos: dict[str, Any] | None = None
    diferencias_cierre: list[str] = field(default_factory=list)


@dataclass
class CheckpointWorkflow:
    """Checkpoint persistido de un workflow de orden vinculado por correlation_id."""

    correlation_id: str
    evento_id: str
    orden_id: str | None = None
    cierre_id: str | None = None
    etapa_actual: str = "iniciado"  # iniciado | esperando_enlace | pendiente_aprobacion | aprobada | ejecutada | cerrada | certificada | rechazada | observada
    estado_workflow: str = "activo"  # activo | suspendido | completado
    payload_estado: dict[str, Any] = field(default_factory=dict)
    reintentos: int = 0


@dataclass
class ItemColaAnalisis:
    """Evento Escalado (o con fotos) encolado para análisis resiliente ante cortes de enlace o energía."""

    evento_id: str
    correlation_id: str
    contexto_json: dict[str, Any]
    procesado: bool = False
    requiere_enlace: bool = False
    intentos: int = 0
    ultimo_error: str | None = None


class RepositorioPlanta:
    """Repositorio durable en PostgreSQL 16 con respaldo automático en memoria para tests unitarios."""

    def __init__(self, database_url: str | None = None, usar_memoria: bool = False):
        raw_url = database_url or settings.database_url
        self.conninfo = raw_url.replace("postgresql+psycopg://", "postgresql://")
        self.usar_memoria = usar_memoria

        # Almacenamiento en memoria (usado en tests unitarios o si Postgres no está disponible)
        self._mem_conversaciones: dict[str, EstadoConversacionNota] = {}
        self._mem_msg_a_nota: dict[str, str] = {}
        self._mem_eventos: dict[str, dict[str, Any]] = {}
        self._mem_cola_pendientes: list[str] = []
        self._mem_ordenes: dict[str, RegistroOrdenTrabajo] = {}
        self._mem_checkpoints: dict[str, CheckpointWorkflow] = {}
        self._mem_cola_analisis: dict[str, ItemColaAnalisis] = {}
        self._mem_trazas_llm: list[dict[str, Any]] = []

    def inicializar_esquema(self) -> None:
        if self.usar_memoria:
            return
        try:
            with psycopg.connect(self.conninfo, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversaciones_notas (
                            nota_id TEXT PRIMARY KEY,
                            recorrida_id TEXT NOT NULL,
                            ts_nota TEXT NOT NULL,
                            operador TEXT NOT NULL,
                            audio_url TEXT,
                            fotos JSONB NOT NULL DEFAULT '[]'::jsonb,
                            texto_transcripto TEXT NOT NULL DEFAULT '',
                            estado TEXT NOT NULL DEFAULT 'Borrador',
                            intento INTEGER NOT NULL DEFAULT 0,
                            ultimo_mensaje_tablet_id TEXT,
                            filas JSONB NOT NULL DEFAULT '[]'::jsonb,
                            ambiguedad_pendiente JSONB,
                            actualizado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_conv_msg_tablet
                            ON conversaciones_notas(ultimo_mensaje_tablet_id);

                        CREATE TABLE IF NOT EXISTS eventos_planta (
                            evento_id TEXT PRIMARY KEY,
                            ts_evento TEXT NOT NULL,
                            tipo TEXT NOT NULL,
                            equipo TEXT,
                            origen TEXT NOT NULL,
                            nota_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                            resumen TEXT NOT NULL,
                            payload JSONB NOT NULL,
                            sincronizado BOOLEAN NOT NULL DEFAULT TRUE,
                            creado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS ordenes_trabajo (
                            orden_id TEXT PRIMARY KEY,
                            correlation_id TEXT NOT NULL,
                            ts_orden TEXT NOT NULL,
                            estado TEXT NOT NULL,
                            equipo TEXT NOT NULL,
                            tareas JSONB NOT NULL DEFAULT '[]'::jsonb,
                            detiene_equipo BOOLEAN NOT NULL DEFAULT TRUE,
                            reduce_inyeccion BOOLEAN NOT NULL DEFAULT FALSE,
                            ejecutor TEXT NOT NULL DEFAULT 'propio',
                            repuestos JSONB NOT NULL DEFAULT '[]'::jsonb,
                            motivo TEXT,
                            habilita TEXT,
                            monto_habilita_usd DOUBLE PRECISION,
                            evento_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                            citas_validadas JSONB NOT NULL DEFAULT '[]'::jsonb,
                            dano_estructural_grave BOOLEAN NOT NULL DEFAULT FALSE,
                            analisis_resumen JSONB NOT NULL DEFAULT '{}'::jsonb,
                            fotos_analizadas JSONB NOT NULL DEFAULT '[]'::jsonb,
                            cierre_datos JSONB,
                            diferencias_cierre JSONB NOT NULL DEFAULT '[]'::jsonb,
                            actualizado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_ordenes_equipo_estado
                            ON ordenes_trabajo(equipo, estado);
                        CREATE INDEX IF NOT EXISTS idx_ordenes_correlation
                            ON ordenes_trabajo(correlation_id);

                        CREATE TABLE IF NOT EXISTS checkpoints_workflow (
                            correlation_id TEXT PRIMARY KEY,
                            evento_id TEXT NOT NULL,
                            orden_id TEXT,
                            cierre_id TEXT,
                            etapa_actual TEXT NOT NULL,
                            estado_workflow TEXT NOT NULL DEFAULT 'activo',
                            payload_estado JSONB NOT NULL DEFAULT '{}'::jsonb,
                            reintentos INTEGER NOT NULL DEFAULT 0,
                            actualizado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS cola_eventos_analisis (
                            evento_id TEXT PRIMARY KEY,
                            correlation_id TEXT NOT NULL,
                            contexto_json JSONB NOT NULL,
                            procesado BOOLEAN NOT NULL DEFAULT FALSE,
                            requiere_enlace BOOLEAN NOT NULL DEFAULT FALSE,
                            intentos INTEGER NOT NULL DEFAULT 0,
                            ultimo_error TEXT,
                            creado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS trazas_llm (
                            id SERIAL PRIMARY KEY,
                            correlation_id TEXT NOT NULL,
                            orden_id TEXT,
                            paso TEXT NOT NULL,
                            modelo TEXT NOT NULL,
                            proveedor TEXT NOT NULL,
                            herramientas JSONB NOT NULL DEFAULT '[]'::jsonb,
                            tokens_entrada INTEGER NOT NULL DEFAULT 0,
                            tokens_salida INTEGER NOT NULL DEFAULT 0,
                            costo_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                            latencia_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                            exito BOOLEAN NOT NULL DEFAULT TRUE,
                            error TEXT,
                            creado_en TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
            log.info("Esquema PostgreSQL (Entrega 1 + Durabilidad Entrega 2) inicializado correctamente.")
        except Exception as e:
            log.warning("No se pudo conectar a PostgreSQL (%s). Usando almacenamiento en memoria.", e)
            self.usar_memoria = True

    # ------------------------------------------------------------------ Conversaciones de notas
    def guardar_conversacion(self, conv: EstadoConversacionNota) -> None:
        if self.usar_memoria:
            self._mem_conversaciones[conv.nota_id] = conv
            if conv.ultimo_mensaje_tablet_id:
                self._mem_msg_a_nota[conv.ultimo_mensaje_tablet_id] = conv.nota_id
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversaciones_notas (
                        nota_id, recorrida_id, ts_nota, operador, audio_url, fotos,
                        texto_transcripto, estado, intento, ultimo_mensaje_tablet_id,
                        filas, ambiguedad_pendiente, actualizado_en
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
                    ON CONFLICT (nota_id) DO UPDATE SET
                        texto_transcripto = EXCLUDED.texto_transcripto,
                        estado = EXCLUDED.estado,
                        intento = EXCLUDED.intento,
                        ultimo_mensaje_tablet_id = EXCLUDED.ultimo_mensaje_tablet_id,
                        filas = EXCLUDED.filas,
                        ambiguedad_pendiente = EXCLUDED.ambiguedad_pendiente,
                        actualizado_en = NOW();
                    """,
                    (
                        conv.nota_id,
                        conv.recorrida_id,
                        conv.ts_nota,
                        conv.operador,
                        conv.audio_url,
                        json.dumps(conv.fotos),
                        conv.texto_transcripto,
                        conv.estado,
                        conv.intento,
                        conv.ultimo_mensaje_tablet_id,
                        json.dumps(conv.filas, ensure_ascii=False),
                        json.dumps(conv.ambiguedad_pendiente, ensure_ascii=False)
                        if conv.ambiguedad_pendiente
                        else None,
                    ),
                )

    def obtener_conversacion_por_nota(self, nota_id: str) -> EstadoConversacionNota | None:
        if self.usar_memoria:
            return self._mem_conversaciones.get(nota_id)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT nota_id, recorrida_id, ts_nota, operador, audio_url, fotos,
                           texto_transcripto, estado, intento, ultimo_mensaje_tablet_id,
                           filas, ambiguedad_pendiente
                    FROM conversaciones_notas WHERE nota_id = %s
                    """,
                    (nota_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_a_conv(row)

    def obtener_conversacion_por_mensaje_tablet(self, mensaje_id: str) -> EstadoConversacionNota | None:
        if self.usar_memoria:
            nid = self._mem_msg_a_nota.get(mensaje_id)
            return self._mem_conversaciones.get(nid) if nid else None

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT nota_id, recorrida_id, ts_nota, operador, audio_url, fotos,
                           texto_transcripto, estado, intento, ultimo_mensaje_tablet_id,
                           filas, ambiguedad_pendiente
                    FROM conversaciones_notas WHERE ultimo_mensaje_tablet_id = %s
                    """,
                    (mensaje_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_a_conv(row)

    @staticmethod
    def _row_a_conv(row: tuple[Any, ...]) -> EstadoConversacionNota:
        return EstadoConversacionNota(
            nota_id=row[0],
            recorrida_id=row[1],
            ts_nota=row[2],
            operador=row[3],
            audio_url=row[4],
            fotos=row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]"),
            texto_transcripto=row[6] or "",
            estado=row[7] or "Borrador",
            intento=int(row[8] or 0),
            ultimo_mensaje_tablet_id=row[9],
            filas=row[10] if isinstance(row[10], list) else json.loads(row[10] or "[]"),
            ambiguedad_pendiente=row[11]
            if isinstance(row[11], dict) or row[11] is None
            else json.loads(row[11]),
        )

    # ------------------------------------------------------------------ Eventos y sincronización offline
    def guardar_evento(self, evento: EventoSistema, sincronizado: bool = True) -> None:
        payload_dict = json.loads(evento.model_dump_json())
        if self.usar_memoria:
            self._mem_eventos[evento.id] = payload_dict
            if not sincronizado and evento.id not in self._mem_cola_pendientes:
                self._mem_cola_pendientes.append(evento.id)
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eventos_planta (
                        evento_id, ts_evento, tipo, equipo, origen, nota_ids, resumen, payload, sincronizado
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
                    ON CONFLICT (evento_id) DO UPDATE SET
                        sincronizado = EXCLUDED.sincronizado;
                    """,
                    (
                        evento.id,
                        evento.ts.isoformat(),
                        evento.tipo.value,
                        evento.equipo,
                        evento.origen,
                        json.dumps(evento.nota_ids),
                        evento.resumen,
                        json.dumps(payload_dict, ensure_ascii=False),
                        sincronizado,
                    ),
                )

    def contar_pendientes_sincronizar(self) -> int:
        if self.usar_memoria:
            pend_cola = sum(1 for it in self._mem_cola_analisis.values() if not it.procesado)
            return len(self._mem_cola_pendientes) + pend_cola

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM eventos_planta WHERE sincronizado = FALSE;")
                r1 = cur.fetchone()
                cur.execute("SELECT COUNT(*) FROM cola_eventos_analisis WHERE procesado = FALSE;")
                r2 = cur.fetchone()
                return (int(r1[0]) if r1 else 0) + (int(r2[0]) if r2 else 0)

    def marcar_pendientes_sincronizados(self) -> int:
        """Cuando vuelve el enlace, marca sincronizados los eventos en cola."""
        if self.usar_memoria:
            n = len(self._mem_cola_pendientes)
            self._mem_cola_pendientes.clear()
            return n

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE eventos_planta SET sincronizado = TRUE WHERE sincronizado = FALSE;"
                )
                return cur.rowcount or 0

    # ------------------------------------------------------------------ Órdenes de trabajo (Entrega 2)
    def guardar_orden(self, reg: RegistroOrdenTrabajo) -> None:
        o = reg.orden
        if self.usar_memoria:
            self._mem_ordenes[o.id] = reg
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ordenes_trabajo (
                        orden_id, correlation_id, ts_orden, estado, equipo, tareas,
                        detiene_equipo, reduce_inyeccion, ejecutor, repuestos, motivo,
                        habilita, monto_habilita_usd, evento_ids, citas_validadas,
                        dano_estructural_grave, analisis_resumen, fotos_analizadas,
                        cierre_datos, diferencias_cierre, actualizado_en
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s,
                        %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s::jsonb,
                        %s::jsonb, %s::jsonb, NOW()
                    )
                    ON CONFLICT (orden_id) DO UPDATE SET
                        ts_orden = EXCLUDED.ts_orden,
                        estado = EXCLUDED.estado,
                        tareas = EXCLUDED.tareas,
                        detiene_equipo = EXCLUDED.detiene_equipo,
                        reduce_inyeccion = EXCLUDED.reduce_inyeccion,
                        ejecutor = EXCLUDED.ejecutor,
                        repuestos = EXCLUDED.repuestos,
                        motivo = EXCLUDED.motivo,
                        habilita = EXCLUDED.habilita,
                        monto_habilita_usd = EXCLUDED.monto_habilita_usd,
                        evento_ids = EXCLUDED.evento_ids,
                        citas_validadas = EXCLUDED.citas_validadas,
                        dano_estructural_grave = EXCLUDED.dano_estructural_grave,
                        analisis_resumen = EXCLUDED.analisis_resumen,
                        fotos_analizadas = EXCLUDED.fotos_analizadas,
                        cierre_datos = EXCLUDED.cierre_datos,
                        diferencias_cierre = EXCLUDED.diferencias_cierre,
                        actualizado_en = NOW();
                    """,
                    (
                        o.id,
                        reg.correlation_id,
                        o.ts.isoformat(),
                        o.estado.value,
                        o.equipo,
                        json.dumps(o.tareas),
                        o.detiene_equipo,
                        o.reduce_inyeccion,
                        o.ejecutor,
                        json.dumps([r.model_dump() for r in o.repuestos]),
                        o.motivo,
                        o.habilita,
                        o.monto_habilita_usd,
                        json.dumps(o.evento_ids),
                        json.dumps(reg.citas_validadas, ensure_ascii=False),
                        reg.dano_estructural_grave,
                        json.dumps(reg.analisis_resumen, ensure_ascii=False),
                        json.dumps(reg.fotos_analizadas, ensure_ascii=False),
                        json.dumps(reg.cierre_datos, ensure_ascii=False) if reg.cierre_datos is not None else None,
                        json.dumps(reg.diferencias_cierre, ensure_ascii=False),
                    ),
                )

    @staticmethod
    def _row_a_orden(row: tuple[Any, ...]) -> RegistroOrdenTrabajo:
        def _j(v: Any, default: Any) -> Any:
            if v is None:
                return default
            return v if isinstance(v, (list, dict)) else json.loads(v)

        orden = OrdenSistema(
            id=row[0],
            ts=datetime.fromisoformat(row[2]),
            estado=EstadoOrden(row[3]),
            equipo=row[4],
            tareas=_j(row[5], []),
            detiene_equipo=bool(row[6]),
            reduce_inyeccion=bool(row[7]),
            ejecutor=row[8],
            repuestos=_j(row[9], []),
            motivo=row[10],
            habilita=row[11],
            monto_habilita_usd=row[12],
            evento_ids=_j(row[13], []),
        )
        return RegistroOrdenTrabajo(
            orden=orden,
            correlation_id=row[1],
            citas_validadas=_j(row[14], []),
            dano_estructural_grave=bool(row[15]),
            analisis_resumen=_j(row[16], {}),
            fotos_analizadas=_j(row[17], []),
            cierre_datos=_j(row[18], None),
            diferencias_cierre=_j(row[19], []),
        )

    def obtener_orden(self, orden_id: str) -> RegistroOrdenTrabajo | None:
        if self.usar_memoria:
            return self._mem_ordenes.get(orden_id)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT orden_id, correlation_id, ts_orden, estado, equipo, tareas,
                           detiene_equipo, reduce_inyeccion, ejecutor, repuestos, motivo,
                           habilita, monto_habilita_usd, evento_ids, citas_validadas,
                           dano_estructural_grave, analisis_resumen, fotos_analizadas,
                           cierre_datos, diferencias_cierre
                    FROM ordenes_trabajo WHERE orden_id = %s
                    """,
                    (orden_id,),
                )
                row = cur.fetchone()
                return self._row_a_orden(row) if row else None

    def listar_ordenes(self, estado: str | None = None) -> list[RegistroOrdenTrabajo]:
        if self.usar_memoria:
            valores = list(self._mem_ordenes.values())
            if estado:
                valores = [r for r in valores if r.orden.estado.value == estado]
            return sorted(valores, key=lambda r: r.orden.ts)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                if estado:
                    cur.execute(
                        """
                        SELECT orden_id, correlation_id, ts_orden, estado, equipo, tareas,
                               detiene_equipo, reduce_inyeccion, ejecutor, repuestos, motivo,
                               habilita, monto_habilita_usd, evento_ids, citas_validadas,
                               dano_estructural_grave, analisis_resumen, fotos_analizadas,
                               cierre_datos, diferencias_cierre
                        FROM ordenes_trabajo WHERE estado = %s ORDER BY actualizado_en ASC
                        """,
                        (estado,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT orden_id, correlation_id, ts_orden, estado, equipo, tareas,
                               detiene_equipo, reduce_inyeccion, ejecutor, repuestos, motivo,
                               habilita, monto_habilita_usd, evento_ids, citas_validadas,
                               dano_estructural_grave, analisis_resumen, fotos_analizadas,
                               cierre_datos, diferencias_cierre
                        FROM ordenes_trabajo ORDER BY actualizado_en ASC
                        """
                    )
                return [self._row_a_orden(r) for r in cur.fetchall()]

    def buscar_orden_abierta(self, equipo: str, tareas: list[str]) -> RegistroOrdenTrabajo | None:
        """Busca una orden no terminal con el mismo equipo y conjunto de tareas (garantiza E3 y F3)."""
        objetivo = tuple(sorted(t.upper() for t in tareas))
        for reg in self.listar_ordenes():
            if reg.orden.estado.value in ESTADOS_ORDEN_TERMINALES:
                continue
            if reg.orden.equipo.upper() == equipo.upper() and tuple(sorted(t.upper() for t in reg.orden.tareas)) == objetivo:
                return reg
        return None

    # ------------------------------------------------------------------ Checkpoints (Durabilidad 3.3 / F3)
    def guardar_checkpoint(self, ckpt: CheckpointWorkflow) -> None:
        if self.usar_memoria:
            self._mem_checkpoints[ckpt.correlation_id] = ckpt
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO checkpoints_workflow (
                        correlation_id, evento_id, orden_id, cierre_id,
                        etapa_actual, estado_workflow, payload_estado, reintentos, actualizado_en
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, NOW())
                    ON CONFLICT (correlation_id) DO UPDATE SET
                        orden_id = COALESCE(EXCLUDED.orden_id, checkpoints_workflow.orden_id),
                        cierre_id = COALESCE(EXCLUDED.cierre_id, checkpoints_workflow.cierre_id),
                        etapa_actual = EXCLUDED.etapa_actual,
                        estado_workflow = EXCLUDED.estado_workflow,
                        payload_estado = EXCLUDED.payload_estado,
                        reintentos = EXCLUDED.reintentos,
                        actualizado_en = NOW();
                    """,
                    (
                        ckpt.correlation_id,
                        ckpt.evento_id,
                        ckpt.orden_id,
                        ckpt.cierre_id,
                        ckpt.etapa_actual,
                        ckpt.estado_workflow,
                        json.dumps(ckpt.payload_estado, ensure_ascii=False),
                        ckpt.reintentos,
                    ),
                )

    def obtener_checkpoint(self, correlation_id: str) -> CheckpointWorkflow | None:
        if self.usar_memoria:
            return self._mem_checkpoints.get(correlation_id)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT correlation_id, evento_id, orden_id, cierre_id,
                           etapa_actual, estado_workflow, payload_estado, reintentos
                    FROM checkpoints_workflow WHERE correlation_id = %s
                    """,
                    (correlation_id,),
                )
                r = cur.fetchone()
                if not r:
                    return None
                return CheckpointWorkflow(
                    correlation_id=r[0],
                    evento_id=r[1],
                    orden_id=r[2],
                    cierre_id=r[3],
                    etapa_actual=r[4],
                    estado_workflow=r[5],
                    payload_estado=r[6] if isinstance(r[6], dict) else json.loads(r[6] or "{}"),
                    reintentos=int(r[7] or 0),
                )

    def listar_checkpoints(self, solo_no_completados: bool = False) -> list[CheckpointWorkflow]:
        if self.usar_memoria:
            ckpts = list(self._mem_checkpoints.values())
            if solo_no_completados:
                ckpts = [c for c in ckpts if c.estado_workflow != "completado"]
            return ckpts

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                if solo_no_completados:
                    cur.execute(
                        """
                        SELECT correlation_id, evento_id, orden_id, cierre_id,
                               etapa_actual, estado_workflow, payload_estado, reintentos
                        FROM checkpoints_workflow WHERE estado_workflow != 'completado'
                        ORDER BY actualizado_en ASC
                        """
                    )
                else:
                    cur.execute(
                        """
                        SELECT correlation_id, evento_id, orden_id, cierre_id,
                               etapa_actual, estado_workflow, payload_estado, reintentos
                        FROM checkpoints_workflow ORDER BY actualizado_en ASC
                        """
                    )
                return [
                    CheckpointWorkflow(
                        correlation_id=r[0],
                        evento_id=r[1],
                        orden_id=r[2],
                        cierre_id=r[3],
                        etapa_actual=r[4],
                        estado_workflow=r[5],
                        payload_estado=r[6] if isinstance(r[6], dict) else json.loads(r[6] or "{}"),
                        reintentos=int(r[7] or 0),
                    )
                    for r in cur.fetchall()
                ]

    # ------------------------------------------------------------------ Cola de análisis resiliente
    def encolar_evento_analisis(self, item: ItemColaAnalisis) -> None:
        if self.usar_memoria:
            if item.evento_id not in self._mem_cola_analisis:
                self._mem_cola_analisis[item.evento_id] = item
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cola_eventos_analisis (
                        evento_id, correlation_id, contexto_json, procesado,
                        requiere_enlace, intentos, ultimo_error
                    ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (evento_id) DO NOTHING;
                    """,
                    (
                        item.evento_id,
                        item.correlation_id,
                        json.dumps(item.contexto_json, ensure_ascii=False),
                        item.procesado,
                        item.requiere_enlace,
                        item.intentos,
                        item.ultimo_error,
                    ),
                )

    def obtener_eventos_analisis_pendientes(self, enlace_ok: bool) -> list[ItemColaAnalisis]:
        if self.usar_memoria:
            return [
                it
                for it in self._mem_cola_analisis.values()
                if not it.procesado and (enlace_ok or not it.requiere_enlace) and it.intentos < 3
            ]

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                if enlace_ok:
                    cur.execute(
                        """
                        SELECT evento_id, correlation_id, contexto_json, procesado,
                               requiere_enlace, intentos, ultimo_error
                        FROM cola_eventos_analisis
                        WHERE procesado = FALSE AND intentos < 3
                        ORDER BY creado_en ASC
                        """
                    )
                else:
                    cur.execute(
                        """
                        SELECT evento_id, correlation_id, contexto_json, procesado,
                               requiere_enlace, intentos, ultimo_error
                        FROM cola_eventos_analisis
                        WHERE procesado = FALSE AND requiere_enlace = FALSE AND intentos < 3
                        ORDER BY creado_en ASC
                        """
                    )
                return [
                    ItemColaAnalisis(
                        evento_id=r[0],
                        correlation_id=r[1],
                        contexto_json=r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                        procesado=bool(r[3]),
                        requiere_enlace=bool(r[4]),
                        intentos=int(r[5] or 0),
                        ultimo_error=r[6],
                    )
                    for r in cur.fetchall()
                ]

    def marcar_evento_analisis(
        self, evento_id: str, procesado: bool, error: str | None = None
    ) -> None:
        if self.usar_memoria:
            it = self._mem_cola_analisis.get(evento_id)
            if it:
                it.procesado = procesado
                it.intentos += 1
                it.ultimo_error = error
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE cola_eventos_analisis
                    SET procesado = %s, intentos = intentos + 1, ultimo_error = %s
                    WHERE evento_id = %s
                    """,
                    (procesado, error, evento_id),
                )

    # ------------------------------------------------------------------ Trazas LLM (Requisito F)
    def registrar_traza_llm(
        self,
        correlation_id: str,
        paso: str,
        modelo: str,
        proveedor: str,
        herramientas: list[str] | None = None,
        tokens_entrada: int = 0,
        tokens_salida: int = 0,
        costo_usd: float = 0.0,
        latencia_ms: float = 0.0,
        exito: bool = True,
        error: str | None = None,
        orden_id: str | None = None,
    ) -> None:
        registro = {
            "correlation_id": correlation_id,
            "orden_id": orden_id,
            "paso": paso,
            "modelo": modelo,
            "proveedor": proveedor,
            "herramientas": list(herramientas or []),
            "tokens_entrada": tokens_entrada,
            "tokens_salida": tokens_salida,
            "costo_usd": costo_usd,
            "latencia_ms": latencia_ms,
            "exito": exito,
            "error": error,
        }
        if self.usar_memoria:
            self._mem_trazas_llm.append(registro)
            return

        with psycopg.connect(self.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trazas_llm (
                        correlation_id, orden_id, paso, modelo, proveedor, herramientas,
                        tokens_entrada, tokens_salida, costo_usd, latencia_ms, exito, error
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        correlation_id,
                        orden_id,
                        paso,
                        modelo,
                        proveedor,
                        json.dumps(registro["herramientas"]),
                        tokens_entrada,
                        tokens_salida,
                        costo_usd,
                        latencia_ms,
                        exito,
                        error,
                    ),
                )

    def listar_trazas_llm(self, correlation_id: str | None = None) -> list[dict[str, Any]]:
        if self.usar_memoria:
            if correlation_id:
                return [t for t in self._mem_trazas_llm if t["correlation_id"] == correlation_id]
            return list(self._mem_trazas_llm)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                if correlation_id:
                    cur.execute(
                        """
                        SELECT correlation_id, orden_id, paso, modelo, proveedor, herramientas,
                               tokens_entrada, tokens_salida, costo_usd, latencia_ms, exito, error
                        FROM trazas_llm WHERE correlation_id = %s ORDER BY id ASC
                        """,
                        (correlation_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT correlation_id, orden_id, paso, modelo, proveedor, herramientas,
                               tokens_entrada, tokens_salida, costo_usd, latencia_ms, exito, error
                        FROM trazas_llm ORDER BY id DESC LIMIT 100
                        """
                    )
                return [
                    {
                        "correlation_id": r[0],
                        "orden_id": r[1],
                        "paso": r[2],
                        "modelo": r[3],
                        "proveedor": r[4],
                        "herramientas": r[5] if isinstance(r[5], list) else json.loads(r[5] or "[]"),
                        "tokens_entrada": r[6],
                        "tokens_salida": r[7],
                        "costo_usd": r[8],
                        "latencia_ms": r[9],
                        "exito": r[10],
                        "error": r[11],
                    }
                    for r in cur.fetchall()
                ]
