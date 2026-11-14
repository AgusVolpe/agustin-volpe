"""Esquema SQL y operaciones de lectura/escritura persistentes en PostgreSQL 16 (con fallback en memoria para tests).
Persiste:
- El estado de la conversación de cada nota (mensaje de tablet, nota, borrador, ambigüedad e intento).
- Los eventos generados (SinNovedad, Escalado, AlertaSeguridad).
- La cola de sincronización para cuando el enlace de planta cae (fuga-sin-enlace, tormenta-perfecta).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg
from planta_contrato import EventoSistema, FilaParte

from sistema.config import settings

log = logging.getLogger(__name__)


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


class RepositorioPlanta:
    """Repositorio de persistencia en PostgreSQL 16 con soporte automático en memoria para tests unitarios."""

    def __init__(self, database_url: str | None = None, usar_memoria: bool = False):
        raw_url = database_url or settings.database_url
        # psycopg3 usa postgresql:// en lugar de postgresql+psycopg://
        self.conninfo = raw_url.replace("postgresql+psycopg://", "postgresql://")
        self.usar_memoria = usar_memoria
        self._mem_conversaciones: dict[str, EstadoConversacionNota] = {}
        self._mem_msg_a_nota: dict[str, str] = {}
        self._mem_eventos: dict[str, dict[str, Any]] = {}
        self._mem_cola_pendientes: list[str] = []

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
                        """
                    )
            log.info("Esquema PostgreSQL inicializado correctamente.")
        except Exception as e:
            log.warning("No se pudo conectar a PostgreSQL (%s). Usando almacenamiento en memoria.", e)
            self.usar_memoria = True

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
            return len(self._mem_cola_pendientes)

        with psycopg.connect(self.conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM eventos_planta WHERE sincronizado = FALSE;")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def marcar_pendientes_sincronizados(self) -> int:
        """Cuando vuelve el enlace, procesa en orden lo pendiente de sincronizar y limpia la cola."""
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
