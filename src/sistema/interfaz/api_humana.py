"""Interfaz humana propia (FastAPI + Panel Web en puerto 8090) para decisiones de Entrega 2.

Provee los canales de decisión humana que no forman parte del contrato MQTT de la planta:
- Planificador : Aprobar o Rechazar órdenes en estado `PendienteAprobacion`.
- Supervisor   : Cerrar órdenes de personal propio en estado `Ejecutada` -> `Cerrada`.
- Inspector    : Certificar (`Certificada` + `habilita` + `monto_habilita_usd`) u Observar
                 (`Observada` + `motivo`) órdenes de contratista en estado `Ejecutada`/`Observada`.

Cada transición actualiza de forma durable `ordenes_trabajo` y `checkpoints_workflow` en
PostgreSQL y publica inmediatamente la `OrdenSistema` actualizada en `sistema/ordenes`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from planta_contrato import EstadoOrden, OrdenSistema, Topicos
from pydantic import BaseModel, ConfigDict, Field

from sistema.persistencia.repositorio import (
    CheckpointWorkflow,
    RegistroOrdenTrabajo,
    RepositorioPlanta,
)
from sistema.procesamiento.cierres import calcular_monto_contrato

log = logging.getLogger(__name__)


class DecisionRechazoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motivo: str = Field(min_length=3, description="Motivo explícito del rechazo del planificador")


class DecisionObservacionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motivo: str = Field(min_length=3, description="Detalle de la diferencia observada a la contratista")


class DecisionCertificacionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    habilita: str | None = Field(default=None, description="Número HAB-AAAA-NNNN (se autogenera si se omite)")
    monto_habilita_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Monto calculado según orden y lista de precios CT-SVP-2025",
    )


def _serializar_registro(reg: RegistroOrdenTrabajo) -> dict[str, Any]:
    return {
        "orden": reg.orden.model_dump(mode="json"),
        "correlation_id": reg.correlation_id,
        "citas_validadas": reg.citas_validadas,
        "dano_estructural_grave": reg.dano_estructural_grave,
        "analisis_resumen": reg.analisis_resumen,
        "fotos_analizadas": reg.fotos_analizadas,
        "cierre_datos": reg.cierre_datos,
        "diferencias_cierre": reg.diferencias_cierre,
    }


def _calcular_monto_contrato_por_defecto(reg: RegistroOrdenTrabajo) -> float:
    # Si ya se calculó durante el procesamiento del cierre, priorizamos ese monto contractual
    ev = (reg.cierre_datos or {}).get("evaluacion") if isinstance(reg.cierre_datos, dict) else None
    if isinstance(ev, dict) and ev.get("monto_habilita_usd") is not None:
        return round(float(ev["monto_habilita_usd"]), 2)
    _, _, monto = calcular_monto_contrato(
        tareas=reg.orden.tareas,
        fecha_inicio=reg.orden.ts,
        equipo=reg.orden.equipo,
    )
    return monto


def _generar_numero_habilita(ts: datetime, contador: int) -> str:
    return f"HAB-{ts.year}-{contador:04d}"


HTML_PANEL = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PIA-3 · Panel de Decisiones Humanas</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
  h1 { margin: 0 0 6px; font-size: 1.5rem; color: #38bdf8; }
  p.sub { margin: 0 0 20px; color: #94a3b8; font-size: 0.9rem; }
  .toolbar { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
  button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button:hover { background: #334155; }
  button.ok { background: #15803d; border-color: #16a34a; }
  button.warn { background: #b45309; border-color: #d97706; }
  button.err { background: #b91c1c; border-color: #dc2626; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 14px; }
  .row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .badge { padding: 3px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
  .PendienteAprobacion { background: #854d0e; color: #fef08a; }
  .Aprobada { background: #166534; color: #bbf7d0; }
  .Rechazada { background: #7f1d1d; color: #fecaca; }
  .Ejecutada { background: #1d4ed8; color: #bfdbfe; }
  .Observada { background: #9a3412; color: #fed7aa; }
  .Cerrada, .Certificada { background: #0f766e; color: #99f6e4; }
  .meta { font-size: 0.85rem; color: #cbd5e1; margin-top: 8px; line-height: 1.5; }
  .actions { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
  .empty { color: #64748b; font-style: italic; padding: 24px; text-align: center; }
</style>
</head>
<body>
  <h1>Planta PIA-3 · Gestión Humana de Órdenes</h1>
  <p class="sub">Planificador (Aprobar / Rechazar) · Supervisor (Cerrar propio) · Inspector (Certificar / Observar contratista)</p>
  <div class="toolbar">
    <button onclick="cargar()">↻ Actualizar</button>
    <button onclick="filtrar('')">Todas</button>
    <button onclick="filtrar('PendienteAprobacion')">Pendientes (Planificador)</button>
    <button onclick="filtrar('Ejecutada')">Ejecutadas (Supervisor / Inspector)</button>
    <button onclick="filtrar('Observada')">Observadas</button>
  </div>
  <div id="lista"></div>
<script>
let filtroEstado = '';
function filtrar(e) { filtroEstado = e; cargar(); }
async function accion(id, path, body) {
  const res = await fetch(`/api/ordenes/${encodeURIComponent(id)}/${path}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : '{}'
  });
  const data = await res.json();
  if (!res.ok) { alert('Error: ' + (data.detail || JSON.stringify(data))); return; }
  cargar();
}
async function cargar() {
  const url = filtroEstado ? `/api/ordenes?estado=${encodeURIComponent(filtroEstado)}` : '/api/ordenes';
  const res = await fetch(url);
  const items = await res.json();
  const cont = document.getElementById('lista');
  if (!items.length) { cont.innerHTML = '<div class="empty">No hay órdenes en esta vista.</div>'; return; }
  cont.innerHTML = items.map(it => {
    const o = it.orden;
    const reps = (o.repuestos || []).map(r => `${r.codigo} x${r.cantidad}`).join(', ') || 'Sin repuestos';
    const citas = (it.citas_validadas || []).join(', ') || '—';
    const diffs = (it.diferencias_cierre || []).join(' | ');
    let btns = '';
    if (o.estado === 'PendienteAprobacion') {
      btns += `<button class="ok" onclick="accion('${o.id}','aprobar')">✓ Aprobar (Planificador)</button>`;
      btns += `<button class="err" onclick="const m=prompt('Motivo de rechazo:'); if(m) accion('${o.id}','rechazar',{motivo:m})">✕ Rechazar</button>`;
    }
    if (o.estado === 'Ejecutada' && o.ejecutor.toLowerCase() === 'propio') {
      btns += `<button class="ok" onclick="accion('${o.id}','cerrar')">✓ Cerrar Trabajo Propio (Supervisor)</button>`;
    }
    if ((o.estado === 'Ejecutada' || o.estado === 'Observada') && o.ejecutor.toLowerCase() !== 'propio') {
      btns += `<button class="ok" onclick="accion('${o.id}','certificar',{})">✓ Certificar y Emitir Habilita (Inspector)</button>`;
      btns += `<button class="warn" onclick="const m=prompt('Motivo de observación:'); if(m) accion('${o.id}','observar',{motivo:m})">⚠ Observar Cierre</button>`;
    }
    return `<div class="card">
      <div class="row">
        <strong>${o.id} · ${o.equipo} · [${(o.tareas||[]).join(', ')}]</strong>
        <span class="badge ${o.estado}">${o.estado}</span>
      </div>
      <div class="meta">
        <b>Correlation ID:</b> ${it.correlation_id} · <b>Ejecutor:</b> ${o.ejecutor} · <b>Detiene:</b> ${o.detiene_equipo} · <b>Reduce iny.:</b> ${o.reduce_inyeccion}<br/>
        <b>Repuestos:</b> ${reps}<br/>
        <b>Citas validadas:</b> ${citas}<br/>
        ${o.motivo ? `<b>Motivo:</b> ${o.motivo}<br/>` : ''}
        ${o.habilita ? `<b>Habilita:</b> ${o.habilita} (USD ${o.monto_habilita_usd})<br/>` : ''}
        ${diffs ? `<b>Diferencias en cierre:</b> ${diffs}<br/>` : ''}
      </div>
      <div class="actions">${btns}</div>
    </div>`;
  }).join('');
}
cargar();
</script>
</body>
</html>
"""


def crear_app_humana(
    repo: RepositorioPlanta,
    publicar_orden_fn: Callable[[OrdenSistema], None],
    obtener_ts_fn: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Construye la aplicación FastAPI conectada al repositorio y al publicador MQTT."""
    app = FastAPI(title="PIA-3 Interfaz Humana de Órdenes", version="2.0.0")
    ts_fn = obtener_ts_fn or (lambda: datetime.now(timezone.utc))

    def _actualizar_checkpoint(reg: RegistroOrdenTrabajo, etapa: str, estado_wf: str) -> None:
        ckpt = repo.obtener_checkpoint(reg.correlation_id)
        if ckpt is None:
            ckpt = CheckpointWorkflow(
                correlation_id=reg.correlation_id,
                evento_id=(reg.orden.evento_ids[0] if reg.orden.evento_ids else reg.correlation_id),
                orden_id=reg.orden.id,
                etapa_actual=etapa,
                estado_workflow=estado_wf,
                payload_estado={"estado_orden": reg.orden.estado.value},
            )
        else:
            ckpt.orden_id = reg.orden.id
            ckpt.etapa_actual = etapa
            ckpt.estado_workflow = estado_wf
            ckpt.payload_estado = {**ckpt.payload_estado, "estado_orden": reg.orden.estado.value}
        repo.guardar_checkpoint(ckpt)

    @app.get("/", response_class=HTMLResponse)
    def panel_web() -> str:
        return HTML_PANEL

    @app.get("/api/salud")
    def salud() -> dict[str, Any]:
        return {
            "ok": True,
            "ordenes_totales": len(repo.listar_ordenes()),
            "pendientes_aprobacion": len(repo.listar_ordenes(estado=EstadoOrden.pendiente.value)),
            "pendientes_sincronizar": repo.contar_pendientes_sincronizar(),
        }

    @app.get("/api/ordenes")
    def listar_ordenes(estado: str | None = None) -> list[dict[str, Any]]:
        return [_serializar_registro(r) for r in repo.listar_ordenes(estado=estado)]

    @app.get("/api/ordenes/{orden_id}")
    def obtener_orden(orden_id: str) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        return _serializar_registro(reg)

    @app.get("/api/checkpoints")
    def listar_checkpoints(solo_pendientes: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "correlation_id": c.correlation_id,
                "evento_id": c.evento_id,
                "orden_id": c.orden_id,
                "cierre_id": c.cierre_id,
                "etapa_actual": c.etapa_actual,
                "estado_workflow": c.estado_workflow,
                "reintentos": c.reintentos,
            }
            for c in repo.listar_checkpoints(solo_no_completados=solo_pendientes)
        ]

    @app.get("/api/trazas")
    def listar_trazas(correlation_id: str | None = None) -> list[dict[str, Any]]:
        return repo.listar_trazas_llm(correlation_id=correlation_id)

    # ------------------------------------------------------------------ Planificador: Aprobar / Rechazar
    @app.post("/api/ordenes/{orden_id}/aprobar")
    def aprobar_orden(orden_id: str) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        if reg.orden.estado != EstadoOrden.pendiente:
            raise HTTPException(
                status_code=409,
                detail=f"Solo se pueden aprobar órdenes en PendienteAprobacion (actual: {reg.orden.estado.value}).",
            )
        reg.orden = reg.orden.model_copy(
            update={"estado": EstadoOrden.aprobada, "ts": ts_fn(), "motivo": None}
        )
        repo.guardar_orden(reg)
        _actualizar_checkpoint(reg, etapa="aprobada", estado_wf="activo")
        publicar_orden_fn(reg.orden)
        log.info("Planificador APROBÓ orden %s (correlation_id=%s).", orden_id, reg.correlation_id)
        return _serializar_registro(reg)

    @app.post("/api/ordenes/{orden_id}/rechazar")
    def rechazar_orden(orden_id: str, datos: DecisionRechazoIn) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        if reg.orden.estado != EstadoOrden.pendiente:
            raise HTTPException(
                status_code=409,
                detail=f"Solo se pueden rechazar órdenes en PendienteAprobacion (actual: {reg.orden.estado.value}).",
            )
        reg.orden = reg.orden.model_copy(
            update={"estado": EstadoOrden.rechazada, "ts": ts_fn(), "motivo": datos.motivo.strip()}
        )
        repo.guardar_orden(reg)
        _actualizar_checkpoint(reg, etapa="rechazada", estado_wf="completado")
        publicar_orden_fn(reg.orden)
        log.info("Planificador RECHAZÓ orden %s: %s", orden_id, datos.motivo)
        return _serializar_registro(reg)

    # ------------------------------------------------------------------ Supervisor: Cerrar propio
    @app.post("/api/ordenes/{orden_id}/cerrar")
    def cerrar_orden_propia(orden_id: str) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        if reg.orden.ejecutor.strip().lower() != "propio":
            raise HTTPException(
                status_code=409,
                detail="Las órdenes de contratista no se cierran por supervisor: deben ser certificadas por el inspector.",
            )
        if reg.orden.estado != EstadoOrden.ejecutada:
            raise HTTPException(
                status_code=409,
                detail=f"Solo se pueden cerrar órdenes en estado Ejecutada (actual: {reg.orden.estado.value}).",
            )
        reg.orden = reg.orden.model_copy(update={"estado": EstadoOrden.cerrada, "ts": ts_fn()})
        repo.guardar_orden(reg)
        _actualizar_checkpoint(reg, etapa="cerrada", estado_wf="completado")
        publicar_orden_fn(reg.orden)
        log.info("Supervisor CERRÓ orden propia %s.", orden_id)
        return _serializar_registro(reg)

    # ------------------------------------------------------------------ Inspector: Certificar / Observar
    @app.post("/api/ordenes/{orden_id}/certificar")
    def certificar_orden_contratista(orden_id: str, datos: DecisionCertificacionIn) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        if reg.orden.ejecutor.strip().lower() == "propio":
            raise HTTPException(
                status_code=409,
                detail="Las órdenes de personal propio no llevan habilita ni certificación: usar /cerrar.",
            )
        if reg.orden.estado not in (EstadoOrden.ejecutada, EstadoOrden.observada):
            raise HTTPException(
                status_code=409,
                detail=f"Solo se pueden certificar órdenes Ejecutadas u Observadas (actual: {reg.orden.estado.value}).",
            )
        ahora = ts_fn()
        monto = (
            datos.monto_habilita_usd
            if datos.monto_habilita_usd is not None
            else _calcular_monto_contrato_por_defecto(reg)
        )
        n_cert = len(repo.listar_ordenes(estado=EstadoOrden.certificada.value)) + 1
        nro_hab = datos.habilita or _generar_numero_habilita(ahora, n_cert)

        reg.orden = reg.orden.model_copy(
            update={
                "estado": EstadoOrden.certificada,
                "ts": ahora,
                "habilita": nro_hab,
                "monto_habilita_usd": round(float(monto), 2),
                "motivo": None,
            }
        )
        repo.guardar_orden(reg)
        _actualizar_checkpoint(reg, etapa="certificada", estado_wf="completado")
        publicar_orden_fn(reg.orden)
        log.info("Inspector CERTIFICÓ orden %s -> %s (USD %.2f).", orden_id, nro_hab, monto)
        return _serializar_registro(reg)

    @app.post("/api/ordenes/{orden_id}/observar")
    def observar_orden_contratista(orden_id: str, datos: DecisionObservacionIn) -> dict[str, Any]:
        reg = repo.obtener_orden(orden_id)
        if reg is None:
            raise HTTPException(status_code=404, detail=f"Orden {orden_id} no encontrada.")
        if reg.orden.ejecutor.strip().lower() == "propio":
            raise HTTPException(
                status_code=409,
                detail="Solo se observan cierres de contratista.",
            )
        if reg.orden.estado != EstadoOrden.ejecutada:
            raise HTTPException(
                status_code=409,
                detail=f"Solo se pueden observar órdenes en estado Ejecutada (actual: {reg.orden.estado.value}).",
            )
        reg.orden = reg.orden.model_copy(
            update={"estado": EstadoOrden.observada, "ts": ts_fn(), "motivo": datos.motivo.strip()}
        )
        repo.guardar_orden(reg)
        _actualizar_checkpoint(reg, etapa="observada", estado_wf="suspendido")
        publicar_orden_fn(reg.orden)
        log.info("Inspector OBSERVÓ orden %s: %s", orden_id, datos.motivo)
        return _serializar_registro(reg)

    return app
