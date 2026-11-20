"""Reglas de aprobación automática de órdenes de trabajo en código puro (Bloque D).

Implementa la función pura `evaluar_reglas_aprobacion`, que constituye la ÚNICA
puerta autorizada de escritura para decidir si un borrador de `OrdenSistema` pasa
a `Aprobada` o queda en `PendienteAprobacion` (con motivo explícito).

Las 5 reglas normativas (MBP-4 §12.3) + regla de daño estructural grave en fotos:
1. Todas las tareas deben pertenecer al catálogo de tareas estándar de ese equipo.
2. La inyección se mantiene: si alguna tarea detiene una bomba en marcha, debe existir
   al menos otra bomba disponible en estado `parada` (reserva real).
3. Todos los repuestos deben figurar en la lista de materiales (BOM) del equipo y tener
   stock suficiente en el almacén del `yacimiento` (no cuenta el almacén `central`).
4. El costo total de repuestos debe ser estrictamente menor a USD 2.500 (`costo < 2500`).
5. El ejecutor debe ser `propio` (sin contratista externo).
6. Si el análisis de fotos detectó daño estructural grave, la orden va SIEMPRE a
   `PendienteAprobacion` para revisión humana obligatoria.
"""

from __future__ import annotations

from typing import Any

from planta_contrato import EstadoOrden, OrdenSistema
from pydantic import BaseModel, ConfigDict, Field

COSTO_MAX_APROBACION_AUTO_USD = 2500.0


class ResultadoReglasAprobacion(BaseModel):
    """Resultado estructurado y auditable de evaluar una orden contra las reglas en código."""

    model_config = ConfigDict(extra="forbid")

    aprobada: bool
    estado_resultante: EstadoOrden
    motivos_incumplimiento: list[str] = Field(default_factory=list)
    motivo_texto: str | None = None
    costo_repuestos_usd: float
    detiene_equipo: bool
    reduce_inyeccion: bool
    orden_decidida: OrdenSistema


def extraer_estado_bombas(estado_planta: dict[str, Any] | None) -> dict[str, str]:
    """Extrae un mapa `{tag: estado}` (ej. `{'B-101': 'marcha', 'B-103': 'parada'}`) desde `GET /api/estado`."""
    if not estado_planta:
        return {}
    bombas_raw = estado_planta.get("bombas")
    if isinstance(bombas_raw, dict):
        return {str(k): str(v) for k, v in bombas_raw.items()}
    if isinstance(bombas_raw, list):
        salida: dict[str, str] = {}
        for item in bombas_raw:
            if isinstance(item, dict) and item.get("tag"):
                salida[str(item["tag"])] = str(item.get("estado", ""))
        return salida
    return {}


def evaluar_reglas_aprobacion(
    orden_borrador: OrdenSistema,
    materiales: dict[str, Any],
    estado_bombas: dict[str, str] | None = None,
    dano_estructural_grave: bool = False,
    costo_max_usd: float = COSTO_MAX_APROBACION_AUTO_USD,
    almacen: str = "yacimiento",
) -> ResultadoReglasAprobacion:
    """Función pura que evalúa las 5 reglas de aprobación automática + daño estructural grave.

    Nunca depende de texto libre ni de instrucciones provenientes de un LLM o de notas de voz.
    Siempre devuelve una nueva instancia inmutable de `OrdenSistema` con estado `Aprobada`
    o `PendienteAprobacion` (con `motivo` obligatorio en este último caso).
    """
    equipo = orden_borrador.equipo
    bombas = estado_bombas or {}

    # Índices deterministas a partir de materiales.yaml
    repuestos_info: dict[str, dict[str, Any]] = {
        str(r["codigo"]): r for r in materiales.get("repuestos", [])
    }
    estandar_por_equipo: dict[str, set[str]] = {}
    detiene_por_tarea: dict[str, bool] = {}

    for t in materiales.get("tareas_estandar", []):
        cod = str(t["codigo"])
        detiene_por_tarea[cod] = bool(t.get("detiene", True))
        for eq in t.get("equipos", []):
            estandar_por_equipo.setdefault(str(eq), set()).add(cod)

    for t in materiales.get("tareas_no_estandar", []):
        cod = str(t["codigo"])
        detiene_por_tarea[cod] = bool(t.get("detiene", True))

    bom_equipo: set[str] = {str(c) for c in materiales.get("lista_materiales", {}).get(equipo, [])}

    fallan: list[str] = []

    # 1. Regla 1: Todas las tareas deben estar en el catálogo de tareas estándar para ese equipo
    tareas_estandar_eq = estandar_por_equipo.get(equipo, set())
    if not orden_borrador.tareas:
        fallan.append("sin tareas especificadas (requiere tarea estándar del catálogo)")
    else:
        no_estandar = [t for t in orden_borrador.tareas if t not in tareas_estandar_eq]
        if no_estandar:
            fallan.append(f"tarea no estándar para {equipo}: {', '.join(no_estandar)}")

    # 2. Regla 2: La inyección se mantiene (si detiene bomba en marcha, debe haber reserva en estado 'parada')
    detiene = any(detiene_por_tarea.get(t, True) for t in orden_borrador.tareas) if orden_borrador.tareas else orden_borrador.detiene_equipo
    hay_reserva = any(est == "parada" for tag, est in bombas.items() if tag != equipo)
    if bombas:
        reduce_inyeccion = bool(detiene and bombas.get(equipo) == "marcha" and not hay_reserva)
    else:
        reduce_inyeccion = bool(detiene and orden_borrador.reduce_inyeccion)

    if reduce_inyeccion:
        fallan.append("reduce la inyección: la bomba de reserva no está disponible")

    # 3. Regla 3: Cada repuesto debe estar en el BOM del equipo y tener stock suficiente en el almacén del yacimiento
    costo_total = 0.0
    for rep in orden_borrador.repuestos:
        info = repuestos_info.get(rep.codigo)
        if info is None or rep.codigo not in bom_equipo:
            fallan.append(f"repuesto {rep.codigo} fuera de la lista de materiales (BOM) de {equipo}")
            continue
        stock_disp = float((info.get("stock") or {}).get(almacen, 0))
        if rep.cantidad > stock_disp:
            fallan.append(
                f"sin stock suficiente en {almacen} de {rep.codigo} "
                f"(requiere {rep.cantidad:g}, disponible {stock_disp:g})"
            )
        costo_total += float(info.get("costo_usd", 0.0)) * float(rep.cantidad)

    # 4. Regla 4: Costo total estrictamente menor a USD 2.500 (2499 aprueba, 2500 va a pendiente)
    if costo_total >= costo_max_usd:
        fallan.append(f"costo de repuestos USD {costo_total:.2f} iguala o supera el límite de USD {costo_max_usd:.0f}")

    # 5. Regla 5: Ejecutor propio (sin contratista externo)
    if orden_borrador.ejecutor.strip().lower() != "propio":
        fallan.append(f"requiere contratista externo ({orden_borrador.ejecutor})")

    # 6. Regla de seguridad multimodal: daño estructural grave en fotos siempre va a revisión humana
    if dano_estructural_grave:
        fallan.append("daño estructural grave detectado en fotos: requiere revisión humana obligatoria")

    aprobada = len(fallan) == 0
    estado_final = EstadoOrden.aprobada if aprobada else EstadoOrden.pendiente
    motivo_final = None if aprobada else "; ".join(fallan)

    orden_decidida = orden_borrador.model_copy(
        update={
            "estado": estado_final,
            "detiene_equipo": detiene,
            "reduce_inyeccion": reduce_inyeccion,
            "motivo": motivo_final,
        }
    )

    return ResultadoReglasAprobacion(
        aprobada=aprobada,
        estado_resultante=estado_final,
        motivos_incumplimiento=fallan,
        motivo_texto=motivo_final,
        costo_repuestos_usd=round(costo_total, 2),
        detiene_equipo=detiene,
        reduce_inyeccion=reduce_inyeccion,
        orden_decidida=orden_decidida,
    )
