"""Paso Revisor del workflow de órdenes de trabajo (Requisito 3.2).

Valida de forma determinista el borrador generado por el paso agéntico antes de
enviarlo a las Reglas de Aprobación en Código:
1. Verifica que cada cita documental exista en el corpus (`fragmentos.jsonl`) y
   esté vigente (`vigente == True`), rechazando citas inventadas o del manual
   reemplazado `TX350-C` (Rev. C).
2. Verifica que cada repuesto del borrador exista en `materiales.yaml` y pertenezca
   a la lista de materiales (BOM) del equipo objetivo. Traduce códigos obsoletos
   de Rev. C (`350-EMP-03` -> `TX-EMP-KIT`, etc.) y remueve repuestos ajenos al BOM.
3. Verifica que cada código de tarea sea válido en el catálogo (`tareas_estandar`,
   `tareas_no_estandar` o `"NO-ESTANDAR"`), normaliza el ejecutor (`propio` o
   `Servipat`) y completa los repuestos estándar del catálogo si faltaban.
"""

from __future__ import annotations

from typing import Any

from planta_contrato import OrdenSistema, Repuesto
from pydantic import BaseModel, ConfigDict, Field

from sistema.rag.buscador import BuscadorRAG

# Tabla de códigos obsoletos de TX-350 Rev. C -> Rev. D (TX350-D §10.4)
_MAPA_CODIGOS_REVC: dict[str, str] = {
    "350-EMP-03": "TX-EMP-KIT",
    "350-VAL-11": "TX-VAL-KIT",
    "350-ROD-02": "TX-ROD-CIG",
    "350-SEL-04": "TX-SELLO-CAR",
    "ACE-ISO150": "ACE-ISO220",
}


class ResultadoRevisionOrden(BaseModel):
    """Resultado auditable del paso Revisor."""

    model_config = ConfigDict(extra="forbid")

    valido_sin_cambios: bool
    citas_validadas: list[str] = Field(default_factory=list)
    citas_rechazadas: list[str] = Field(default_factory=list)
    observaciones: list[str] = Field(default_factory=list)
    orden_revisada: OrdenSistema


class RevisorOrden:
    """Verificador determinista de citas vigentes, BOM de repuestos y catálogo de tareas."""

    def __init__(self, materiales: dict[str, Any], buscador_rag: BuscadorRAG) -> None:
        self.materiales = materiales
        self.buscador_rag = buscador_rag
        self.repuestos_validos: set[str] = {str(r["codigo"]) for r in materiales.get("repuestos", [])}
        self.bom_por_equipo: dict[str, set[str]] = {
            str(eq): {str(c) for c in cods}
            for eq, cods in materiales.get("lista_materiales", {}).items()
        }
        self.tareas_estandar: dict[tuple[str, str], dict[str, Any]] = {}
        self.tareas_catalogo: dict[str, dict[str, Any]] = {}

        for t in materiales.get("tareas_estandar", []):
            cod = str(t["codigo"])
            self.tareas_catalogo[cod] = t
            for eq in t.get("equipos", []):
                self.tareas_estandar[(str(eq), cod)] = t

        for t in materiales.get("tareas_no_estandar", []):
            cod = str(t["codigo"])
            self.tareas_catalogo[cod] = t

    def revisar(
        self,
        orden_borrador: OrdenSistema,
        citas_propuestas: list[str],
        citas_contexto_rag: list[str] | None = None,
    ) -> ResultadoRevisionOrden:
        """Audita y sanea citas, tareas, ejecutor y repuestos contra el corpus vigente y materiales.yaml."""
        observaciones: list[str] = []
        citas_ok: list[str] = []
        citas_mal: list[str] = []

        # 1. Validación estricta de citas (existencia + vigencia)
        for cita in citas_propuestas:
            frag = self.buscador_rag.por_cita.get(cita)
            if frag is None:
                citas_mal.append(cita)
                observaciones.append(f"Cita inexistente descartada: '{cita}'.")
            elif not frag.get("vigente", False):
                citas_mal.append(cita)
                observaciones.append(f"Cita de documento no vigente rechazada (Rev. C): '{cita}'.")
                # Si existe el equivalente en TX350-D, sugerirlo si está vigente
                if cita.startswith("TX350-C "):
                    equiv = cita.replace("TX350-C ", "TX350-D ", 1)
                    if equiv in self.buscador_rag.por_cita and self.buscador_rag.por_cita[equiv].get("vigente"):
                        if equiv not in citas_ok:
                            citas_ok.append(equiv)
                            observaciones.append(f"Reemplazada cita vencida '{cita}' por vigente '{equiv}'.")
            else:
                if cita not in citas_ok:
                    citas_ok.append(cita)

        # Si quedaron 0 citas válidas pero el contexto RAG tenía citas vigentes, respaldar con las top-2 vigentes
        if not citas_ok and citas_contexto_rag:
            for c_ctx in citas_contexto_rag:
                frag = self.buscador_rag.por_cita.get(c_ctx)
                if frag and frag.get("vigente", False) and c_ctx not in citas_ok:
                    citas_ok.append(c_ctx)
                if len(citas_ok) >= 2:
                    break
            if citas_ok:
                observaciones.append("Sin citas válidas del modelo; respaldadas con citas vigentes del contexto RAG.")

        # 2. Validación de tareas contra catálogo
        equipo = orden_borrador.equipo
        tareas_limpias: list[str] = []
        for t in orden_borrador.tareas:
            t_norm = t.strip().upper()
            if t_norm in self.tareas_catalogo or t_norm == "NO-ESTANDAR":
                if t_norm not in tareas_limpias:
                    tareas_limpias.append(t_norm)
            else:
                observaciones.append(f"Tarea '{t}' no existe en el catálogo; reemplazada por 'NO-ESTANDAR'.")
                if "NO-ESTANDAR" not in tareas_limpias:
                    tareas_limpias.append("NO-ESTANDAR")

        if not tareas_limpias:
            tareas_limpias = ["I-VAL" if equipo.startswith("B-") else "NO-ESTANDAR"]
            observaciones.append(f"Orden sin tareas; asignada inspección/tarea por defecto {tareas_limpias[0]}.")

        # 3. Normalización de ejecutor según catálogo
        ejecutor = orden_borrador.ejecutor.strip() or "propio"
        for t in tareas_limpias:
            info_t = self.tareas_catalogo.get(t)
            if info_t and info_t.get("ejecutor") == "contratista":
                if ejecutor.lower() == "propio":
                    ejecutor = "Servipat"
                    observaciones.append(f"Tarea {t} requiere contratista: ejecutor ajustado a 'Servipat'.")

        # 4. Validación de repuestos contra inventario y BOM del equipo
        bom_eq = self.bom_por_equipo.get(equipo, set())
        repuestos_limpios: list[Repuesto] = []
        codigos_ya: set[str] = set()

        for rep in orden_borrador.repuestos:
            cod = rep.codigo.strip().upper()
            if cod in _MAPA_CODIGOS_REVC:
                nuevo_cod = _MAPA_CODIGOS_REVC[cod]
                observaciones.append(f"Repuesto obsoleto Rev. C '{cod}' actualizado a '{nuevo_cod}'.")
                cod = nuevo_cod
            if cod not in self.repuestos_validos:
                observaciones.append(f"Repuesto inexistente '{cod}' eliminado del borrador.")
                continue
            if bom_eq and cod not in bom_eq:
                observaciones.append(f"Repuesto '{cod}' fuera de la lista de materiales (BOM) de {equipo}: eliminado.")
                continue
            if rep.cantidad <= 0:
                observaciones.append(f"Cantidad inválida ({rep.cantidad}) para '{cod}': eliminada.")
                continue
            if cod not in codigos_ya:
                repuestos_limpios.append(Repuesto(codigo=cod, cantidad=float(rep.cantidad)))
                codigos_ya.add(cod)

        # 4b. Si TODAS las tareas son estándar del catálogo, restringir repuestos estrictamente a los de esas tareas
        # (defensa contra inyección de repuestos ajenos como TX-VAL-KIT sobre T-B-EMP)
        todas_estandar = bool(tareas_limpias) and all((equipo, t) in self.tareas_estandar for t in tareas_limpias)
        if todas_estandar:
            permitidos_por_tarea: dict[str, float] = {}
            for t in tareas_limpias:
                est = self.tareas_estandar[(equipo, t)]
                for r_cat in est.get("repuestos", []):
                    c_cat = str(r_cat["codigo"])
                    permitidos_por_tarea[c_cat] = permitidos_por_tarea.get(c_cat, 0.0) + float(r_cat["cantidad"])
            filtrados: list[Repuesto] = []
            for rep in repuestos_limpios:
                if rep.codigo in permitidos_por_tarea:
                    filtrados.append(rep)
                else:
                    observaciones.append(
                        f"Repuesto '{rep.codigo}' no corresponde a la(s) tarea(s) estándar {tareas_limpias}: eliminado."
                    )
            repuestos_limpios = filtrados
            codigos_ya = {r.codigo for r in repuestos_limpios}

        # Si el borrador quedó sin repuestos pero la tarea estándar (o T-B-ROD / T-B-ALI) lleva repuestos conocidos
        if not repuestos_limpios:
            for t in tareas_limpias:
                est = self.tareas_estandar.get((equipo, t))
                if est:
                    for r_cat in est.get("repuestos", []):
                        c_cat = str(r_cat["codigo"])
                        if c_cat not in codigos_ya and (not bom_eq or c_cat in bom_eq):
                            repuestos_limpios.append(Repuesto(codigo=c_cat, cantidad=float(r_cat["cantidad"])))
                            codigos_ya.add(c_cat)
                            observaciones.append(f"Repuesto estándar {c_cat} x{r_cat['cantidad']} completado desde catálogo.")
                elif t == "T-B-ROD" and equipo.startswith("B-"):
                    for c_rod in ("TX-ROD-CIG", "TX-SELLO-CAR"):
                        if c_rod not in codigos_ya and (not bom_eq or c_rod in bom_eq):
                            repuestos_limpios.append(Repuesto(codigo=c_rod, cantidad=1.0))
                            codigos_ya.add(c_rod)
                            observaciones.append(f"Repuesto {c_rod} x1 completado para T-B-ROD (TX350-D §9.7).")
                elif t == "T-B-ALI" and equipo.startswith("B-"):
                    if "ACO-ELAST-90" not in codigos_ya and (not bom_eq or "ACO-ELAST-90" in bom_eq):
                        repuestos_limpios.append(Repuesto(codigo="ACO-ELAST-90", cantidad=1.0))
                        codigos_ya.add("ACO-ELAST-90")
                        observaciones.append("Repuesto ACO-ELAST-90 x1 completado para T-B-ALI (TX350-D §9.8).")

        detiene_cat = any(bool(self.tareas_catalogo.get(t, {}).get("detiene", True)) for t in tareas_limpias)
        orden_revisada = orden_borrador.model_copy(
            update={
                "tareas": tareas_limpias,
                "ejecutor": ejecutor,
                "repuestos": repuestos_limpios,
                "detiene_equipo": detiene_cat,
            }
        )

        return ResultadoRevisionOrden(
            valido_sin_cambios=(len(observaciones) == 0),
            citas_validadas=citas_ok,
            citas_rechazadas=citas_mal,
            observaciones=observaciones,
            orden_revisada=orden_revisada,
        )
