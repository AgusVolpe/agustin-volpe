"""Constructor de contexto (Context Builder) para el análisis de anomalías y órdenes (Sección 2.3).

Prepara el paquete completo de información que consumirá el agente planificador en la
Entrega 2 cuando un evento de planta es `Escalado`. Combina:
1. Consultas directas deterministas (Tools / API HTTP): límites oficiales de alarma y
   disparo, unidad del manómetro local, tendencia e historial de sensores de 45 días,
   lista de materiales (BOM), stock en tiempo real en el almacén del yacimiento y
   catálogo de tareas estándar/no estándar.
2. Recuperación híbrida (RAG con `BuscadorRAG`): secciones de diagnóstico y procedimientos
   de los manuales vigentes (`TX350-D`, `FC48-3`, `MBP-4`) y órdenes históricas similares
   (`historial OT-...`) con sus citas canónicas listas para usar.
"""

from __future__ import annotations

from planta_contrato import FilaParte
from pydantic import BaseModel, ConfigDict, Field

from sistema.dominio.deteccion import ContextoEscalado
from sistema.integraciones.tools import (
    ConsultarCatalogoTareasInput,
    ConsultarEquiposInput,
    ConsultarHistorialOrdenesInput,
    ConsultarHistorialSensoresInput,
    ConsultarHistorialSensoresOutput,
    ConsultarInventarioInput,
    EquipoOut,
    OrdenHistoricaOut,
    PlantaTools,
    RepuestoInventarioOut,
    TareaCatalogoOut,
)
from sistema.rag.buscador import BuscadorRAG


class CitaDocumentalContexto(BaseModel):
    """Fragmento recuperado por RAG con su cita canónica lista para fundamentar una orden."""

    model_config = ConfigDict(extra="forbid")

    cita: str
    documento: str
    titulo: str
    vigente: bool
    puntaje: float
    extracto: str


class ContextoAnalisisAnomalia(BaseModel):
    """Paquete estructurado de contexto para diagnosticar un evento Escalado."""

    model_config = ConfigDict(extra="forbid")

    equipo: str
    motivo_escalado: str
    filas_confirmadas: list[FilaParte] = Field(default_factory=list)
    # Datos obtenidos por consulta directa (API / Tools deterministas)
    info_equipo_y_limites: EquipoOut | None = None
    tendencias_sensores: list[ConsultarHistorialSensoresOutput] = Field(default_factory=list)
    inventario_y_bom: list[RepuestoInventarioOut] = Field(default_factory=list)
    tareas_aplicables: list[TareaCatalogoOut] = Field(default_factory=list)
    ultimas_ordenes_api: list[OrdenHistoricaOut] = Field(default_factory=list)
    # Datos obtenidos por RAG (Manuales vigentes + Casos similares en historial OT)
    consulta_rag_generada: str
    secciones_manuales_rag: list[CitaDocumentalContexto] = Field(default_factory=list)
    ordenes_similares_rag: list[CitaDocumentalContexto] = Field(default_factory=list)
    citas_listas: list[str] = Field(default_factory=list)


class ContextBuilderAnomalias:
    """Construye el contexto enriquecido ante un evento `Escalado` de PIA-3."""

    def __init__(self, tools: PlantaTools, buscador_rag: BuscadorRAG | None = None) -> None:
        self.tools = tools
        self.buscador_rag = buscador_rag or BuscadorRAG()

    def _armar_consulta_rag(self, equipo: str, contexto_escalado: ContextoEscalado) -> str:
        """Construye una consulta semántica y léxica a partir de las filas y el motivo."""
        partes: list[str] = [equipo, contexto_escalado.evento.resumen]
        partes.extend(contexto_escalado.motivos_anomalia)
        for fila in contexto_escalado.filas_parte:
            if fila.variable:
                partes.append(str(fila.variable))
            if fila.observacion:
                partes.append(str(fila.observacion))
        return " ".join(partes)

    def construir_contexto(
        self,
        equipo: str,
        contexto_escalado: ContextoEscalado,
        top_k_manuales: int = 4,
        top_k_historial: int = 3,
    ) -> ContextoAnalisisAnomalia:
        """Reúne datos deterministas por Tools y evidencia documental por RAG con citas."""
        tag = equipo.upper()

        # 1. Consultas directas (Límites oficiales, BOM/Stock, Tareas y Órdenes por API)
        eq_out = self.tools.consultar_equipos(ConsultarEquiposInput(consulta=tag))
        info_equipo = eq_out.equipos[0] if eq_out.equipos else None

        inv_out = self.tools.consultar_inventario_y_bom(ConsultarInventarioInput(equipo=tag))
        tareas_out = self.tools.consultar_catalogo_tareas(
            ConsultarCatalogoTareasInput(equipo=tag, solo_estandar=False)
        )
        ordenes_api = self.tools.consultar_historial_ordenes(
            ConsultarHistorialOrdenesInput(equipo=tag, limite=5)
        )

        # 2. Tendencia de sensores de las variables involucradas
        variables_a_consultar: set[str] = set()
        for f in contexto_escalado.filas_parte:
            if f.variable and f.variable in (
                "presion_succion",
                "presion_descarga",
                "temperatura_aceite",
                "vibracion",
                "presion_diferencial",
                "caudal",
                "nivel",
            ):
                variables_a_consultar.add(f.variable)
        if not variables_a_consultar and info_equipo:
            for lim in info_equipo.sensores:
                variables_a_consultar.add(lim.variable)

        tendencias: list[ConsultarHistorialSensoresOutput] = []
        for var in sorted(variables_a_consultar):
            try:
                hist_s = self.tools.consultar_historial_sensores(
                    ConsultarHistorialSensoresInput(tag=tag, variable=var, ultimas_muestras=10)
                )
                tendencias.append(hist_s)
            except Exception:
                continue

        # 3. Recuperación RAG: secciones de manuales vigentes + órdenes históricas similares
        consulta_rag = self._armar_consulta_rag(tag, contexto_escalado)
        frag_manuales = self.buscador_rag.buscar_fragmentos(
            pregunta=f"diagnostico falla procedimiento {consulta_rag}",
            k=top_k_manuales + 4,
            filtro_equipo=tag,
            solo_vigentes=True,
        )
        secciones_manuales: list[CitaDocumentalContexto] = []
        for fm in frag_manuales:
            if fm["doc"] == "historial":
                continue
            secciones_manuales.append(
                CitaDocumentalContexto(
                    cita=fm["cita"],
                    documento=fm["doc"],
                    titulo=fm["titulo"],
                    vigente=fm["vigente"],
                    puntaje=fm["puntaje"],
                    extracto=fm["texto"][:450],
                )
            )
            if len(secciones_manuales) >= top_k_manuales:
                break

        frag_hist = self.buscador_rag.buscar_fragmentos(
            pregunta=f"que se encontro antes las otras veces {consulta_rag}",
            k=top_k_historial + 4,
            filtro_equipo=tag,
            solo_vigentes=True,
        )
        ordenes_similares: list[CitaDocumentalContexto] = []
        for fh in frag_hist:
            if fh["doc"] != "historial":
                continue
            ordenes_similares.append(
                CitaDocumentalContexto(
                    cita=fh["cita"],
                    documento=fh["doc"],
                    titulo=fh["titulo"],
                    vigente=fh["vigente"],
                    puntaje=fh["puntaje"],
                    extracto=fh["texto"][:450],
                )
            )
            if len(ordenes_similares) >= top_k_historial:
                break

        citas_listas: list[str] = []
        for item in secciones_manuales + ordenes_similares:
            if item.cita not in citas_listas:
                citas_listas.append(item.cita)

        motivo_str = "; ".join(contexto_escalado.motivos_anomalia) or contexto_escalado.evento.resumen
        return ContextoAnalisisAnomalia(
            equipo=tag,
            motivo_escalado=motivo_str,
            filas_confirmadas=list(contexto_escalado.filas_parte),
            info_equipo_y_limites=info_equipo,
            tendencias_sensores=tendencias,
            inventario_y_bom=inv_out.repuestos,
            tareas_aplicables=tareas_out.tareas,
            ultimas_ordenes_api=ordenes_api.ordenes,
            consulta_rag_generada=consulta_rag,
            secciones_manuales_rag=secciones_manuales,
            ordenes_similares_rag=ordenes_similares,
            citas_listas=citas_listas,
        )
