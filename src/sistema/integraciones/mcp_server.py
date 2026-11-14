"""Servidor Model Context Protocol (MCP) que expone las herramientas de acceso a datos de la planta."""

from __future__ import annotations

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import (
    BuscarManualesInput,
    ConsultarCatalogoTareasInput,
    ConsultarEquiposInput,
    ConsultarHistorialOrdenesInput,
    ConsultarHistorialSensoresInput,
    ConsultarInventarioInput,
    PlantaTools,
)

mcp = FastMCP("planta-pia3-mcp")
_tools = PlantaTools(PlantaAPIClient())


@mcp.tool()
def mcp_consultar_equipos(consulta: str | None = None) -> dict:
    """Consulta equipos de la planta, nombres informales, unidades de manómetros locales y límites de alarma."""
    entrada = ConsultarEquiposInput(consulta=consulta)
    return _tools.consultar_equipos(entrada).model_dump()


@mcp.tool()
def mcp_consultar_inventario_y_bom(equipo: str | None = None, codigo_repuesto: str | None = None) -> dict:
    """Consulta inventario de repuestos, stock en almacén del yacimiento/central, costos USD y BOM por equipo."""
    entrada = ConsultarInventarioInput(equipo=equipo, codigo_repuesto=codigo_repuesto)
    return _tools.consultar_inventario_y_bom(entrada).model_dump()


@mcp.tool()
def mcp_consultar_historial_sensores(
    tag: str, variable: str, desde: str | None = None, hasta: str | None = None, ultimas_muestras: int = 60
) -> dict:
    """Consulta el historial de un sensor en el historiador de la planta (rango histórico y tendencia)."""
    entrada = ConsultarHistorialSensoresInput(
        tag=tag, variable=variable, desde=desde, hasta=hasta, ultimas_muestras=ultimas_muestras
    )
    return _tools.consultar_historial_sensores(entrada).model_dump()


@mcp.tool()
def mcp_consultar_historial_ordenes(
    equipo: str | None = None, desde: str | None = None, hasta: str | None = None, limite: int = 20
) -> dict:
    """Consulta el historial de órdenes de trabajo de los últimos 2 años más las del simulador."""
    entrada = ConsultarHistorialOrdenesInput(equipo=equipo, desde=desde, hasta=hasta, limite=limite)
    return _tools.consultar_historial_ordenes(entrada).model_dump()


@mcp.tool()
def mcp_consultar_catalogo_tareas(
    equipo: str | None = None, codigo_tarea: str | None = None, solo_estandar: bool = False
) -> dict:
    """Consulta el catálogo de tareas estándar y no estándar del plan de mantenimiento."""
    entrada = ConsultarCatalogoTareasInput(
        equipo=equipo, codigo_tarea=codigo_tarea, solo_estandar=solo_estandar
    )
    return _tools.consultar_catalogo_tareas(entrada).model_dump()


@mcp.tool()
def mcp_buscar_en_manuales(consulta: str, equipo: str | None = None, top_k: int = 5) -> dict:
    """Busca en los manuales vigentes de la planta y devuelve las secciones relevantes con su cita."""
    entrada = BuscarManualesInput(consulta=consulta, equipo=equipo, top_k=top_k)
    return _tools.buscar_en_manuales(entrada).model_dump()


if __name__ == "__main__":
    mcp.run()
