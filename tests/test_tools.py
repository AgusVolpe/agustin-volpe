"""Tests automatizados para cada una de las 6 tools, validación estricta Pydantic y el servidor MCP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sistema.integraciones.mcp_server import (
    mcp,
    mcp_consultar_catalogo_tareas,
    mcp_consultar_inventario_y_bom,
)
from sistema.integraciones.planta_api import PlantaAPIClient, PlantaAPIError
from sistema.integraciones.tools import (
    BuscarManualesInput,
    ConsultarCatalogoTareasInput,
    ConsultarEquiposInput,
    ConsultarHistorialOrdenesInput,
    ConsultarHistorialSensoresInput,
    ConsultarInventarioInput,
    PlantaTools,
)

RUTAS_PLANTA_YAML = [
    Path("/opt/planta_datos/planta.yaml"),
    Path(__file__).resolve().parents[2] / "academia2026-tp" / "datos" / "planta.yaml",
]


class FakePlantaAPIClient(PlantaAPIClient):
    """Cliente simulado determinístico para tests unitarios rápidos sin requerir red activa."""

    def __init__(self):
        super().__init__(base_url="http://sim-fake:8100", timeout_s=0.5, max_intentos=2)
        ruta = next(r for r in RUTAS_PLANTA_YAML if r.exists())
        self._planta_data = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    def obtener_planta(self, usar_cache: bool = True) -> dict[str, Any]:
        return self._planta_data

    def obtener_historian(
        self, tag: str, variable: str, desde: str | None = None, hasta: str | None = None
    ) -> list[dict[str, Any]]:
        if tag == "B-102" and variable == "vibracion":
            return [
                {"ts": f"2026-11-03T0{i // 10}:{i % 10}0:00-03:00", "valor": 3.0 + (0.2 * i)}
                for i in range(12)
            ]
        return []

    def obtener_historial_ordenes(
        self, equipo: str | None = None, desde: str | None = None, hasta: str | None = None
    ) -> list[dict[str, Any]]:
        ordenes = [
            {
                "id": "OT-25-0010",
                "fecha_aviso": "2025-05-10T10:00:00-03:00",
                "equipo": "B-101",
                "tarea": "T-B-EMP",
                "falla": "F01",
                "nota": "Cambio de empaquetadura por goteo",
                "repuestos": ["TX-EMP-KIT"],
                "horas": 4.0,
                "origen": "archivo",
            },
            {
                "id": "OT-25-0022",
                "fecha_aviso": "2025-06-15T14:00:00-03:00",
                "equipo": "F-201B",
                "tarea": "T-F-CAR",
                "falla": "F07",
                "nota": "Cambio de cartuchos 10 micrones",
                "repuestos": ["FC-CART-10"],
                "horas": 3.0,
                "origen": "archivo",
            },
        ]
        if equipo:
            ordenes = [o for o in ordenes if o["equipo"] == equipo]
        return ordenes


@pytest.fixture
def tools() -> PlantaTools:
    return PlantaTools(api_client=FakePlantaAPIClient())


def test_pydantic_estricto_rechaza_campos_extra():
    """Verifica que las 6 entradas de tools tengan extra='forbid'."""
    for schema_cls, kwargs_base in [
        (ConsultarEquiposInput, {}),
        (ConsultarHistorialSensoresInput, {"tag": "B-101", "variable": "vibracion"}),
        (ConsultarHistorialOrdenesInput, {}),
        (ConsultarInventarioInput, {}),
        (ConsultarCatalogoTareasInput, {}),
        (BuscarManualesInput, {"consulta": "vibracion"}),
    ]:
        with pytest.raises(ValidationError):
            schema_cls(**kwargs_base, campo_no_permitido="invalido")


def test_tool_1_consultar_equipos_nombres_informales_y_ambiguedad(tools: PlantaTools):
    # Nombre informal exacto: "la dos" -> B-102 con manómetro en psi
    res = tools.consultar_equipos(ConsultarEquiposInput(consulta="la dos"))
    assert not res.ambiguo
    assert len(res.equipos) == 1
    assert res.equipos[0].tag == "B-102"
    assert res.equipos[0].unidad_manometro_local == "psi"

    # Nombre informal de filtro: "el de la izquierda" -> F-201A
    res_f = tools.consultar_equipos(ConsultarEquiposInput(consulta="el de la izquierda"))
    assert len(res_f.equipos) == 1
    assert res_f.equipos[0].tag == "F-201A"

    # Ambigüedad: "la bomba" a secas o expresiones genéricas como "bomba triplex" o "filtro de cartuchos"
    for consulta_ambigua, esperados in [
        ("la bomba", {"B-101", "B-102", "B-103"}),
        ("una bomba triplex", {"B-101", "B-102", "B-103"}),
        ("el filtro de cartuchos", {"F-201A", "F-201B"}),
    ]:
        res_amb = tools.consultar_equipos(ConsultarEquiposInput(consulta=consulta_ambigua))
        assert res_amb.ambiguo is True
        assert {e.tag for e in res_amb.equipos} == esperados
        assert res_amb.pregunta_aclaracion is not None
        assert len(res_amb.opciones_aclaracion) == len(esperados)

    # Equipo indeterminado (no coincide con nada): también pide aclaración y ofrece las opciones de la planta
    res_indet = tools.consultar_equipos(ConsultarEquiposInput(consulta="el compresor del galpón"))
    assert res_indet.ambiguo is True
    assert res_indet.equipos == []
    assert res_indet.pregunta_aclaracion is not None
    assert "especificar" in res_indet.pregunta_aclaracion.lower()
    assert len(res_indet.opciones_aclaracion) == 7

    # En cambio, si hay un único equipo de esa categoría (ej. "el caudalimetro" o "el tanque"), no es ambiguo
    res_unico = tools.consultar_equipos(ConsultarEquiposInput(consulta="el caudalímetro"))
    assert res_unico.ambiguo is False
    assert len(res_unico.equipos) == 1
    assert res_unico.equipos[0].tag == "FT-401"
    assert res_unico.pregunta_aclaracion is None


def test_tool_2_consultar_historial_sensores_rango_y_tendencia(tools: PlantaTools):
    res = tools.consultar_historial_sensores(
        ConsultarHistorialSensoresInput(tag="B-102", variable="vibracion", ultimas_muestras=12)
    )
    assert res.cantidad_muestras == 12
    assert res.minimo_historico == 3.0
    assert res.maximo_historico == 5.2
    assert res.tendencia == "subiendo"


def test_tool_3_consultar_historial_ordenes(tools: PlantaTools):
    res = tools.consultar_historial_ordenes(ConsultarHistorialOrdenesInput(equipo="B-101"))
    assert res.total_encontradas == 1
    assert res.ordenes[0].id == "OT-25-0010"
    assert res.ordenes[0].tarea == "T-B-EMP"


def test_tool_4_consultar_inventario_y_bom(tools: PlantaTools):
    res = tools.consultar_inventario_y_bom(ConsultarInventarioInput(equipo="F-201B"))
    assert "FC-CART-10" in res.lista_materiales_equipo
    cart_10 = next(r for r in res.repuestos if r.codigo == "FC-CART-10")
    assert cart_10.stock_yacimiento == 40
    assert cart_10.costo_usd == 17.0


def test_tool_5_consultar_catalogo_tareas(tools: PlantaTools):
    res_est = tools.consultar_catalogo_tareas(
        ConsultarCatalogoTareasInput(equipo="B-101", solo_estandar=True)
    )
    codigos = {t.codigo for t in res_est.tareas}
    assert "T-B-EMP" in codigos
    assert "T-B-ROD" not in codigos  # No estándar

    res_todas = tools.consultar_catalogo_tareas(
        ConsultarCatalogoTareasInput(codigo_tarea="T-B-ROD", solo_estandar=False)
    )
    assert len(res_todas.tareas) == 1
    assert res_todas.tareas[0].estandar is False
    assert res_todas.tareas[0].ejecutor == "contratista"


def test_tool_6_buscar_en_manuales_excluye_rev_c_por_defecto(tools: PlantaTools):
    res = tools.buscar_en_manuales(
        BuscarManualesInput(consulta="límites de alarma vibración bomba triplex disparo", top_k=5)
    )
    assert len(res.resultados) > 0
    # Ningún resultado debe provenir de la revisión vencida TX350-C
    assert all(r.vigente is True for r in res.resultados)
    assert all(not r.cita.startswith("TX350-C") for r in res.resultados)
    assert any(r.cita.startswith(("TX350-D", "MBP-4")) for r in res.resultados)


def test_reintentos_y_manejo_de_error_en_planta_api():
    cliente_invalido = PlantaAPIClient(base_url="http://127.0.0.1:59999", timeout_s=0.1, max_intentos=2)
    with pytest.raises(PlantaAPIError):
        cliente_invalido.obtener_estado()


def test_mcp_server_expone_tools():
    res_inv = mcp_consultar_inventario_y_bom(equipo="B-101", codigo_repuesto="TX-EMP-KIT")
    assert res_inv["repuestos"][0]["codigo"] == "TX-EMP-KIT"
    assert res_inv["repuestos"][0]["stock_yacimiento"] == 3

    res_tar = mcp_consultar_catalogo_tareas(codigo_tarea="T-B-ACE", solo_estandar=True)
    assert res_tar["tareas"][0]["detiene"] is False
    assert mcp.name == "planta-pia3-mcp"
