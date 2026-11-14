"""Tests unitarios y de evaluación para RAG (`IndexadorRAG`, `BuscadorRAG`) y `ContextBuilderAnomalias`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from planta_contrato import EventoSistema, FilaParte, TipoEvento

from evaluacion.evaluar_rag import cargar_suite_y_fragmentos, evaluar
from sistema.dominio.deteccion import ContextoEscalado
from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import PlantaTools
from sistema.procesamiento.analisis import ContextBuilderAnomalias
from sistema.rag.buscador import BuscadorRAG
from sistema.rag.indexador import IndexadorRAG, cargar_fragmentos_locales


def _cargar_planta_yaml() -> dict[str, Any]:
    for ruta in (
        Path("/opt/planta_datos/planta.yaml"),
        Path("../academia2026-tp/datos/planta.yaml"),
        Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/planta.yaml"),
    ):
        if ruta.exists():
            return yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return {}


def _cargar_materiales_yaml() -> dict[str, Any]:
    for ruta in (
        Path("/opt/planta_datos/materiales.yaml"),
        Path("../academia2026-tp/datos/materiales.yaml"),
        Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/materiales.yaml"),
    ):
        if ruta.exists():
            return yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return {}


@pytest.fixture(scope="module")
def buscador_instancia() -> BuscadorRAG:
    frags = cargar_fragmentos_locales()
    indexador = IndexadorRAG(fragmentos_crudos=frags)
    return BuscadorRAG(indexador=indexador)


def test_indexador_enriquece_metadatos_y_excluye_rev_c(buscador_instancia: BuscadorRAG) -> None:
    """Verifica que TX350-C quede marcado como no vigente y que las secciones de otras plantas no apliquen a PIA-3."""
    por_cita = buscador_instancia.por_cita
    assert "TX350-D §3.2" in por_cita
    assert por_cita["TX350-D §3.2"]["vigente"] is True
    assert por_cita["TX350-D §3.2"]["aplica_pia3"] is True

    assert "TX350-C §3.2" in por_cita
    assert por_cita["TX350-C §3.2"]["vigente"] is False
    assert por_cita["TX350-C §3.2"]["aplica_pia3"] is False

    # Sección de otra planta (PIA-1 / PIA-2)
    assert por_cita["MBP-4 §5.3"]["aplica_pia3"] is False
    assert por_cita["TX350-D §3.3"]["aplica_pia3"] is False


def test_buscador_rag_filtra_distractores_y_propaga_correspondencias(buscador_instancia: BuscadorRAG) -> None:
    """Verifica que ante una consulta de diagnóstico o límites se devuelvan TX350-D y MBP-4 sin TX350-C."""
    citas_lim = buscador_instancia.buscar("¿Cuál es la vibración de alarma y de disparo de la bomba?", k=5)
    assert "TX350-D §3.2" in citas_lim
    assert "MBP-4 §5.1" in citas_lim
    assert not any(c.startswith("TX350-C") for c in citas_lim)

    citas_diag = buscador_instancia.buscar(
        "golpeteo metálico en el cabezal y la aguja del manómetro de descarga oscila", k=5
    )
    assert "TX350-D §8.4" in citas_diag
    assert "MBP-4 §6.4" in citas_diag


def test_buscador_rag_recupera_historial_causa_raiz(buscador_instancia: BuscadorRAG) -> None:
    """Verifica que las consultas de historial recuperen las órdenes OT con hallazgos de causa raíz."""
    citas_b1 = buscador_instancia.buscar(
        "la uno vibra sin golpeteo y ya le cambiaron las válvulas, ¿qué fue las otras veces?", k=5
    )
    assert "historial OT-25-0024" in citas_b1 or "historial OT-25-0026" in citas_b1

    citas_b3 = buscador_instancia.buscar("B3 pierde por la empaquetadura otra vez, ¿qué se encontró antes?", k=5)
    assert any(ot in citas_b3 for ot in ("historial OT-25-0065", "historial OT-25-0088", "historial OT-26-0069"))


def test_evaluacion_suite_oficial_supera_baseline(buscador_instancia: BuscadorRAG) -> None:
    """Verifica que sobre la suite oficial de retrieval.json el recall@5 supere ampliamente el 0.66 del baseline y distractor@5 sea 0.0."""
    suite, _ = cargar_suite_y_fragmentos()
    if not suite:
        pytest.skip("Suite retrieval.json no encontrada en el entorno.")
    res = evaluar(buscador_instancia.buscar, suite, ks=(1, 3, 5, 10))
    totales = res["total"]
    assert totales["distractor@5"] == 0.0
    assert totales["recall@5"] >= 0.85
    assert totales["hit@5"] >= 0.90


def test_context_builder_anomalias_combina_tools_y_rag(buscador_instancia: BuscadorRAG) -> None:
    """Verifica que ContextBuilderAnomalias combine datos directos por Tools y citas por RAG."""
    api_mock = MagicMock(spec=PlantaAPIClient)
    api_mock.obtener_planta.return_value = _cargar_planta_yaml()
    api_mock.obtener_materiales.return_value = _cargar_materiales_yaml()
    api_mock.obtener_historian.return_value = [
        {"ts": "2025-10-01T08:00:00-03:00", "valor": 5.2},
        {"ts": "2025-10-01T09:00:00-03:00", "valor": 6.4},
        {"ts": "2025-10-01T10:00:00-03:00", "valor": 7.4},
    ]
    api_mock.obtener_historial_ordenes.return_value = [
        {
            "id": "OT-25-0026",
            "fecha": "2025-03-11",
            "equipo": "B-101",
            "tarea": "T-B-ALI",
            "nota": "alineacion laser contraeje-piñon",
        }
    ]
    tools = PlantaTools(api_client=api_mock)
    builder = ContextBuilderAnomalias(tools=tools, buscador_rag=buscador_instancia)

    evento = EventoSistema(
        id="EV-TEST-B-101",
        ts=datetime.now(timezone.utc),
        tipo=TipoEvento.escalado,
        equipo="B-101",
        origen="parte",
        parte_id="PARTE-TEST",
        nota_ids=["N-TEST"],
        resumen="Anomalía en B-101: vibracion=7.4 mm/s y golpeteo metálico en cabezal",
    )
    ctx_escalado = ContextoEscalado(
        evento=evento,
        filas_parte=[
            FilaParte(
                tag="B-101",
                variable="vibracion",
                valor=7.4,
                unidad="mm/s",
                observacion="golpeteo metálico en el cabezal y la aguja del manómetro oscila",
            )
        ],
        motivos_anomalia=["vibracion=7.4 mm/s >= alarma_alta (7.1)"],
    )

    paquete = builder.construir_contexto(equipo="B-101", contexto_escalado=ctx_escalado)
    assert paquete.equipo == "B-101"
    assert paquete.info_equipo_y_limites is not None
    assert paquete.info_equipo_y_limites.unidad_manometro_local == "kg/cm2"
    assert len(paquete.tendencias_sensores) >= 1
    assert len(paquete.inventario_y_bom) > 0
    assert len(paquete.tareas_aplicables) > 0
    assert len(paquete.secciones_manuales_rag) > 0
    assert any(c.startswith(("TX350-D", "MBP-4")) for c in paquete.citas_listas)
