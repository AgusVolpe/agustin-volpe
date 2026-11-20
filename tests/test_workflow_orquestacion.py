"""Tests unitarios y de integración para Paso 3 de Entrega 2:
- Router de mensajes (`rutear_mensaje`)
- Paso Revisor de citas vigentes y BOM (`RevisorOrden`)
- Análisis multimodal de fotos y cola sin enlace (`AnalizadorFotos`)
- Workflow explícito en LangGraph (`WorkflowOrdenTrabajo`) para F01, F03, F06 historial, F09, F10, S1 daño grave e inspección.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from planta_contrato import EstadoOrden, EventoSistema, FilaParte, OrdenSistema, Repuesto, TipoEvento, Topicos

from sistema.dominio.deteccion import ContextoEscalado
from sistema.integraciones.llm_cliente import (
    CausaProbableOut,
    ClienteLLMLocal,
    SalidaAnalisisLLM,
    TrazaInvocacionLLM,
)
from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import PlantaTools
from sistema.integraciones.vision import AnalizadorFotos
from sistema.orquestacion.revisor import RevisorOrden
from sistema.orquestacion.router import RutaMensaje, rutear_mensaje
from sistema.orquestacion.workflow_orden import WorkflowOrdenTrabajo
from sistema.rag.buscador import BuscadorRAG
from sistema.rag.indexador import IndexadorRAG, cargar_fragmentos_locales


def _cargar_yaml(nombre: str) -> dict[str, Any]:
    for ruta in (
        Path(f"/opt/planta_datos/{nombre}"),
        Path(__file__).resolve().parents[2] / "academia2026-tp" / "datos" / nombre,
        Path(f"D:/Cursos/Curso software agentico/academia2026-tp/datos/{nombre}"),
    ):
        if ruta.exists():
            return yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return {}


@pytest.fixture(scope="module")
def buscador() -> BuscadorRAG:
    return BuscadorRAG(indexador=IndexadorRAG(fragmentos_crudos=cargar_fragmentos_locales()))


@pytest.fixture
def materiales() -> dict[str, Any]:
    return _cargar_yaml("materiales.yaml")


@pytest.fixture
def api_mock(materiales: dict[str, Any]) -> MagicMock:
    api = MagicMock(spec=PlantaAPIClient)
    api.obtener_planta.return_value = _cargar_yaml("planta.yaml")
    api.obtener_materiales.return_value = materiales
    api.obtener_estado.return_value = {
        "bombas": [
            {"tag": "B-101", "estado": "marcha"},
            {"tag": "B-102", "estado": "marcha"},
            {"tag": "B-103", "estado": "parada"},
        ]
    }
    api.obtener_historian.return_value = [
        {"ts": "2026-11-05T08:00:00-03:00", "valor": 4.2},
        {"ts": "2026-11-05T09:00:00-03:00", "valor": 6.8},
    ]
    api.obtener_historial_ordenes.return_value = [
        {
            "id": "OT-25-0026",
            "fecha": "2025-03-11",
            "equipo": "B-101",
            "tarea": "T-B-ALI",
            "nota": "desalineado acople contraeje-piñon elastomero gastado",
        }
    ]
    return api


def _evento_escalado(equipo: str, resumen: str, fotos: list[str] | None = None, filas: list[FilaParte] | None = None) -> ContextoEscalado:
    ev = EventoSistema(
        id=f"EV-TEST-{equipo}",
        ts=datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc),
        tipo=TipoEvento.escalado,
        equipo=equipo,
        origen="parte",
        nota_ids=["NOTA-TEST"],
        resumen=resumen,
    )
    return ContextoEscalado(
        evento=ev,
        filas_parte=filas or [],
        fotos=fotos or [],
        motivos_anomalia=[resumen],
    )


def test_router_mensajes_por_topico_y_seguridad() -> None:
    assert rutear_mensaje(Topicos.RECORRIDAS, es_condicion_insegura=True).ruta == RutaMensaje.seguridad
    assert rutear_mensaje(Topicos.RECORRIDAS, es_condicion_insegura=False).ruta == RutaMensaje.parte
    assert rutear_mensaje("planta/sensores/B-101/vibracion").ruta == RutaMensaje.sensor
    assert rutear_mensaje(Topicos.RESPUESTAS).ruta == RutaMensaje.respuesta_operador
    assert rutear_mensaje(Topicos.CIERRES).ruta == RutaMensaje.cierre
    assert rutear_mensaje(Topicos.ENERGIA).ruta == RutaMensaje.energia


def test_revisor_rechaza_rev_c_citas_inventadas_y_repuestos_fuera_bom(
    materiales: dict[str, Any], buscador: BuscadorRAG
) -> None:
    revisor = RevisorOrden(materiales=materiales, buscador_rag=buscador)
    orden_sucia = OrdenSistema(
        id="OT-REV-001",
        ts=datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc),
        estado=EstadoOrden.borrador,
        equipo="B-101",
        evento_ids=["EV-1"],
        tareas=["T-B-EMP"],
        detiene_equipo=False,
        reduce_inyeccion=False,
        ejecutor="propio",
        repuestos=[
            Repuesto(codigo="350-EMP-03", cantidad=1),  # Código viejo Rev. C -> debe convertirse a TX-EMP-KIT
            Repuesto(codigo="FC-CART-25", cantidad=48),  # Fuera del BOM de B-101 -> debe eliminarse
            Repuesto(codigo="INVENTADO-99", cantidad=2),  # Inexistente -> debe eliminarse
        ],
    )
    res = revisor.revisar(
        orden_borrador=orden_sucia,
        citas_propuestas=["TX350-C §8.3", "MANUAL-FALSO §99", "MBP-4 §6.3"],
    )

    assert res.valido_sin_cambios is False
    assert "TX350-C §8.3" in res.citas_rechazadas
    assert "MANUAL-FALSO §99" in res.citas_rechazadas
    assert "TX350-D §8.3" in res.citas_validadas
    assert "MBP-4 §6.3" in res.citas_validadas
    assert [r.codigo for r in res.orden_revisada.repuestos] == ["TX-EMP-KIT"]
    assert res.orden_revisada.detiene_equipo is True


def test_analizador_fotos_encola_sin_enlace_y_detecta_dano_grave(api_mock: MagicMock) -> None:
    analizador = AnalizadorFotos(api_client=api_mock)

    res_off = analizador.analizar_foto("/media/fotos/rotura_descarga/04.jpg", enlace_ok=False)
    assert res_off.estado_analisis == "encolada_sin_enlace"

    res_grave = analizador.analizar_foto("/media/fotos/rotura_descarga/04.jpg", enlace_ok=True)
    assert res_grave.estado_analisis == "analizada"
    assert res_grave.dano_estructural_grave is True

    res_emp = analizador.analizar_foto("/media/fotos/empaquetadura/01.jpg", enlace_ok=True)
    assert res_emp.estado_analisis == "analizada"
    assert res_emp.dano_estructural_grave is False
    assert res_emp.categoria_detectada == "empaquetadura"


def test_workflow_f01_empaquetadura_aprueba_automaticamente(api_mock: MagicMock, buscador: BuscadorRAG) -> None:
    tools = PlantaTools(api_client=api_mock)
    wf = WorkflowOrdenTrabajo(
        tools=tools,
        buscador_rag=buscador,
        llm_cliente=ClienteLLMLocal(habilitar_ollama=False),
    )
    ctx = _evento_escalado(
        equipo="B-102",
        resumen="Goteo creciente por empaquetadura en B-102 (45 gotas/min)",
        fotos=["/media/fotos/empaquetadura/01.jpg"],
    )
    res = wf.ejecutar(ctx, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)

    assert res.requiere_orden is True
    assert res.pasos_ejecutados == [
        "nodo_contexto",
        "nodo_fotos",
        "nodo_analisis_y_borrador",
        "nodo_revisor",
        "nodo_reglas",
    ]
    assert res.orden_final is not None
    assert res.orden_final.equipo == "B-102"
    assert res.orden_final.tareas == ["T-B-EMP"]
    assert res.orden_final.estado == EstadoOrden.aprobada
    assert [r.codigo for r in res.orden_final.repuestos] == ["TX-EMP-KIT"]
    assert all(not c.startswith("TX350-C") for c in res.citas_validadas)


def test_workflow_f09_cavitacion_redirige_a_filtro_primario(api_mock: MagicMock, buscador: BuscadorRAG) -> None:
    tools = PlantaTools(api_client=api_mock)
    wf = WorkflowOrdenTrabajo(
        tools=tools,
        buscador_rag=buscador,
        llm_cliente=ClienteLLMLocal(habilitar_ollama=False),
    )
    ctx = _evento_escalado(
        equipo="B-102",
        resumen="B-102 hace ruido a piedras (cavitación) con presión de succión baja 0.8 kg/cm2",
    )
    res = wf.ejecutar(ctx, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)

    assert res.orden_final is not None
    assert res.orden_final.equipo == "F-201A"
    assert res.orden_final.tareas == ["T-F-CAR"]
    assert res.orden_final.repuestos == [Repuesto(codigo="FC-CART-25", cantidad=48.0)]
    assert res.orden_final.estado == EstadoOrden.aprobada


def test_workflow_f10_sensor_con_deriva_no_exige_orden(api_mock: MagicMock, buscador: BuscadorRAG) -> None:
    tools = PlantaTools(api_client=api_mock)
    wf = WorkflowOrdenTrabajo(
        tools=tools,
        buscador_rag=buscador,
        llm_cliente=ClienteLLMLocal(habilitar_ollama=False),
    )
    ctx = _evento_escalado(
        equipo="B-103",
        resumen="Presión de descarga clavada con valor plano, no coincide con el manómetro local (F10 deriva).",
    )
    res = wf.ejecutar(ctx, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)

    assert res.requiere_orden is False
    assert res.orden_final is None
    assert "nodo_reglas" not in res.pasos_ejecutados


def test_workflow_historial_b101_acople_y_dano_estructural_van_a_pendiente(
    api_mock: MagicMock, buscador: BuscadorRAG
) -> None:
    tools = PlantaTools(api_client=api_mock)
    wf = WorkflowOrdenTrabajo(
        tools=tools,
        buscador_rag=buscador,
        llm_cliente=ClienteLLMLocal(habilitar_ollama=False),
    )

    # 1. Historial B-101: vibración sin golpeteo -> F06 acople (T-B-ALI, Servipat) -> PendienteAprobacion
    ctx_ali = _evento_escalado(
        equipo="B-101",
        resumen="Vibración moderada 5.8 mm/s en B-101 sin golpeteo en válvulas.",
    )
    res_ali = wf.ejecutar(ctx_ali, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)
    assert res_ali.orden_final is not None
    assert res_ali.orden_final.tareas == ["T-B-ALI"]
    assert res_ali.orden_final.ejecutor == "Servipat"
    assert res_ali.orden_final.estado == EstadoOrden.pendiente
    assert res_ali.orden_final.motivo is not None

    # 2. Daño estructural grave en foto (B-101 en fuga-sin-enlace) -> PendienteAprobacion
    ctx_s1 = _evento_escalado(
        equipo="B-101",
        resumen="Se rompió algo en la descarga de la uno, sale agua a presión.",
        fotos=["/media/fotos/rotura_descarga/04.jpg"],
    )
    res_s1 = wf.ejecutar(ctx_s1, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)
    assert res_s1.dano_estructural_grave is True
    assert res_s1.orden_final is not None
    assert res_s1.orden_final.estado == EstadoOrden.pendiente
    assert "daño estructural grave" in (res_s1.orden_final.motivo or "").lower()


def test_workflow_con_llm_mockeado_pasa_por_revisor_y_sanea_errores(
    api_mock: MagicMock, buscador: BuscadorRAG
) -> None:
    """Verifica que si el LLM de Ollama propone cita vencida TX350-C y repuesto viejo, el Revisor lo corrige."""
    llm_mock = MagicMock(spec=ClienteLLMLocal)
    llm_mock.analizar_anomalia.return_value = (
        SalidaAnalisisLLM(
            equipo_objetivo="B-102",
            falla_principal="F01",
            requiere_orden=True,
            causas_probables=[
                CausaProbableOut(
                    codigo_falla="F01",
                    descripcion="Desgaste de empaquetadura",
                    evidencia="Goteo 55 gotas/min",
                    citas=["TX350-C §8.3"],
                )
            ],
            tareas_propuestas=["T-B-EMP"],
            ejecutor_propuesto="propio",
            repuestos_propuestos=[{"codigo": "350-EMP-03", "cantidad": 1}],
            citas_usadas=["TX350-C §8.3"],
            resumen_tecnico="Cambio de empaquetadura",
        ),
        TrazaInvocacionLLM(
            paso="analisis_anomalia",
            modelo="qwen2.5:3b",
            proveedor="ollama-local",
            tokens_entrada=420,
            tokens_salida=110,
            costo_usd=0.0,
            latencia_ms=950.0,
            exito=True,
        ),
    )

    tools = PlantaTools(api_client=api_mock)
    wf = WorkflowOrdenTrabajo(tools=tools, buscador_rag=buscador, llm_cliente=llm_mock)
    ctx = _evento_escalado(equipo="B-102", resumen="Goteo en empaquetadura B-102")
    res = wf.ejecutar(ctx, ts_decision=datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc), enlace_ok=True)

    assert res.orden_final is not None
    assert res.orden_final.estado == EstadoOrden.aprobada
    assert [r.codigo for r in res.orden_final.repuestos] == ["TX-EMP-KIT"]
    assert "TX350-D §8.3" in res.citas_validadas
    assert "TX350-C §8.3" not in res.citas_validadas
    assert len(res.trazas_llm) == 1
    assert res.trazas_llm[0].tokens_entrada == 420
