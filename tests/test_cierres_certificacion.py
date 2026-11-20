"""Tests unitarios del Bloque G (Paso 5 · Entrega 2):
- Parseo de notas de cierre por voz (`parsear_texto_cierre`).
- Validación 12/12 contra el dataset oficial `cierres_ejemplo.json` (CT-SVP-2025 y CT-SVP-2022).
- Integración con `AgentePlanta.manejar_nota_cierre` -> transición a `Ejecutada` (criterio G1).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from planta_contrato import EstadoOrden, NotaCierre, OrdenSistema, Repuesto

from sistema.agente import AgentePlanta
from sistema.persistencia.repositorio import (
    CheckpointWorkflow,
    RegistroOrdenTrabajo,
    RepositorioPlanta,
)
from sistema.procesamiento.cierres import (
    calcular_monto_contrato,
    evaluar_cierre_y_certificacion,
    parsear_texto_cierre,
)


def _cargar_cierres_ejemplo() -> list[dict[str, Any]]:
    for ruta in (
        Path("/opt/planta_datos/cierres_ejemplo.json"),
        Path("../academia2026-tp/datos/documentos/contrato/cierres_ejemplo.json"),
        Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/documentos/contrato/cierres_ejemplo.json"),
    ):
        if ruta.exists():
            return json.loads(ruta.read_text(encoding="utf-8"))
    raise FileNotFoundError("No se encontró cierres_ejemplo.json en las rutas esperadas.")


def test_parseo_texto_cierre_con_repuestos_horas_y_prueba():
    texto = (
        "Cierro la orden OT B101 T B ROD 1A2B. Hice cambio de rodamientos del cigüeñal. "
        "Usé un juego de rodamientos de cigüeñal TX-350, dos juego de sellos de cárter TX-350. "
        "Probé a ciento diez kilos, sin pérdida. Fueron veintiséis coma cinco horas."
    )
    datos = parsear_texto_cierre(texto)
    assert "T-B-ROD" in datos.tareas_declaradas
    mapa = {r.codigo: r.cantidad for r in datos.repuestos_declarados}
    assert mapa["TX-ROD-CIG"] == 1.0
    assert mapa["TX-SELLO-CAR"] == 2.0
    assert datos.horas_declaradas == 26.5
    assert datos.prueba_resultado is not None and "sin perdida" in datos.prueba_resultado


def test_calculo_contrato_vigencia_2022_vs_2025():
    lista_24, item_24, monto_24 = calcular_monto_contrato(["T-B-ROD"], "2024-11-21T16:30-03:00", equipo="B-101")
    assert lista_24 == "CT-SVP-2022"
    assert item_24 == "SVP-01"
    assert monto_24 == 980.0

    lista_25, item_25, monto_25 = calcular_monto_contrato(["T-B-ROD"], "2025-02-10T08:00-03:00", equipo="B-101")
    assert lista_25 == "CT-SVP-2025"
    assert item_25 == "SVP-01"
    assert monto_25 == 1150.0


def test_validacion_completa_contra_los_12_casos_de_cierres_ejemplo_json():
    casos = _cargar_cierres_ejemplo()
    assert len(casos) == 12

    for caso in casos:
        orden = OrdenSistema(
            id=caso["orden_id"],
            ts=datetime.fromisoformat(caso["fecha_inicio"]),
            estado=EstadoOrden.aprobada,
            equipo=caso["equipo"],
            tareas=caso["tareas"],
            detiene_equipo=caso["tareas"] != ["I-VIB"],
            reduce_inyeccion=False,
            ejecutor="Servipat",
            repuestos=[Repuesto(codigo=r["codigo"], cantidad=r["cantidad"]) for r in caso["repuestos_orden"]],
        )
        res = evaluar_cierre_y_certificacion(
            orden=orden,
            texto_cierre=caso["cierre_texto"],
            fecha_inicio=caso["fecha_inicio"],
            horas_registro_ingreso=caso["horas_registro_ingreso"],
            horas_declaradas_override=caso["horas_declaradas"],
            repuestos_declarados_override=caso["repuestos_declarados"],
            monto_declarado_usd=caso["monto_declarado_usd"],
        )

        assert res.lista_aplicada == caso["lista_aplicada"], f"Lista incorrecta en {caso['orden_id']}"
        assert res.item_contrato == caso["item_contrato"], f"Item incorrecto en {caso['orden_id']}"
        assert abs((res.monto_habilita_usd or 0.0) - caso["monto_habilita_usd"]) <= 1.0, (
            f"Monto incorrecto en {caso['orden_id']}: {res.monto_habilita_usd} vs {caso['monto_habilita_usd']}"
        )
        assert res.estado_sugerido == caso["estado"], (
            f"Estado incorrecto en {caso['orden_id']}: {res.estado_sugerido} vs {caso['estado']} (diffs={res.diferencias})"
        )
        if caso["diferencias"]:
            assert len(res.diferencias) >= 1, f"Faltó detectar diferencia en {caso['orden_id']}"
        else:
            assert res.diferencias == [], f"Falsa diferencia en {caso['orden_id']}: {res.diferencias}"


def test_agente_manejar_nota_cierre_pasa_orden_a_ejecutada_y_detecta_desvios():
    repo = RepositorioPlanta(usar_memoria=True)
    agente = AgentePlanta(iniciar_ui=False, iniciar_worker_analisis=False)
    agente.repo = repo
    publicadas: list[OrdenSistema] = []
    agente.publicar_orden = lambda o: publicadas.append(o)  # type: ignore[method-assign]

    # Mock de descarga + transcripción de cierre con desvío de repuesto extra
    agente.api.descargar_bytes = lambda url: b"RIFFfake"  # type: ignore[method-assign]
    agente.transcriptor.transcribir_wav_bytes = (  # type: ignore[method-assign]
        lambda wav: (
            "Cierro la orden OT-B101-ALI. Hice alineación láser de acople. "
            "Usé 2 elastómero de acople tamaño 90. Probé a 110 kilos, sin pérdida. Fueron 5,5 horas."
        )
    )

    ts_ord = datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc)
    orden_aprob = OrdenSistema(
        id="OT-B101-ALI",
        ts=ts_ord,
        estado=EstadoOrden.aprobada,
        equipo="B-101",
        tareas=["T-B-ALI"],
        detiene_equipo=True,
        reduce_inyeccion=False,
        ejecutor="Servipat",
        repuestos=[Repuesto(codigo="ACO-ELAST-90", cantidad=1)],
        evento_ids=["EV-1"],
    )
    repo.guardar_orden(RegistroOrdenTrabajo(orden=orden_aprob, correlation_id="EV-1"))
    repo.guardar_checkpoint(CheckpointWorkflow(correlation_id="EV-1", evento_id="EV-1", orden_id="OT-B101-ALI"))

    cierre = NotaCierre(
        id="CIERRE-OT-B101-ALI",
        ts=datetime(2026, 11, 9, 18, 0, tzinfo=timezone.utc),
        orden_id="OT-B101-ALI",
        tecnico="Walter Huenchumil (Servipat)",
        empresa="Servipat",
        audio_url="/media/audios/cierre.wav",
        fotos=["/media/fotos/trabajo_terminado/01.jpg"],
    )
    agente.manejar_nota_cierre(cierre.model_dump(mode="json"))

    assert len(publicadas) == 1
    assert publicadas[0].id == "OT-B101-ALI"
    assert publicadas[0].estado == EstadoOrden.ejecutada

    reg = repo.obtener_orden("OT-B101-ALI")
    assert reg is not None
    assert reg.orden.estado == EstadoOrden.ejecutada
    assert any("ACO-ELAST-90" in d for d in reg.diferencias_cierre)
    assert reg.cierre_datos is not None
    assert reg.cierre_datos["evaluacion"]["estado_sugerido"] == "Observada"
    assert reg.cierre_datos["evaluacion"]["monto_habilita_usd"] == 780.0
