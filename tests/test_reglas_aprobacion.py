"""Tests unitarios y de casos límite para las Reglas de Aprobación en Código (Bloque D).

Cubre explícitamente los casos límite requeridos por el Anexo Entrega 2:
- Costo límite exacto: USD 2.499 (aprueba) vs. USD 2.500 (pendiente).
- T-B-VAL (USD 1.900) con reserva y stock en yacimiento (aprueba).
- TX-ROD-CIG sin stock en el yacimiento (pendiente).
- Reserva no disponible (B-103 en mantenimiento o disparada) para tarea que detiene (pendiente).
- F-201B con 48 cartuchos FC-CART-10 cuando solo hay 40 en yacimiento (pendiente).
- Daño estructural grave en fotos (B-101 en fuga-sin-enlace): siempre revisión humana.
- Repuesto fuera del BOM del equipo, tarea no estándar y ejecutor contratista.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from planta_contrato import EstadoOrden, OrdenSistema, Repuesto

from sistema.dominio.reglas_aprobacion import evaluar_reglas_aprobacion, extraer_estado_bombas

RUTAS_MATERIALES = [
    Path("/opt/planta_datos/materiales.yaml"),
    Path(__file__).resolve().parents[2] / "academia2026-tp" / "datos" / "materiales.yaml",
    Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/materiales.yaml"),
]


@pytest.fixture
def materiales() -> dict[str, Any]:
    ruta = next(r for r in RUTAS_MATERIALES if r.exists())
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def _orden_base(**overrides: Any) -> OrdenSistema:
    datos: dict[str, Any] = {
        "id": "OT-TEST-001",
        "ts": datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc),
        "estado": EstadoOrden.borrador,
        "equipo": "B-101",
        "evento_ids": ["EV-TEST-001"],
        "tareas": ["T-B-EMP"],
        "detiene_equipo": True,
        "reduce_inyeccion": False,
        "ejecutor": "propio",
        "repuestos": [Repuesto(codigo="TX-EMP-KIT", cantidad=1)],
        "motivo": None,
    }
    datos.update(overrides)
    return OrdenSistema(**datos)


def test_aprobacion_automatica_t_b_ace_sin_reserva(materiales: dict[str, Any]) -> None:
    """Reposición de aceite no detiene la bomba: aprueba incluso si no hay reserva disponible."""
    orden = _orden_base(
        equipo="B-102",
        tareas=["T-B-ACE"],
        detiene_equipo=False,
        repuestos=[Repuesto(codigo="ACE-ISO220", cantidad=8)],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "mantenimiento"}
    res = evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas)

    assert res.aprobada is True
    assert res.estado_resultante == EstadoOrden.aprobada
    assert res.costo_repuestos_usd == 72.0
    assert res.detiene_equipo is False
    assert res.reduce_inyeccion is False
    assert res.orden_decidida.motivo is None


def test_caso_limite_t_b_val_1900_usd_aprueba(materiales: dict[str, Any]) -> None:
    """Caso límite del Anexo: T-B-VAL cuesta USD 1.900 (< 2.500), hay 1 kit en yacimiento y reserva disponible."""
    orden = _orden_base(
        equipo="B-102",
        tareas=["T-B-VAL"],
        detiene_equipo=True,
        repuestos=[Repuesto(codigo="TX-VAL-KIT", cantidad=1)],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}
    res = evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas)

    assert res.aprobada is True
    assert res.estado_resultante == EstadoOrden.aprobada
    assert res.costo_repuestos_usd == 1900.0
    assert res.orden_decidida.estado == EstadoOrden.aprobada


def test_caso_limite_costo_2499_aprueba_y_2500_pendiente(materiales: dict[str, Any]) -> None:
    """Caso límite exacto del Anexo: USD 2.499 aprueba; USD 2.500 pasa a PendienteAprobacion con motivo."""
    mat_2499 = copy.deepcopy(materiales)
    for r in mat_2499["repuestos"]:
        if r["codigo"] == "TX-VAL-KIT":
            r["costo_usd"] = 2499

    mat_2500 = copy.deepcopy(materiales)
    for r in mat_2500["repuestos"]:
        if r["codigo"] == "TX-VAL-KIT":
            r["costo_usd"] = 2500

    orden = _orden_base(
        equipo="B-101",
        tareas=["T-B-VAL"],
        repuestos=[Repuesto(codigo="TX-VAL-KIT", cantidad=1)],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}

    res_ok = evaluar_reglas_aprobacion(orden, mat_2499, estado_bombas=bombas)
    assert res_ok.aprobada is True
    assert res_ok.costo_repuestos_usd == 2499.0
    assert res_ok.estado_resultante == EstadoOrden.aprobada

    res_pend = evaluar_reglas_aprobacion(orden, mat_2500, estado_bombas=bombas)
    assert res_pend.aprobada is False
    assert res_pend.costo_repuestos_usd == 2500.0
    assert res_pend.estado_resultante == EstadoOrden.pendiente
    assert res_pend.orden_decidida.motivo is not None
    assert "2500" in res_pend.orden_decidida.motivo


def test_caso_limite_tx_rod_cig_sin_stock_yacimiento(materiales: dict[str, Any]) -> None:
    """Caso límite del Anexo: TX-ROD-CIG tiene stock=0 en yacimiento (aunque haya 2 en central)."""
    orden = _orden_base(
        equipo="B-101",
        tareas=["T-B-ROD"],
        ejecutor="Servipat",
        repuestos=[
            Repuesto(codigo="TX-ROD-CIG", cantidad=1),
            Repuesto(codigo="TX-SELLO-CAR", cantidad=1),
        ],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}
    res = evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas)

    assert res.aprobada is False
    assert res.estado_resultante == EstadoOrden.pendiente
    assert res.orden_decidida.motivo is not None
    # Debe detectar los 4 incumplimientos: tarea no estándar, sin stock en yacimiento, costo >= 2500 y contratista
    texto_motivo = res.orden_decidida.motivo.lower()
    assert "no estándar" in texto_motivo
    assert "tx-rod-cig" in texto_motivo
    assert "yacimiento" in texto_motivo
    assert "2500" in texto_motivo
    assert "contratista" in texto_motivo


def test_caso_limite_reserva_no_disponible(materiales: dict[str, Any]) -> None:
    """Caso límite del Anexo: tarea estándar que detiene bomba en marcha sin reserva disponible."""
    orden = _orden_base(
        equipo="B-101",
        tareas=["T-B-EMP"],
        detiene_equipo=True,
        reduce_inyeccion=False,  # Incluso si el borrador dijera False, las reglas recalculan con estado real
        repuestos=[Repuesto(codigo="TX-EMP-KIT", cantidad=1)],
    )
    bombas_sin_reserva = {"B-101": "marcha", "B-102": "marcha", "B-103": "mantenimiento"}
    res = evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas_sin_reserva)

    assert res.aprobada is False
    assert res.reduce_inyeccion is True
    assert res.orden_decidida.reduce_inyeccion is True
    assert res.estado_resultante == EstadoOrden.pendiente
    assert res.orden_decidida.motivo is not None
    assert "reserva" in res.orden_decidida.motivo.lower()


def test_filtro_secundario_f201b_sin_stock_suficiente_en_yacimiento(materiales: dict[str, Any]) -> None:
    """F-201B requiere 48 FC-CART-10 pero solo hay 40 en yacimiento -> PendienteAprobacion."""
    orden = _orden_base(
        equipo="F-201B",
        tareas=["T-F-CAR"],
        detiene_equipo=False,
        repuestos=[Repuesto(codigo="FC-CART-10", cantidad=48)],
    )
    res = evaluar_reglas_aprobacion(orden, materiales)

    assert res.aprobada is False
    assert res.estado_resultante == EstadoOrden.pendiente
    assert res.orden_decidida.motivo is not None
    assert "fc-cart-10" in res.orden_decidida.motivo.lower()


def test_dano_estructural_grave_en_foto_fuerza_revision_humana(materiales: dict[str, Any]) -> None:
    """Regla de fuga-sin-enlace (B-101): si las fotos muestran daño estructural grave, va SIEMPRE a PendienteAprobacion."""
    orden = _orden_base(
        equipo="B-101",
        tareas=["T-B-EMP"],
        repuestos=[Repuesto(codigo="TX-EMP-KIT", cantidad=1)],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}

    # Sin daño estructural aprobaría
    assert evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas, dano_estructural_grave=False).aprobada is True

    # Con daño estructural grave debe quedar PendienteAprobacion con motivo explícito
    res = evaluar_reglas_aprobacion(orden, materiales, estado_bombas=bombas, dano_estructural_grave=True)
    assert res.aprobada is False
    assert res.estado_resultante == EstadoOrden.pendiente
    assert res.orden_decidida.motivo is not None
    assert "daño estructural grave" in res.orden_decidida.motivo.lower()


def test_repuesto_fuera_de_bom_y_tarea_inspeccion_no_estandar(materiales: dict[str, Any]) -> None:
    """Verifica rechazo automático si un repuesto no pertenece al BOM o si la tarea es inspección no estándar."""
    orden_bom_mal = _orden_base(
        equipo="F-201B",
        tareas=["T-F-CAR"],
        detiene_equipo=False,
        repuestos=[Repuesto(codigo="FC-CART-25", cantidad=48)],  # 25 µm es solo de F-201A, no F-201B
    )
    res_bom = evaluar_reglas_aprobacion(orden_bom_mal, materiales)
    assert res_bom.aprobada is False
    assert "lista de materiales" in (res_bom.orden_decidida.motivo or "").lower()

    orden_insp = _orden_base(
        equipo="B-101",
        tareas=["I-VAL"],
        ejecutor="propio",
        repuestos=[],
    )
    bombas = {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}
    res_insp = evaluar_reglas_aprobacion(orden_insp, materiales, estado_bombas=bombas)
    assert res_insp.aprobada is False
    assert "no estándar" in (res_insp.orden_decidida.motivo or "").lower()


def test_extraer_estado_bombas_desde_api_estado() -> None:
    """Verifica extracción del mapa de bombas desde ambos formatos posibles de GET /api/estado."""
    api_lista = {"bombas": [{"tag": "B-101", "estado": "marcha"}, {"tag": "B-103", "estado": "parada"}]}
    assert extraer_estado_bombas(api_lista) == {"B-101": "marcha", "B-103": "parada"}

    api_dict = {"bombas": {"B-101": "marcha", "B-102": "disparada"}}
    assert extraer_estado_bombas(api_dict) == {"B-101": "marcha", "B-102": "disparada"}
