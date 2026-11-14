"""Tests unitarios para la detección de anomalías y alertas de seguridad en código puro."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from planta_contrato import FilaParte, LecturaSensor, TipoEvento

from sistema.dominio.deteccion import DetectorPlanta

RUTAS_PLANTA_YAML = [
    Path("/opt/planta_datos/planta.yaml"),
    Path(__file__).resolve().parents[2] / "academia2026-tp" / "datos" / "planta.yaml",
]


@pytest.fixture
def detector() -> DetectorPlanta:
    ruta = next(r for r in RUTAS_PLANTA_YAML if r.exists())
    definicion = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return DetectorPlanta(definicion_planta=definicion, muestras_sostenidas=3)


def test_alerta_seguridad_s1_a_s4_sin_falsas_alarmas(detector: DetectorPlanta):
    ts = datetime(2026, 11, 5, 9, 40, tzinfo=timezone.utc)

    # S1: Proyección de agua a presión
    r1 = detector.evaluar_seguridad_nota(
        "NOTA-0001",
        ts,
        "¡Atención! Se rompió algo en la descarga de la uno, sale agua a presión para todos lados, me alejé.",
    )
    assert r1.es_inseguro is True
    assert r1.codigo == "S1"
    assert r1.equipo == "B-101"
    assert r1.evento is not None and r1.evento.tipo == TipoEvento.alerta_seguridad
    assert r1.aviso_tablet is not None and r1.aviso_tablet.espera_respuesta is False

    # S2: Olor a gas
    r2 = detector.evaluar_seguridad_nota(
        "NOTA-0002", ts, "Hay olor a gas fuerte del lado del portón, viene del lado del separador."
    )
    assert r2.es_inseguro is True and r2.codigo == "S2"

    # S3: Humo / olor a quemado
    r3 = detector.evaluar_seguridad_nota(
        "NOTA-0003", ts, "Sale humo del motor de la dos, hay olor a quemado."
    )
    assert r3.es_inseguro is True and r3.codigo == "S3" and r3.equipo == "B-102"

    # S4: Persona lesionada
    r4 = detector.evaluar_seguridad_nota(
        "NOTA-0004", ts, "Mi compañero se golpeó la mano con la llave en el skid de filtros, está sangrando bastante."
    )
    assert r4.es_inseguro is True and r4.codigo == "S4"

    # Notas normales o con fallas mecánicas comunes NO deben disparar falsa alarma de seguridad
    for texto_normal in [
        "La dos está en mil seiscientos psi de descarga, sigue goteando por la empaquetadura.",
        "La tres está parada, en reserva. El filtro B me dio uno coma dos de diferencial.",
        "El pulmón está en cinco coma cuatro metros, está rebalsando a la pileta.",
    ]:
        rn = detector.evaluar_seguridad_nota("NOTA-NORM", ts, texto_normal)
        assert rn.es_inseguro is False


def test_parada_en_reserva_no_es_anomalia(detector: DetectorPlanta):
    ts = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)
    filas = [
        FilaParte(tag="B-101", variable="presion_descarga_local", valor=112.0, unidad="kg/cm2"),
        FilaParte(tag="B-103", variable="estado", valor=None, unidad=None, observacion="parada, en reserva"),
    ]
    res = detector.evaluar_parte_confirmado("NOTA-0010", ts, filas)
    por_tag = {c.evento.equipo: c.evento.tipo for c in res}
    assert por_tag["B-101"] == TipoEvento.sin_novedad
    assert por_tag["B-103"] == TipoEvento.sin_novedad


def test_conversion_unidades_y_limites_planta_no_rev_c(detector: DetectorPlanta):
    ts = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
    # En B-102, 1600 psi = 112.49 kg/cm2 -> Dentro de rango normal (95 a 125 kg/cm2)
    # En F-201B, diferencial 1.6 kg/cm2 >= alarma_alta (1.5 kg/cm2) -> Escalado
    filas = [
        FilaParte(tag="B-102", variable="presion_descarga_local", valor=1600.0, unidad="psi"),
        FilaParte(tag="F-201B", variable="presion_diferencial", valor=1.6, unidad="kg/cm2", observacion="diferencial alto"),
    ]
    res = detector.evaluar_parte_confirmado("NOTA-0011", ts, filas, fotos=["/media/fotos/manometro/01.jpg"])
    por_tag = {c.evento.equipo: c for c in res}

    assert por_tag["B-102"].evento.tipo == TipoEvento.sin_novedad
    assert por_tag["F-201B"].evento.tipo == TipoEvento.escalado
    assert por_tag["F-201B"].fotos == ["/media/fotos/manometro/01.jpg"]

    # Trampa Rev. C: vibración de 5.5 mm/s era alarma en Rev. C (4.5 mm/s), pero en Rev. D / planta.yaml la alarma es 7.1 mm/s
    fuera_vib, _ = detector.valor_fuera_de_limite("B-101", "vibracion", 5.5, "mm/s")
    assert fuera_vib is False


def test_sensor_escala_solo_anomalia_sostenida_no_pico_aislado(detector: DetectorPlanta):
    ts = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)

    # 1 muestra alta aislada (vibración 8.0 >= 7.1) seguida de lectura normal (4.0) -> NO debe escalar
    l1 = LecturaSensor(tag="B-102", variable="vibracion", valor=8.0, unidad="mm/s", ts=ts)
    l2 = LecturaSensor(tag="B-102", variable="vibracion", valor=4.0, unidad="mm/s", ts=ts)
    assert detector.evaluar_lectura_sensor(l1) is None
    assert detector.evaluar_lectura_sensor(l2) is None

    # 3 muestras consecutivas altas -> Escala en la 3ra muestra y no duplica en la 4ta
    assert detector.evaluar_lectura_sensor(l1) is None
    assert detector.evaluar_lectura_sensor(l1) is None
    esc = detector.evaluar_lectura_sensor(l1)
    assert esc is not None
    assert esc.evento.tipo == TipoEvento.escalado
    assert esc.evento.origen == "sensor"
    assert detector.evaluar_lectura_sensor(l1) is None
