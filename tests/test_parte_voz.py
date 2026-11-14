"""Tests unitarios para el flujo completo del Parte por Voz (A1 a A4b) y persistencia de conversación."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from sistema.integraciones.planta_api import PlantaAPIClient
from sistema.integraciones.tools import PlantaTools
from sistema.persistencia.repositorio import EstadoConversacionNota, RepositorioPlanta
from sistema.procesamiento.parte_voz import ProcesadorParteVoz, extraer_numero_hablado

RUTAS_PLANTA_YAML = [
    Path("/opt/planta_datos/planta.yaml"),
    Path(__file__).resolve().parents[2] / "academia2026-tp" / "datos" / "planta.yaml",
]


class FakeAPIParteVoz(PlantaAPIClient):
    def __init__(self):
        super().__init__(base_url="http://sim-fake:8100", timeout_s=0.5, max_intentos=1)
        ruta = next(r for r in RUTAS_PLANTA_YAML if r.exists())
        self._planta_data = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    def obtener_planta(self, usar_cache: bool = True) -> dict[str, Any]:
        return self._planta_data

    def obtener_estado(self) -> dict[str, Any]:
        return {"bombas": [{"tag": "B-102", "goteo": 45.0}, {"tag": "B-103", "goteo": 0.0}]}

    def obtener_historian(
        self, tag: str, variable: str, desde: str | None = None, hasta: str | None = None
    ) -> list[dict[str, Any]]:
        return [{"ts": "2026-11-01T00:00:00-03:00", "valor": 0.8}, {"ts": "2026-11-02T00:00:00-03:00", "valor": 1.4}]


@pytest.fixture
def proc() -> ProcesadorParteVoz:
    return ProcesadorParteVoz(tools=PlantaTools(api_client=FakeAPIParteVoz()))


def test_extraccion_numeros_hablados_de_campo():
    assert extraer_numero_hablado("ciento doce de descarga")[0] == 112.0
    assert extraer_numero_hablado("mil seiscientos veinte psi")[0] == 1620.0
    assert extraer_numero_hablado("uno ocho de diferencial")[0] == 1.8
    assert extraer_numero_hablado("uno coma cuatro metros")[0] == 1.4
    assert extraer_numero_hablado("uno y medio de diferencial")[0] == 1.5
    val_18, es_ent = extraer_numero_hablado("dieciocho de diferencial")
    assert val_18 == 18.0 and es_ent is True


def test_extraccion_nota_mezclada_con_unidades_por_bomba(proc: ProcesadorParteVoz):
    nota = (
        "La dos está en mil seiscientos veinte de descarga, sigue goteando por la empaquetadura. "
        "Ah, y el filtro B me dio uno ocho de diferencial."
    )
    res = proc.extraer_parte_de_texto(nota)
    assert res.ambiguedad is None

    por_var = {(f.tag, f.variable): f for f in res.filas}
    # B-102 sin unidad explícita adopta la de su manómetro local (psi)
    assert por_var[("B-102", "presion_descarga_local")].valor == 1620.0
    assert por_var[("B-102", "presion_descarga_local")].unidad == "psi"
    # Goteo en B-102
    assert por_var[("B-102", "goteo_empaquetadura")].observacion == "goteo por empaquetadura"
    # Filtro B diferencial
    assert por_var[("F-201B", "presion_diferencial")].valor == 1.8
    assert por_var[("F-201B", "presion_diferencial")].unidad == "kg/cm2"
    assert por_var[("F-201B", "presion_diferencial")].observacion == "diferencial alto"


def test_ambiguedad_equipo_la_bomba_y_resolucion_con_respuesta(proc: ProcesadorParteVoz):
    ts = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)
    nota = "La bomba está en ciento doce kilos de descarga."
    ext = proc.extraer_parte_de_texto(nota)
    assert ext.ambiguedad is not None
    assert ext.ambiguedad["tipo"] == "equipo"

    # El primer mensaje a la tablet debe ser una pregunta con opciones y sin filas
    msg_preg, confirmado = proc.armar_mensaje_tablet_inicial("NOTA-01", ts, ext)
    assert confirmado is False
    assert msg_preg.pantalla is not None
    assert msg_preg.pantalla.filas == []
    assert len(msg_preg.pantalla.opciones) == 3

    # El operador contesta "Era la uno." -> se arma el borrador con B-101
    confirmo, filas_resueltas = proc.procesar_respuesta_operador(
        "Era la uno.", ext.filas, nota, ext.ambiguedad
    )
    assert confirmo is False
    assert len(filas_resueltas) == 1
    assert filas_resueltas[0].tag == "B-101"
    assert filas_resueltas[0].valor == 112.0
    assert filas_resueltas[0].unidad == "kg/cm2"


def test_ambiguedad_valor_dieciocho_diferencial(proc: ProcesadorParteVoz):
    ts = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)
    nota = "El de la izquierda me dio dieciocho de diferencial."
    ext = proc.extraer_parte_de_texto(nota)
    assert ext.ambiguedad is not None
    assert ext.ambiguedad["tipo"] == "valor"
    assert ext.filas[0].tag == "F-201A"
    assert ext.filas[0].valor == 1.8

    msg, _ = proc.armar_mensaje_tablet_inicial("NOTA-02", ts, ext)
    assert msg.pantalla is not None
    assert len(msg.pantalla.opciones) == 2
    assert len(msg.pantalla.filas) == 1


def test_correccion_del_operador_en_tablet(proc: ProcesadorParteVoz):
    ext = proc.extraer_parte_de_texto("La uno está en ciento veinte kilos de descarga.")
    confirmo, corregidas = proc.procesar_respuesta_operador(
        "No. La uno estaba en ciento doce kilos, no en ciento veinte. Corregilo y te confirmo.",
        ext.filas,
        "La uno está en ciento veinte kilos de descarga.",
        None,
    )
    assert confirmo is False
    assert corregidas[0].valor == 112.0

    confirmo_final, _ = proc.procesar_respuesta_operador(
        "Sí, confirmo.", corregidas, "", None
    )
    assert confirmo_final is True


def test_persistencia_conversacion_nota():
    repo = RepositorioPlanta(usar_memoria=True)
    conv = EstadoConversacionNota(
        nota_id="NOTA-100",
        recorrida_id="REC-10",
        ts_nota="2026-11-03T08:00:00-03:00",
        operador="Héctor Llancafil",
        texto_transcripto="La uno está en ciento doce kilos de descarga.",
        estado="Borrador",
        intento=1,
        ultimo_mensaje_tablet_id="TAB-NOTA-100-1",
        filas=[{"tag": "B-101", "variable": "presion_descarga_local", "valor": 112.0, "unidad": "kg/cm2"}],
    )
    repo.guardar_conversacion(conv)
    recuperada = repo.obtener_conversacion_por_mensaje_tablet("TAB-NOTA-100-1")
    assert recuperada is not None
    assert recuperada.nota_id == "NOTA-100"
    assert recuperada.filas[0]["valor"] == 112.0
