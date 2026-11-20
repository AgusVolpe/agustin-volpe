"""Tests unitarios del Paso 4 (Entrega 2):
- Durabilidad ante apagón (`sin-luz`, criterio F3) y anti-duplicación (`E3`).
- Cola resiliente de análisis con fotos durante corte de enlace (`fuga-sin-enlace`, criterio E2).
- Trazabilidad LLM/VLM y checkpoints por `correlation_id`.
- Interfaz Humana FastAPI (Planificador, Supervisor e Inspector + validaciones de transición 409).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi.testclient import TestClient
from planta_contrato import EstadoOrden, EventoSistema, FilaParte, OrdenSistema, Repuesto, TipoEvento

from sistema.agente import AgentePlanta
from sistema.dominio.deteccion import ContextoEscalado
from sistema.integraciones.llm_cliente import TrazaInvocacionLLM
from sistema.integraciones.tools import PlantaTools
from sistema.integraciones.vision import ResultadoAnalisisFoto
from sistema.interfaz.api_humana import crear_app_humana
from sistema.orquestacion.workflow_orden import WorkflowOrdenTrabajo
from sistema.persistencia.repositorio import (
    CheckpointWorkflow,
    RegistroOrdenTrabajo,
    RepositorioPlanta,
)
from sistema.rag.buscador import BuscadorRAG


class _FakeAPI:
    def __init__(self, bombas_estado: dict[str, str] | None = None):
        self.bombas_estado = bombas_estado or {"B-101": "marcha", "B-102": "marcha", "B-103": "parada"}

    def obtener_planta(self) -> dict[str, Any]:
        return {
            "equipos": [
                {"tag": "B-101", "tipo": "bomba", "descripcion": "Bomba 1 TX-350", "nombres_informales": ["la uno"], "sensores": []},
                {"tag": "B-102", "tipo": "bomba", "descripcion": "Bomba 2 TX-350", "nombres_informales": ["la dos"], "sensores": []},
                {"tag": "B-103", "tipo": "bomba", "descripcion": "Bomba 3 TX-350", "nombres_informales": ["la tres"], "sensores": []},
                {"tag": "F-201A", "tipo": "filtro", "descripcion": "Filtro primario 25um", "nombres_informales": ["filtro a"], "sensores": []},
                {"tag": "F-201B", "tipo": "filtro", "descripcion": "Filtro secundario 10um", "nombres_informales": ["filtro b"], "sensores": []},
            ]
        }

    def obtener_estado(self) -> dict[str, Any]:
        return {"equipos": {k: {"estado": v} for k, v in self.bombas_estado.items()}}

    def obtener_materiales(self) -> dict[str, Any]:
        return {
            "repuestos": [
                {"codigo": "FC-CART-10", "nombre": "Cartucho 10um", "unidad": "unidad", "costo_usd": 17, "stock": {"yacimiento": 40, "central": 600}},
                {"codigo": "FC-CART-25", "nombre": "Cartucho 25um", "unidad": "unidad", "costo_usd": 14, "stock": {"yacimiento": 96, "central": 600}},
                {"codigo": "TX-VAL-KIT", "nombre": "Kit valvulas", "unidad": "kit", "costo_usd": 1900, "stock": {"yacimiento": 1, "central": 4}},
                {"codigo": "ACO-ELAST-90", "nombre": "Elastomero 90", "unidad": "unidad", "costo_usd": 150, "stock": {"yacimiento": 1, "central": 5}},
            ],
            "lista_materiales": {
                "B-101": ["TX-VAL-KIT", "ACO-ELAST-90"],
                "B-102": ["TX-VAL-KIT", "ACO-ELAST-90"],
                "F-201A": ["FC-CART-25"],
                "F-201B": ["FC-CART-10"],
            },
            "tareas_estandar": [
                {
                    "codigo": "T-F-CAR",
                    "nombre": "Cambio cartuchos",
                    "equipos": ["F-201A"],
                    "detiene": False,
                    "horas": 3,
                    "ejecutor": "propio",
                    "repuestos": [{"codigo": "FC-CART-25", "cantidad": 48}],
                },
                {
                    "codigo": "T-F-CAR",
                    "nombre": "Cambio cartuchos",
                    "equipos": ["F-201B"],
                    "detiene": False,
                    "horas": 3,
                    "ejecutor": "propio",
                    "repuestos": [{"codigo": "FC-CART-10", "cantidad": 48}],
                },
                {
                    "codigo": "T-B-VAL",
                    "nombre": "Cambio valvulas",
                    "equipos": ["B-101", "B-102", "B-103"],
                    "detiene": True,
                    "horas": 6,
                    "ejecutor": "propio",
                    "repuestos": [{"codigo": "TX-VAL-KIT", "cantidad": 1}],
                },
            ],
            "tareas_no_estandar": [
                {"codigo": "T-B-ALI", "nombre": "Alineacion", "detiene": True, "horas": 5, "ejecutor": "contratista"},
                {"codigo": "I-VAL", "nombre": "Revision valvulas", "detiene": True, "horas": 2, "ejecutor": "propio"},
            ],
        }

    def obtener_historial_ordenes(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def obtener_historial_sensores(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def descargar_bytes(self, ruta: str) -> bytes:
        return b"\xff\xd8\xff\xe0"


class _FakeLLM:
    def analizar_anomalia(self, prompt_sistema: str, prompt_usuario: str, herramientas_contexto: list[str]):
        traza = TrazaInvocacionLLM(
            paso="analisis_anomalia",
            modelo="qwen2.5:3b",
            proveedor="ollama_local",
            herramientas_usadas=herramientas_contexto,
            tokens_entrada=250,
            tokens_salida=80,
            latencia_ms=320.0,
            exito=True,
        )
        return None, traza


class _FakeAnalizadorFotos:
    def analizar_foto(self, foto_url: str, enlace_ok: bool) -> ResultadoAnalisisFoto:
        if not enlace_ok:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="encolada_sin_enlace",
                modelo_usado="ninguno",
            )
        es_rotura = "rotura" in foto_url.lower()
        return ResultadoAnalisisFoto(
            foto_url=foto_url,
            estado_analisis="analizada",
            modelo_usado="google/gemma-4-31b-it",
            categoria_detectada="rotura_descarga" if es_rotura else "empaquetadura",
            dano_estructural_grave=es_rotura,
            descripcion="Niple fisurado con proyección de agua a presión." if es_rotura else "Goteo en prensaestopas.",
            tokens_entrada=390,
            tokens_salida=55,
            latencia_ms=1950.0,
        )


def _crear_agente_en_memoria(repo_compartido: RepositorioPlanta) -> tuple[AgentePlanta, list[OrdenSistema]]:
    agente = AgentePlanta(iniciar_ui=False, iniciar_worker_analisis=False)
    agente.repo = repo_compartido
    api = _FakeAPI()
    tools = PlantaTools(api_client=api)  # type: ignore[arg-type]
    agente.workflow_orden = WorkflowOrdenTrabajo(
        tools=tools,
        buscador_rag=BuscadorRAG(indexador=None),
        llm_cliente=_FakeLLM(),  # type: ignore[arg-type]
        analizador_fotos=_FakeAnalizadorFotos(),  # type: ignore[arg-type]
    )
    publicadas: list[OrdenSistema] = []
    agente.publicar_orden = lambda o: publicadas.append(o)  # type: ignore[method-assign]
    agente.obtener_ts_planta = lambda: datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)  # type: ignore[method-assign]
    return agente, publicadas


# ------------------------------------------------------------------ 1. Durabilidad ante apagón (F3) y anti-duplicación (E3)
def test_sin_luz_orden_pendiente_sobrevive_reinicio_sin_retroceder_ni_duplicarse():
    repo_db = RepositorioPlanta(usar_memoria=True)

    # --- ANTES DEL CORTE DE LUZ (Instancia 1 del agente) ---
    agente_antes, publicadas_antes = _crear_agente_en_memoria(repo_db)
    ctx_f201b = ContextoEscalado(
        evento=EventoSistema(
            id="EV-SINLUZ-F201B-1",
            ts=datetime(2026, 11, 6, 9, 0, tzinfo=timezone.utc),
            tipo=TipoEvento.escalado,
            equipo="F-201B",
            origen="sensor",
            resumen="Desvío sostenido en sensor F-201B/presion_diferencial: 2.3 >= 2.0",
        ),
        motivos_anomalia=["presion_diferencial >= 2.0"],
    )
    agente_antes.encolar_escalado_para_analisis(ctx_f201b, enlace_ok=True)

    assert len(publicadas_antes) == 1
    orden_original = publicadas_antes[0]
    assert orden_original.equipo == "F-201B"
    assert orden_original.tareas == ["T-F-CAR"]
    assert orden_original.estado == EstadoOrden.pendiente
    assert orden_original.motivo and "stock" in orden_original.motivo.lower()

    # --- APAGÓN Y REINICIO (Nueva instancia de AgentePlanta leyendo la misma DB persistida) ---
    agente_despues, publicadas_despues = _crear_agente_en_memoria(repo_db)
    resumen_rec = agente_despues.recuperar_checkpoints_al_arrancar()
    assert resumen_rec["ordenes_totales"] == 1
    assert resumen_rec["pendientes_aprobacion"] == 1

    # Tras volver la luz, llega una nueva lectura sostenida del mismo problema en F-201B
    ctx_f201b_post = ContextoEscalado(
        evento=EventoSistema(
            id="EV-SINLUZ-F201B-2",
            ts=datetime(2026, 11, 6, 16, 0, tzinfo=timezone.utc),
            tipo=TipoEvento.escalado,
            equipo="F-201B",
            origen="parte",
            resumen="Filtro B con diferencial alto 2.4 kg/cm2",
        ),
        filas_parte=[FilaParte(tag="F-201B", variable="presion_diferencial", valor=2.4, unidad="kg/cm2")],
        motivos_anomalia=["presion_diferencial >= 2.0"],
    )
    agente_despues.encolar_escalado_para_analisis(ctx_f201b_post, enlace_ok=True)

    # Criterio F3 + E3: NO se duplicó ni retrocedió a Borrador; sigue existiendo 1 sola orden con el mismo ID
    assert publicadas_despues == []
    ordenes_db = repo_db.listar_ordenes()
    assert len(ordenes_db) == 1
    assert ordenes_db[0].orden.id == orden_original.id
    assert ordenes_db[0].orden.estado == EstadoOrden.pendiente
    assert "EV-SINLUZ-F201B-2" in ordenes_db[0].orden.evento_ids


# ------------------------------------------------------------------ 2. Cola de fotos sin enlace (fuga-sin-enlace / E2)
def test_cola_fotos_sin_enlace_espera_y_drena_al_volver_enlace():
    repo_db = RepositorioPlanta(usar_memoria=True)
    agente, publicadas = _crear_agente_en_memoria(repo_db)

    ctx_s1 = ContextoEscalado(
        evento=EventoSistema(
            id="EV-SEG-N01",
            ts=datetime(2026, 11, 5, 9, 40, tzinfo=timezone.utc),
            tipo=TipoEvento.alerta_seguridad,
            equipo="B-101",
            origen="parte",
            nota_ids=["N01"],
            resumen="Alerta S1: Pérdida con proyección de agua a presión en B-101",
        ),
        fotos=["/fotos/rotura_descarga/04.jpg"],
        motivos_anomalia=["proyeccion de agua a presion"],
    )

    # Con enlace caído: queda encolado (pendientes > 0), no publica orden todavía
    corr_id = agente.encolar_escalado_para_analisis(ctx_s1, enlace_ok=False)
    assert repo_db.contar_pendientes_sincronizar() == 1
    assert publicadas == []
    ckpt_susp = repo_db.obtener_checkpoint(corr_id)
    assert ckpt_susp is not None and ckpt_susp.etapa_actual == "esperando_enlace"

    # Si intentamos drenar mientras el enlace sigue caído, respeta `requiere_enlace=True`
    assert agente.drenar_cola_analisis(enlace_ok=False) == []
    assert repo_db.contar_pendientes_sincronizar() == 1

    # Cuando vuelve el enlace: drena, analiza la foto con VLM, registra trazas y emite PendienteAprobacion
    resultados = agente.drenar_cola_analisis(enlace_ok=True)
    assert len(resultados) == 1
    assert resultados[0].dano_estructural_grave is True
    assert repo_db.contar_pendientes_sincronizar() == 0
    assert len(publicadas) == 1
    assert publicadas[0].equipo == "B-101"
    assert publicadas[0].estado == EstadoOrden.pendiente
    assert publicadas[0].motivo is not None

    # Trazas registradas tanto para Ollama local como para NVIDIA NIM VLM
    trazas = repo_db.listar_trazas_llm(correlation_id=corr_id)
    modelos = {t["modelo"] for t in trazas}
    assert "qwen2.5:3b" in modelos
    assert "google/gemma-4-31b-it" in modelos


# ------------------------------------------------------------------ 3. Interfaz Humana FastAPI (3 roles + errores 409)
def test_interfaz_humana_flujos_planificador_supervisor_inspector():
    repo = RepositorioPlanta(usar_memoria=True)
    publicadas: list[OrdenSistema] = []
    ts_fijo = datetime(2026, 11, 6, 14, 30, tzinfo=timezone.utc)

    # Sembramos 3 órdenes: una PendienteAprobacion (propio), una Ejecutada (propio), una Ejecutada (Servipat)
    o_pend = OrdenSistema(
        id="OT-F201B-01",
        ts=ts_fijo,
        estado=EstadoOrden.pendiente,
        equipo="F-201B",
        tareas=["T-F-CAR"],
        detiene_equipo=False,
        reduce_inyeccion=False,
        ejecutor="propio",
        repuestos=[Repuesto(codigo="FC-CART-10", cantidad=48)],
        motivo="Sin stock suficiente en yacimiento",
        evento_ids=["EV-1"],
    )
    o_propio_ejec = OrdenSistema(
        id="OT-B102-02",
        ts=ts_fijo,
        estado=EstadoOrden.ejecutada,
        equipo="B-102",
        tareas=["T-B-VAL"],
        detiene_equipo=True,
        reduce_inyeccion=False,
        ejecutor="propio",
        repuestos=[Repuesto(codigo="TX-VAL-KIT", cantidad=1)],
        evento_ids=["EV-2"],
    )
    o_contr_ejec = OrdenSistema(
        id="OT-B101-03",
        ts=ts_fijo,
        estado=EstadoOrden.ejecutada,
        equipo="B-101",
        tareas=["T-B-ALI"],
        detiene_equipo=True,
        reduce_inyeccion=False,
        ejecutor="Servipat",
        repuestos=[Repuesto(codigo="ACO-ELAST-90", cantidad=1)],
        evento_ids=["EV-3"],
    )
    for idx, ord_obj in enumerate((o_pend, o_propio_ejec, o_contr_ejec), start=1):
        repo.guardar_orden(RegistroOrdenTrabajo(orden=ord_obj, correlation_id=f"CORR-{idx}", citas_validadas=["MBP-4 §6.4"]))
        repo.guardar_checkpoint(
            CheckpointWorkflow(correlation_id=f"CORR-{idx}", evento_id=f"EV-{idx}", orden_id=ord_obj.id, etapa_actual="activo")
        )

    app = crear_app_humana(repo=repo, publicar_orden_fn=publicadas.append, obtener_ts_fn=lambda: ts_fijo)
    client = TestClient(app)

    # Panel Web HTML y salud
    assert client.get("/").status_code == 200
    salud = client.get("/api/salud").json()
    assert salud["ordenes_totales"] == 3
    assert salud["pendientes_aprobacion"] == 1

    # 1) Planificador aprueba OT-F201B-01
    r_ap = client.post("/api/ordenes/OT-F201B-01/aprobar")
    assert r_ap.status_code == 200
    assert r_ap.json()["orden"]["estado"] == "Aprobada"
    assert publicadas[-1].id == "OT-F201B-01" and publicadas[-1].estado == EstadoOrden.aprobada

    # Intentar rechazar una orden ya aprobada -> 409 Conflict
    r_err = client.post("/api/ordenes/OT-F201B-01/rechazar", json={"motivo": "Ya no aplica"})
    assert r_err.status_code == 409

    # 2) Supervisor cierra orden propia OT-B102-02
    r_cer = client.post("/api/ordenes/OT-B102-02/cerrar")
    assert r_cer.status_code == 200
    assert r_cer.json()["orden"]["estado"] == "Cerrada"
    assert publicadas[-1].estado == EstadoOrden.cerrada

    # Supervisor NO puede cerrar orden de contratista -> 409
    assert client.post("/api/ordenes/OT-B101-03/cerrar").status_code == 409

    # 3) Inspector observa y luego certifica orden de contratista Servipat OT-B101-03
    r_obs = client.post("/api/ordenes/OT-B101-03/observar", json={"motivo": "Cobraron 2 elastómeros en vez de 1"})
    assert r_obs.status_code == 200
    assert r_obs.json()["orden"]["estado"] == "Observada"
    assert "elastómeros" in r_obs.json()["orden"]["motivo"]

    r_cert = client.post("/api/ordenes/OT-B101-03/certificar", json={})
    assert r_cert.status_code == 200
    datos_cert = r_cert.json()["orden"]
    assert datos_cert["estado"] == "Certificada"
    assert datos_cert["habilita"] == "HAB-2026-0001"
    assert datos_cert["monto_habilita_usd"] == 780.0
    assert publicadas[-1].estado == EstadoOrden.certificada

    # Checkpoints reflejan el ciclo completo
    ckpts = {c["correlation_id"]: c for c in client.get("/api/checkpoints").json()}
    assert ckpts["CORR-1"]["etapa_actual"] == "aprobada"
    assert ckpts["CORR-2"]["estado_workflow"] == "completado"
    assert ckpts["CORR-3"]["etapa_actual"] == "certificada"
    assert ckpts["CORR-3"]["estado_workflow"] == "completado"
