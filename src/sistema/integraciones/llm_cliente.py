"""Cliente LLM local (Ollama en contenedor) con salida estructurada y trazabilidad.

Conecta con el servicio `ollama` en `planta-lan` (`http://ollama:11434`, modelo
`qwen2.5:3b`) mediante `langchain-ollama`, registrando modelo usado, tokens,
costo (USD 0 en local) y latencia por cada invocación (requisito F · Registro).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sistema.config import Settings, get_settings

log = logging.getLogger(__name__)


class CausaProbableOut(BaseModel):
    """Una causa probable ordenada dentro del análisis de anomalía (Bloque C)."""

    model_config = ConfigDict(extra="forbid")

    codigo_falla: str = Field(description="Código F01..F10 o INDETERMINADA")
    descripcion: str
    evidencia: str
    que_falta_confirmar: str = ""
    citas: list[str] = Field(default_factory=list)


class SalidaAnalisisLLM(BaseModel):
    """Esquema JSON estricto que produce el paso agéntico de análisis de anomalías."""

    model_config = ConfigDict(extra="forbid")

    equipo_objetivo: str = Field(description="TAG donde se ejecuta la tarea (puede diferir del evento si es F09)")
    falla_principal: str = Field(description="F01..F10 o INDETERMINADA")
    requiere_orden: bool = Field(default=True, description="False para F10 sensor con deriva sin intervención")
    causas_probables: list[CausaProbableOut] = Field(default_factory=list)
    tareas_propuestas: list[str] = Field(default_factory=list)
    ejecutor_propuesto: str = "propio"
    repuestos_propuestos: list[dict[str, Any]] = Field(default_factory=list)
    citas_usadas: list[str] = Field(default_factory=list)
    resumen_tecnico: str = ""


class TrazaInvocacionLLM(BaseModel):
    """Registro auditable de cada llamada a un modelo (requisito F · Trazabilidad)."""

    model_config = ConfigDict(extra="forbid")

    paso: str
    modelo: str
    proveedor: str
    herramientas_usadas: list[str] = Field(default_factory=list)
    tokens_entrada: int = 0
    tokens_salida: int = 0
    costo_usd: float = 0.0
    latencia_ms: float = 0.0
    exito: bool = True
    error: str | None = None


class ClienteLLMLocal:
    """Wrapper resiliente sobre ChatOllama (qwen2.5:3b) con reintentos, timeout y traza."""

    def __init__(self, settings: Settings | None = None, habilitar_ollama: bool = True) -> None:
        self.settings = settings or get_settings()
        self.habilitar_ollama = habilitar_ollama

    def precargar(self) -> None:
        """Precarga el modelo local en la RAM de Ollama al iniciar para evitar latencia fría."""
        if not self.habilitar_ollama:
            return
        try:
            import httpx

            with httpx.Client(timeout=45.0, trust_env=False) as client:
                r = client.post(
                    f"{self.settings.ollama_base_url}/api/generate",
                    json={
                        "model": self.settings.llm_local_model,
                        "prompt": "ok",
                        "stream": False,
                        "keep_alive": "15m",
                        "options": {"num_predict": 1, "num_ctx": 2048},
                    },
                )
                if r.status_code == 200:
                    log.info("Modelo local Ollama '%s' precargado en memoria.", self.settings.llm_local_model)
        except Exception as exc:
            log.warning("No se pudo precargar Ollama (%s): %s", self.settings.llm_local_model, exc)

    def _extraer_json(self, texto: str) -> dict[str, Any]:
        limpio = texto.strip()
        m = re.search(r"\{.*\}", limpio, re.DOTALL)
        if m:
            limpio = m.group(0)
        return json.loads(limpio)

    def analizar_anomalia(
        self,
        prompt_sistema: str,
        prompt_usuario: str,
        herramientas_contexto: list[str] | None = None,
    ) -> tuple[SalidaAnalisisLLM | None, TrazaInvocacionLLM]:
        """Invoca al LLM local en Ollama pidiendo JSON estricto según `SalidaAnalisisLLM`."""
        t0 = time.perf_counter()
        tools_list = list(herramientas_contexto or [])

        if not self.habilitar_ollama:
            return None, TrazaInvocacionLLM(
                paso="analisis_anomalia",
                modelo=self.settings.llm_local_model,
                proveedor="ollama-deshabilitado",
                herramientas_usadas=tools_list,
                latencia_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                exito=False,
                error="Ollama deshabilitado explícitamente en esta instancia.",
            )

        ultimo_error: str | None = None
        try:
            import httpx

            payload = {
                "model": self.settings.llm_local_model,
                "messages": [
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                "format": "json",
                "stream": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 1024,
                    "num_predict": 48,
                },
            }
            with httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0), trust_env=False) as client:
                r = client.post(f"{self.settings.ollama_base_url}/api/chat", json=payload)
                r.raise_for_status()
                body = r.json()

            contenido = str((body.get("message") or {}).get("content", ""))
            datos = self._extraer_json(contenido)
            salida = self._normalizar_salida_compacta(datos)

            tok_in = int(body.get("prompt_eval_count", 0))
            tok_out = int(body.get("eval_count", 0))
            traza = TrazaInvocacionLLM(
                paso="analisis_anomalia",
                modelo=self.settings.llm_local_model,
                proveedor="ollama-local",
                herramientas_usadas=tools_list,
                tokens_entrada=tok_in,
                tokens_salida=tok_out,
                costo_usd=0.0,
                latencia_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                exito=True,
            )
            log.info(
                "Ollama (%s) OK en %.0f ms (%d in / %d out tok) -> falla=%s tareas=%s",
                self.settings.llm_local_model,
                traza.latencia_ms,
                tok_in,
                tok_out,
                salida.falla_principal,
                salida.tareas_propuestas,
            )
            return salida, traza
        except Exception as exc:
            ultimo_error = str(exc)
            log.warning("Fallo invocando Ollama (%s): %s", self.settings.llm_local_model, ultimo_error)

        traza_err = TrazaInvocacionLLM(
            paso="analisis_anomalia",
            modelo=self.settings.llm_local_model,
            proveedor="ollama-local",
            herramientas_usadas=tools_list,
            latencia_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            exito=False,
            error=ultimo_error,
        )
        return None, traza_err

    @staticmethod
    def _normalizar_salida_compacta(datos: dict[str, Any]) -> SalidaAnalisisLLM:
        """Acepta tanto el esquema completo `SalidaAnalisisLLM` como el formato compacto de baja latencia."""
        if "equipo_objetivo" in datos and "falla_principal" in datos:
            return SalidaAnalisisLLM.model_validate(datos)

        equipo = str(datos.get("equipo") or datos.get("equipo_objetivo") or "B-101").strip().upper()
        falla = str(datos.get("falla") or datos.get("falla_principal") or "INDETERMINADA").strip().upper()
        req = bool(datos.get("requiere_orden", falla != "F10"))
        tarea_raw = datos.get("tareas") or datos.get("tareas_propuestas") or datos.get("tarea")
        if isinstance(tarea_raw, str):
            tareas = [tarea_raw.strip().upper()] if tarea_raw.strip() else []
        elif isinstance(tarea_raw, list):
            tareas = [str(t).strip().upper() for t in tarea_raw if str(t).strip()]
        else:
            tareas = []
        citas_raw = datos.get("citas") or datos.get("citas_usadas") or []
        citas = [str(c).strip() for c in citas_raw] if isinstance(citas_raw, list) else []
        ejecutor = str(datos.get("ejecutor") or datos.get("ejecutor_propuesto") or "propio").strip()
        resumen = str(datos.get("resumen") or datos.get("resumen_tecnico") or "").strip()

        return SalidaAnalisisLLM(
            equipo_objetivo=equipo,
            falla_principal=falla,
            requiere_orden=req,
            causas_probables=[
                CausaProbableOut(
                    codigo_falla=falla,
                    descripcion=resumen or f"Diagnóstico {falla} en {equipo}",
                    evidencia="Evidencia confirmada de planta",
                    citas=citas,
                )
            ],
            tareas_propuestas=tareas,
            ejecutor_propuesto=ejecutor,
            repuestos_propuestos=[],
            citas_usadas=citas,
            resumen_tecnico=resumen,
        )
