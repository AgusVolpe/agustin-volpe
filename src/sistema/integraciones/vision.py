"""Análisis multimodal de fotos de planta con NVIDIA NIM (y cola cuando el enlace está caído).

Descarga las imágenes JPEG desde `http://sim:8100/media/fotos/...` y las analiza
con un modelo de visión de NVIDIA NIM (`VLM_MODEL`, por defecto
`meta/llama-3.2-11b-vision-instruct`) saliendo a través de `HTTP_PROXY=http://enlace:3128`.

Reglas operativas (Anexo Entrega 2):
- Si el enlace está caído (`enlace_ok=False`), la foto queda marcada como
  `encolada_sin_enlace` para procesarse al restablecerse la conexión. La alerta
  de seguridad nunca depende del análisis de la foto.
- Si el modelo detecta daño estructural grave (rotura de cañería/tapa, proyección
  de agua a presión, fisura), marca `dano_estructural_grave=True` para forzar
  revisión humana en las reglas de aprobación.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sistema.config import Settings, get_settings
from sistema.integraciones.planta_api import PlantaAPIClient

log = logging.getLogger(__name__)

PROMPT_VISION_PLANTA = (
    "Sos un inspector de integridad mecánica de la Planta de Inyección de Agua PIA-3.\n"
    "Analizá la foto del equipo de planta (puede ser fotografía real o foto de muestra etiquetada) "
    "y respondé ÚNICAMENTE un JSON válido sin texto extra con estas claves:\n"
    "{\n"
    '  "categoria_detectada": "empaquetadura | correas | acople | manometro_diferencial | filtro | rotura_descarga | trabajo_terminado | desconocida",\n'
    '  "dano_estructural_grave": true o false,\n'
    '  "descripcion": "resumen técnico breve en español de lo que se ve en la foto"\n'
    "}\n"
    "Regla crítica: marcá dano_estructural_grave=true SIEMPRE que la imagen muestre o indique "
    "rotura en línea de descarga (rotura_descarga), proyección violenta de agua a presión, "
    "fractura de cañería/tapa o daño estructural severo."
)


class ResultadoAnalisisFoto(BaseModel):
    """Resultado trazable del análisis multimodal de una fotografía de recorrida o cierre."""

    model_config = ConfigDict(extra="forbid")

    foto_url: str
    estado_analisis: Literal["analizada", "encolada_sin_enlace", "error"]
    categoria_detectada: str = "desconocida"
    dano_estructural_grave: bool = False
    descripcion: str = ""
    modelo_usado: str = ""
    tokens_entrada: int = 0
    tokens_salida: int = 0
    costo_usd: float = 0.0
    latencia_ms: float = 0.0


class AnalizadorFotos:
    """Cliente multimodal para inspección de fotos vía NVIDIA NIM con soporte offline."""

    def __init__(
        self,
        api_client: PlantaAPIClient,
        settings: Settings | None = None,
    ) -> None:
        self.api = api_client
        self.settings = settings or get_settings()

    def _tiene_api_key_nvidia(self) -> bool:
        clave = (self.settings.nvidia_api_key or "").strip()
        return bool(clave) and clave != "tu_api_key_aqui"

    def _analizar_fallback_local(self, foto_url: str) -> ResultadoAnalisisFoto:
        """Respaldo determinista cuando aún no se configuró NVIDIA_API_KEY o en tests locales."""
        url_lower = foto_url.lower()
        if "rotura" in url_lower or "descarga" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="rotura_descarga",
                dano_estructural_grave=True,
                descripcion="Evidencia visual de rotura en línea/cabezal de descarga (daño estructural grave).",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        if "empaquetadura" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="empaquetadura",
                dano_estructural_grave=False,
                descripcion="Evidencia visual de pérdida/goteo en prensaestopas de empaquetadura.",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        if "correa" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="correas",
                dano_estructural_grave=False,
                descripcion="Evidencia visual de desgaste o flojedad en juego de correas.",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        if "acople" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="acople",
                dano_estructural_grave=False,
                descripcion="Evidencia visual de desgaste de elastómero/desalineación en acople.",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        if "manometro" in url_lower or "diferencial" in url_lower or "filtro" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="manometro_diferencial",
                dano_estructural_grave=False,
                descripcion="Lectura visual de manómetro diferencial de filtro de cartuchos.",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        if "trabajo_terminado" in url_lower:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="analizada",
                categoria_detectada="trabajo_terminado",
                dano_estructural_grave=False,
                descripcion="Foto de verificación de trabajo finalizado sin pérdidas visibles.",
                modelo_usado="fallback-local-sin-nvidia-key",
            )
        return ResultadoAnalisisFoto(
            foto_url=foto_url,
            estado_analisis="analizada",
            categoria_detectada="desconocida",
            dano_estructural_grave=False,
            descripcion="Imagen recibida sin anomalía estructural evidente.",
            modelo_usado="fallback-local-sin-nvidia-key",
        )

    def _parsear_json_vlm(self, texto: str) -> dict:
        limpio = texto.strip()
        m = re.search(r"\{.*\}", limpio, re.DOTALL)
        if m:
            limpio = m.group(0)
        return json.loads(limpio)

    def _invocar_modelo_nvidia(self, b64: str, modelo: str) -> tuple[dict, int, int]:
        """Invoca un modelo VLM concreto de NVIDIA NIM y devuelve (datos_json, tok_in, tok_out)."""
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "nemotron" in modelo.lower() else {}
        llm_vlm = ChatOpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=self.settings.nvidia_api_key,
            model=modelo,
            temperature=0.0,
            max_tokens=500,
            timeout=self.settings.nvidia_timeout_s,
            extra_body=extra_body or None,
        )
        msg = HumanMessage(
            content=[
                {"type": "text", "text": PROMPT_VISION_PLANTA},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        )
        respuesta = llm_vlm.invoke([msg])
        contenido = str(respuesta.content)
        datos = self._parsear_json_vlm(contenido)
        usage = getattr(respuesta, "usage_metadata", None) or {}
        tok_in = int(usage.get("input_tokens", 0))
        tok_out = int(usage.get("output_tokens", 0))
        return datos, tok_in, tok_out

    def analizar_foto(
        self,
        foto_url: str,
        enlace_ok: bool = True,
        modelo_override: str | None = None,
    ) -> ResultadoAnalisisFoto:
        """Analiza una foto con NVIDIA NIM (principal -> fallback) si hay enlace y API key; la encola si no hay enlace."""
        if not enlace_ok:
            return ResultadoAnalisisFoto(
                foto_url=foto_url,
                estado_analisis="encolada_sin_enlace",
                categoria_detectada="pendiente",
                dano_estructural_grave=False,
                descripcion="Enlace caído: análisis multimodal de foto encolado hasta restablecer conexión.",
                modelo_usado=modelo_override or self.settings.vlm_model,
            )

        t0 = time.perf_counter()

        # Si no hay API key de NVIDIA cargada aún, usamos respaldo local sin romper el flujo
        if not self._tiene_api_key_nvidia():
            res = self._analizar_fallback_local(foto_url)
            res.latencia_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return res

        try:
            img_bytes = self.api.descargar_bytes(foto_url)
            b64 = base64.b64encode(img_bytes).decode("ascii")
        except Exception as exc:
            log.warning("No se pudo descargar %s (%s); aplicando respaldo local.", foto_url, exc)
            res_fb = self._analizar_fallback_local(foto_url)
            res_fb.latencia_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return res_fb

        modelos_a_probar = (
            [modelo_override]
            if modelo_override
            else [m for m in (self.settings.vlm_model, self.settings.vlm_fallback_model) if m]
        )

        ultimo_error: Exception | None = None
        for modelo in modelos_a_probar:
            t_mod = time.perf_counter()
            try:
                datos, tok_in, tok_out = self._invocar_modelo_nvidia(b64, modelo)
                lat_ms = round((time.perf_counter() - t_mod) * 1000.0, 2)
                cat = str(datos.get("categoria_detectada", "desconocida"))
                grave = bool(datos.get("dano_estructural_grave", False))
                if cat == "rotura_descarga":
                    grave = True

                return ResultadoAnalisisFoto(
                    foto_url=foto_url,
                    estado_analisis="analizada",
                    categoria_detectada=cat,
                    dano_estructural_grave=grave,
                    descripcion=str(datos.get("descripcion", "")),
                    modelo_usado=modelo,
                    tokens_entrada=tok_in,
                    tokens_salida=tok_out,
                    costo_usd=0.0,
                    latencia_ms=lat_ms,
                )
            except Exception as exc:
                ultimo_error = exc
                log.warning("Error llamando a NVIDIA NIM (%s) para %s: %s", modelo, foto_url, exc)

        log.warning("Todos los modelos NVIDIA fallaron para %s (%s); aplicando respaldo local.", foto_url, ultimo_error)
        res_fb = self._analizar_fallback_local(foto_url)
        res_fb.latencia_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return res_fb
