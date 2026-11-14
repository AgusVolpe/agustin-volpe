"""Cliente HTTP con timeouts y reintentos para consultar http://sim:8100."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from sistema.config import settings

log = logging.getLogger(__name__)

# Rutas donde buscar materiales.yaml y fragmentos.jsonl (en contenedor Docker o en carpeta vecina local)
RUTAS_MATERIALES = [
    Path("/opt/planta_datos/materiales.yaml"),
    Path(__file__).resolve().parents[4] / "academia2026-tp" / "datos" / "materiales.yaml",
]
RUTAS_FRAGMENTOS = [
    Path("/opt/planta_datos/fragmentos.jsonl"),
    Path(__file__).resolve().parents[4] / "academia2026-tp" / "datos" / "documentos" / "fragmentos.jsonl",
]


class PlantaAPIError(Exception):
    """Error al comunicarse con la API de la planta tras agotar los reintentos."""


class PlantaAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_intentos: int | None = None,
    ):
        self.base_url = (base_url or settings.sim_url).rstrip("/")
        self.timeout_s = timeout_s or settings.timeout_tool_s
        self.max_intentos = max_intentos or settings.max_intentos_tool
        self._client = httpx.Client(timeout=self.timeout_s, trust_env=False)
        self._planta_cache: dict[str, Any] | None = None
        self._materiales_cache: dict[str, Any] | None = None
        self._fragmentos_cache: list[dict[str, Any]] | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        ultimo_error: Exception | None = None
        for intento in range(1, self.max_intentos + 1):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                ultimo_error = e
                log.warning("Intento %d/%d fallido en GET %s: %s", intento, self.max_intentos, url, e)
                if intento < self.max_intentos:
                    time.sleep(0.2 * intento)
        raise PlantaAPIError(f"Error consultando {url} tras {self.max_intentos} intentos: {ultimo_error}")

    def descargar_bytes(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        ultimo_error: Exception | None = None
        for intento in range(1, self.max_intentos + 1):
            try:
                resp = self._client.get(url)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                ultimo_error = e
                if intento < self.max_intentos:
                    time.sleep(0.2 * intento)
        raise PlantaAPIError(f"No se pudo descargar {url}: {ultimo_error}")

    def obtener_planta(self, usar_cache: bool = True) -> dict[str, Any]:
        if usar_cache and self._planta_cache is not None:
            return self._planta_cache
        datos = self._get("/api/planta")
        self._planta_cache = datos
        return datos

    def obtener_estado(self) -> dict[str, Any]:
        return self._get("/api/estado")

    def obtener_historian(
        self, tag: str, variable: str, desde: str | None = None, hasta: str | None = None
    ) -> list[dict[str, Any]]:
        params = {}
        if desde:
            params["desde"] = desde
        if hasta:
            params["hasta"] = hasta
        return self._get(f"/api/historian/{tag}/{variable}", params=params or None)

    def obtener_historial_ordenes(
        self, equipo: str | None = None, desde: str | None = None, hasta: str | None = None
    ) -> list[dict[str, Any]]:
        params = {}
        if equipo:
            params["equipo"] = equipo
        if desde:
            params["desde"] = desde
        if hasta:
            params["hasta"] = hasta
        return self._get("/api/historial-ordenes", params=params or None)

    def obtener_materiales(self) -> dict[str, Any]:
        if self._materiales_cache is not None:
            return self._materiales_cache
        for ruta in RUTAS_MATERIALES:
            if ruta.exists():
                self._materiales_cache = yaml.safe_load(ruta.read_text(encoding="utf-8"))
                return self._materiales_cache
        raise PlantaAPIError("No se encontró el archivo oficial materiales.yaml")

    def obtener_secciones_documento(self, doc_id: str, numero: str | None = None) -> list[dict[str, Any]]:
        params = {"numero": numero} if numero else None
        return self._get(f"/api/documentos/{doc_id}/secciones", params=params)

    def obtener_fragmentos_corpus(self, usar_cache: bool = True) -> list[dict[str, Any]]:
        """Obtiene fragmentos.jsonl desde http://sim:8100/documentos/fragmentos.jsonl (o archivo local en tests)."""
        if usar_cache and self._fragmentos_cache is not None:
            return self._fragmentos_cache
        try:
            raw = self.descargar_bytes("/documentos/fragmentos.jsonl").decode("utf-8")
            self._fragmentos_cache = [json.loads(line) for line in raw.splitlines() if line.strip()]
            return self._fragmentos_cache
        except PlantaAPIError:
            for ruta in RUTAS_FRAGMENTOS:
                if ruta.exists():
                    raw = ruta.read_text(encoding="utf-8")
                    self._fragmentos_cache = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    return self._fragmentos_cache
            raise
