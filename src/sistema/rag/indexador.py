"""Indexador del corpus documental y de órdenes históricas en PostgreSQL con pgvector.

Procesa los fragmentos de `fragmentos.jsonl` (manuales TX350-D, TX350-C, FC48-3,
MBP-4, contrato CT-SVP-2025, ejemplos CERT-EJ e historial de órdenes OT), extrae
metadatos estructurados (cita canónica, documento, sección, vigencia, aplicabilidad
a PIA-3, equipos y fabricante) y almacena vectores densos normalizados en PostgreSQL
con la extensión `pgvector` (con respaldo en memoria cuando la base no está activa).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import psycopg

from sistema.config import Settings, get_settings

logger = logging.getLogger(__name__)

DIM_EMBEDDING = 256

_STOPWORDS = frozenset(
    """a al algo como con cual cuales cuando de del el ella en es esta este esto hay la las le lo los mas me
    mi muy no o otra otras para pero por que se si sin sobre su sus un una uno y ya ante hace hacer tiene son fue ser
    puede qué cuál cuáles cómo cuánto cuántos dónde""".split()
)

# Secciones de manuales vigentes que corresponden a otros modelos, otras plantas u opcionales no instalados en PIA-3
_SECCIONES_NO_APLICAN_PIA3 = frozenset(
    {
        "TX350-D §2.1",
        "TX350-D §3.3",
        "TX350-D §11",
        "TX350-D §11.1",
        "TX350-D §11.2",
        "TX350-D §12",
        "TX350-D §12.1",
        "TX350-D §12.2",
        "TX350-D §12.3",
        "TX350-D §12.4",
        "FC48-3 §2.1",
        "FC48-3 §11",
        "FC48-3 §11.1",
        "FC48-3 §11.2",
        "FC48-3 §16",
        "FC48-3 §16.1",
        "FC48-3 §16.2",
        "FC48-3 §16.3",
        "MBP-4 §5.3",
        "MBP-4 §17",
        "MBP-4 §17.1",
        "MBP-4 §17.2",
        "MBP-4 §17.3",
    }
)


def normalizar_texto_rag(texto: str) -> str:
    """Convierte a minúsculas y elimina tildes manteniendo dígitos y guiones."""
    normalizado = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


def tokenizar_rag(texto: str) -> list[str]:
    """Tokeniza un texto aplicando normalización y recorte de raíz de 6 caracteres."""
    limpio = normalizar_texto_rag(texto)
    tokens: list[str] = []
    for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", limpio):
        if tok not in _STOPWORDS:
            tokens.append(tok[:6])
    return tokens


def calcular_embedding_rag(texto: str, dim: int = DIM_EMBEDDING) -> list[float]:
    """Genera un embedding denso determinista L2-normalizado (feature hashing + bigramas).

    Permite ejecutar búsqueda vectorial híbrida con `pgvector` en la PC de planta
    con latencia sub-milisegundo y sin depender de descargas ni APIs externas.
    """
    vec = [0.0] * dim
    tokens = tokenizar_rag(texto)
    if not tokens:
        return vec

    # Unigramas y bigramas con pesos diferenciados
    features: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
    for i in range(len(tokens) - 1):
        features.append((f"{tokens[i]}_{tokens[i + 1]}", 0.7))

    for feat, peso in features:
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        signo = 1.0 if (digest[2] & 1) == 0 else -1.0
        vec[idx] += signo * peso

    norma = math.sqrt(sum(v * v for v in vec))
    if norma > 0:
        vec = [round(v / norma, 6) for v in vec]
    return vec


def inferir_equipos_fragmento(frag: dict[str, Any]) -> list[str]:
    """Infiere los TAGs de equipos asociados a un fragmento documental o de historial."""
    doc = str(frag.get("doc", ""))
    texto = f"{frag.get('titulo', '')} {frag.get('texto', '')}"
    equipos_detectados: list[str] = []

    for tag in ("B-101", "B-102", "B-103", "F-201A", "F-201B", "TK-301", "FT-401"):
        if tag.lower() in texto.lower():
            equipos_detectados.append(tag)

    # Alias frecuentes en órdenes históricas
    texto_norm = normalizar_texto_rag(texto)
    if re.search(r"\b(b1|bba 1|bomba 1|la uno|la 1)\b", texto_norm) and "B-101" not in equipos_detectados:
        equipos_detectados.append("B-101")
    if re.search(r"\b(b2|bba 2|bomba 2|la dos|la 2)\b", texto_norm) and "B-102" not in equipos_detectados:
        equipos_detectados.append("B-102")
    if (
        re.search(r"\b(b3|bba 3|bomba 3|la tres|la 3|la del fondo)\b", texto_norm)
        and "B-103" not in equipos_detectados
    ):
        equipos_detectados.append("B-103")
    if re.search(r"\b(fa|filtro a|primario)\b", texto_norm) and "F-201A" not in equipos_detectados:
        equipos_detectados.append("F-201A")
    if re.search(r"\b(fb|filtro b|secundario)\b", texto_norm) and "F-201B" not in equipos_detectados:
        equipos_detectados.append("F-201B")

    if not equipos_detectados:
        if doc.startswith("TX350"):
            return ["B-101", "B-102", "B-103"]
        if doc.startswith("FC48"):
            return ["F-201A", "F-201B"]
    return equipos_detectados


def inferir_fabricante_fragmento(doc: str) -> str:
    """Devuelve el emisor o fabricante asociado al documento."""
    if doc.startswith("TX350"):
        return "Trimax Industrial S.A."
    if doc.startswith("FC48"):
        return "Filtrapat S.A."
    if doc in ("CT-SVP-2025", "CERT-EJ"):
        return "Servipat S.A."
    return "Operadora PIA-3"


def enriquecer_fragmento(frag: dict[str, Any]) -> dict[str, Any]:
    """Enriquece un fragmento crudo de `fragmentos.jsonl` con metadatos y vector."""
    cita = str(frag["id"])
    doc = str(frag.get("doc", ""))
    vigente = bool(frag.get("vigente", True)) and doc != "TX350-C"
    aplica_pia3 = vigente and (cita not in _SECCIONES_NO_APLICAN_PIA3)
    equipos = inferir_equipos_fragmento(frag)
    fabricante = inferir_fabricante_fragmento(doc)
    titulo = str(frag.get("titulo") or "")
    clave = str(frag.get("clave") or "")
    texto = str(frag.get("texto") or "")

    texto_indexable = f"{cita} {titulo} {clave} {' '.join(equipos)}\n{texto}"
    embedding = calcular_embedding_rag(texto_indexable)

    return {
        "id": cita,
        "cita": cita,
        "doc": doc,
        "numero": str(frag.get("numero") or ""),
        "titulo": titulo,
        "clave": clave,
        "pagina": frag.get("pagina"),
        "vigente": vigente,
        "aplica_pia3": aplica_pia3,
        "equipos": equipos,
        "fabricante": fabricante,
        "texto": texto,
        "embedding": embedding,
    }


def cargar_fragmentos_locales(ruta_explicita: Path | None = None) -> list[dict[str, Any]]:
    """Carga los fragmentos desde `/opt/planta_datos/fragmentos.jsonl` o el repo base."""
    candidatos: list[Path] = []
    if ruta_explicita is not None:
        candidatos.append(ruta_explicita)
    candidatos.extend(
        [
            Path("/opt/planta_datos/fragmentos.jsonl"),
            Path("../academia2026-tp/datos/documentos/fragmentos.jsonl"),
            Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/documentos/fragmentos.jsonl"),
        ]
    )
    for ruta in candidatos:
        if ruta.exists():
            lineas = ruta.read_text(encoding="utf-8").splitlines()
            return [json.loads(linea) for linea in lineas if linea.strip()]
    return []


class IndexadorRAG:
    """Gestiona la indexación de fragmentos técnicos y órdenes en PostgreSQL + pgvector."""

    def __init__(
        self,
        settings: Settings | None = None,
        fragmentos_crudos: list[dict[str, Any]] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        crudos = fragmentos_crudos if fragmentos_crudos is not None else cargar_fragmentos_locales()
        self.fragmentos_enriquecidos: list[dict[str, Any]] = [enriquecer_fragmento(f) for f in crudos]
        self.pgvector_activo: bool = False

    def indexar_en_postgres(self) -> int:
        """Crea la tabla `documentos_rag` con `pgvector` e inserta todos los fragmentos."""
        if not self.fragmentos_enriquecidos:
            return 0
        try:
            with psycopg.connect(self._settings.postgres_dsn, autocommit=True, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS documentos_rag (
                            cita VARCHAR(120) PRIMARY KEY,
                            doc VARCHAR(64) NOT NULL,
                            numero VARCHAR(64) NOT NULL,
                            titulo TEXT NOT NULL,
                            clave VARCHAR(120),
                            vigente BOOLEAN NOT NULL DEFAULT TRUE,
                            aplica_pia3 BOOLEAN NOT NULL DEFAULT TRUE,
                            equipos JSONB NOT NULL DEFAULT '[]'::jsonb,
                            fabricante VARCHAR(120),
                            texto TEXT NOT NULL,
                            embedding vector({DIM_EMBEDDING}) NOT NULL
                        );
                        """
                    )
                    for frag in self.fragmentos_enriquecidos:
                        vec_str = "[" + ",".join(str(x) for x in frag["embedding"]) + "]"
                        cur.execute(
                            """
                            INSERT INTO documentos_rag (
                                cita, doc, numero, titulo, clave, vigente, aplica_pia3,
                                equipos, fabricante, texto, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::vector)
                            ON CONFLICT (cita) DO UPDATE SET
                                doc = EXCLUDED.doc,
                                numero = EXCLUDED.numero,
                                titulo = EXCLUDED.titulo,
                                clave = EXCLUDED.clave,
                                vigente = EXCLUDED.vigente,
                                aplica_pia3 = EXCLUDED.aplica_pia3,
                                equipos = EXCLUDED.equipos,
                                fabricante = EXCLUDED.fabricante,
                                texto = EXCLUDED.texto,
                                embedding = EXCLUDED.embedding;
                            """,
                            (
                                frag["cita"],
                                frag["doc"],
                                frag["numero"],
                                frag["titulo"],
                                frag["clave"],
                                frag["vigente"],
                                frag["aplica_pia3"],
                                json.dumps(frag["equipos"], ensure_ascii=False),
                                frag["fabricante"],
                                frag["texto"],
                                vec_str,
                            ),
                        )
            self.pgvector_activo = True
            logger.info("Indexados %d fragmentos en PostgreSQL (pgvector).", len(self.fragmentos_enriquecidos))
            return len(self.fragmentos_enriquecidos)
        except Exception as exc:
            self.pgvector_activo = False
            logger.warning("PostgreSQL/pgvector no disponible (%s); usando índice RAG en memoria.", exc)
            return len(self.fragmentos_enriquecidos)
