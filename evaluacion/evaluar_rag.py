"""Evaluación de recuperación (RAG) sobre la suite oficial de 62 preguntas de PIA-3.

Compara la línea de base BM25 sin filtrado de metadatos contra `BuscadorRAG`
(búsqueda híbrida BM25 + pgvector + filtrado de vigencia/aplicabilidad PIA-3 y
propagación de referencias cruzadas entre manuales de fabricantes y MBP-4).

Uso:
    python -m evaluacion.evaluar_rag
    python -m evaluacion.evaluar_rag --detalle
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable

from sistema.rag.buscador import BuscadorRAG
from sistema.rag.indexador import IndexadorRAG, cargar_fragmentos_locales

_STOPWORDS_BASE = set(
    """a al algo como con cual cuales cuando de del el ella en es esta este esto hay la las le lo los mas me
    mi muy no o otra otras para pero por que se si sin sobre su sus un una uno y ya ante hace hacer tiene son fue ser
    puede qué cuál cuáles cómo cuánto cuántos dónde""".split()
)


def _tokens_base(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return [t[:6] for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texto) if t not in _STOPWORDS_BASE]


class BM25Baseline:
    """Implementación idéntica a la línea de base `planta/corpus/evaluar.py`."""

    def __init__(self, fragmentos: list[dict], k1: float = 1.4, b: float = 0.75) -> None:
        self.ids = [f["id"] for f in fragmentos]
        self.docs = [Counter(_tokens_base(f["texto"])) for f in fragmentos]
        self.largos = [sum(d.values()) for d in self.docs]
        self.prom = sum(self.largos) / len(self.largos)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def buscar(self, pregunta: str, k: int) -> list[str]:
        q = _tokens_base(pregunta)
        puntajes = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += (
                        self.idf[t]
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * self.largos[i] / self.prom))
                    )
            if s > 0:
                puntajes.append((s, i))
        puntajes.sort(reverse=True)
        return [self.ids[i] for _, i in puntajes[:k]]


def cargar_suite_y_fragmentos() -> tuple[list[dict], list[dict]]:
    """Carga `retrieval.json` y `fragmentos.jsonl` desde `/opt/planta_datos` o el repo base."""
    rutas_suite = [
        Path("/opt/planta_datos/retrieval.json"),
        Path("../academia2026-tp/datos/documentos/evaluacion/retrieval.json"),
        Path("D:/Cursos/Curso software agentico/academia2026-tp/datos/documentos/evaluacion/retrieval.json"),
    ]
    suite: list[dict] = []
    for r in rutas_suite:
        if r.exists():
            suite = json.loads(r.read_text(encoding="utf-8"))
            break
    frags = cargar_fragmentos_locales()
    return suite, frags


def evaluar(buscar: Callable[[str, int], list[str]], suite: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Calcula `hit@k`, `recall@k`, `distractor@k` y `mrr` para toda la suite."""
    kmax = max(max(ks), 20)
    filas = []
    t0 = time.perf_counter()
    for q in suite:
        res = buscar(q["pregunta"], kmax)
        rel, dis = set(q["relevantes"]), set(q["distractores"])
        pos_rel = next((i for i, r in enumerate(res) if r in rel), None)
        fila = {
            "id": q["id"],
            "tipo": q["tipo"],
            "mrr": 1 / (pos_rel + 1) if pos_rel is not None else 0.0,
            "top": res[:5],
        }
        for k in ks:
            top = res[:k]
            fila[f"hit@{k}"] = float(any(r in rel for r in top))
            fila[f"recall@{k}"] = len(rel & set(top)) / len(rel)
            pos_dis = next((i for i, r in enumerate(top) if r in dis), None)
            fila[f"distractor@{k}"] = float(pos_dis is not None and (pos_rel is None or pos_dis < pos_rel))
        filas.append(fila)
    duracion_ms = (time.perf_counter() - t0) * 1000.0
    metricas = [m for m in filas[0] if "@" in m or m == "mrr"]
    total = {m: round(sum(f[m] for f in filas) / len(filas), 3) for m in metricas}
    por_tipo = {}
    for tipo in sorted({f["tipo"] for f in filas}):
        sub = [f for f in filas if f["tipo"] == tipo]
        por_tipo[tipo] = {
            "n": len(sub),
            **{m: round(sum(f[m] for f in sub) / len(sub), 3) for m in metricas if m.startswith(("recall@", "hit@"))},
        }
    return {
        "n": len(filas),
        "latencia_total_ms": round(duracion_ms, 2),
        "latencia_promedio_ms": round(duracion_ms / max(len(filas), 1), 2),
        "total": total,
        "por_tipo": por_tipo,
        "preguntas": filas,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--detalle", action="store_true", help="Muestra las preguntas con recall@5 < 1.0")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    suite, frags = cargar_suite_y_fragmentos()
    ks = tuple(args.k)

    base_res = evaluar(BM25Baseline(frags).buscar, suite, ks)
    indexador = IndexadorRAG(fragmentos_crudos=frags)
    indexador.indexar_en_postgres()
    buscador = BuscadorRAG(indexador=indexador)
    rag_res = evaluar(buscador.buscar, suite, ks)

    print(f"=== EVALUACIÓN RAG PIA-3 ({rag_res['n']} preguntas · {len(frags)} fragmentos) ===")
    print(f"pgvector activo: {indexador.pgvector_activo} | Latencia promedio: {rag_res['latencia_promedio_ms']} ms/pregunta\n")
    print("| Métrica | Baseline BM25 | BuscadorRAG Híbrido | Mejora |")
    print("|---|---:|---:|---:|")
    for m in [
        "mrr",
        "hit@1",
        "recall@1",
        "distractor@1",
        "hit@3",
        "recall@3",
        "distractor@3",
        "hit@5",
        "recall@5",
        "distractor@5",
        "hit@10",
        "recall@10",
        "distractor@10",
    ]:
        if m in rag_res["total"]:
            v_b = base_res["total"][m]
            v_r = rag_res["total"][m]
            diff = round(v_r - v_b, 3)
            signo = "+" if diff > 0 else ""
            print(f"| `{m}` | {v_b:.3f} | **{v_r:.3f}** | {signo}{diff:.3f} |")

    print("\n=== DESGLOSE POR TIPO (BuscadorRAG) ===")
    for tipo, m in rag_res["por_tipo"].items():
        metricas_str = "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n")
        print(f"  {tipo:14} n={m['n']:2}  {metricas_str}")

    if args.detalle:
        for f in rag_res["preguntas"]:
            if f.get("recall@5", 1.0) < 1.0 or f.get("distractor@5", 0.0) > 0.0:
                q = next(x for x in suite if x["id"] == f["id"])
                print(
                    f"\n[{f['id']}] ({f['tipo']}) {q['pregunta']}\n"
                    f"  relevantes: {q['relevantes']}\n"
                    f"  top-5:      {f['top']} (recall@5={f.get('recall@5'):.2f})"
                )


if __name__ == "__main__":
    main()
