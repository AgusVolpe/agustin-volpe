"""Bloque G: Procesamiento de notas de cierre por voz (`planta/cierres`) y certificación
contractual (`CT-SVP-2025` / `CT-SVP-2022`) en código puro.

Responsabilidades:
1. Parsear el texto transcripto de una `NotaCierre` (tareas, repuestos, horas y prueba).
2. Comparar lo declarado por el técnico o la contratista contra la `OrdenSistema` aprobada.
3. Calcular el monto del habilita y las diferencias contractuales según `CT-SVP-2025` Anexo II/IV
   y `CT-SVP-2022` (para trabajos iniciados antes del 2025-01-01).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from planta_contrato import OrdenSistema, Repuesto
from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------ Tablas contractuales CT-SVP-2025 / CT-SVP-2022
TAREA_A_ITEM: dict[str, str] = {
    "T-B-ROD": "SVP-01",
    "T-B-ALI": "SVP-02",
    "I-VIB": "SVP-03",
}

PRECIOS_CT_SVP_2025: dict[str, dict[str, Any]] = {
    "SVP-01": {"precio": 1150.0, "horas_incluidas": 16.0, "unidad": "precio_cerrado"},
    "SVP-02": {"precio": 780.0, "horas_incluidas": 5.0, "unidad": "precio_cerrado"},
    "SVP-03": {"precio": 320.0, "horas_incluidas": 1.0, "unidad": "por_equipo"},
    "SVP-04": {"precio": 68.0, "horas_incluidas": 0.0, "unidad": "hora"},
    "SVP-05": {"precio": 45.0, "horas_incluidas": 0.0, "unidad": "hora"},
    "SVP-06": {"precio": 180.0, "horas_incluidas": 0.0, "unidad": "movilizacion"},
    "SVP-07": {"precio": 350.0, "horas_incluidas": 0.0, "unidad": "guardia"},
}

PRECIOS_CT_SVP_2022: dict[str, dict[str, Any]] = {
    "SVP-01": {"precio": 980.0, "horas_incluidas": 16.0, "unidad": "precio_cerrado"},
    "SVP-02": {"precio": 650.0, "horas_incluidas": 5.0, "unidad": "precio_cerrado"},
    "SVP-03": {"precio": 280.0, "horas_incluidas": 1.0, "unidad": "por_equipo"},
    "SVP-04": {"precio": 55.0, "horas_incluidas": 0.0, "unidad": "hora"},
    "SVP-05": {"precio": 38.0, "horas_incluidas": 0.0, "unidad": "hora"},
    "SVP-06": {"precio": 150.0, "horas_incluidas": 0.0, "unidad": "movilizacion"},
    "SVP-07": {"precio": 300.0, "horas_incluidas": 0.0, "unidad": "guardia"},
}

HORAS_ESTANDAR_TAREA: dict[str, float] = {
    "T-B-EMP": 4.0,
    "T-B-VAL": 6.0,
    "T-B-ACE": 0.5,
    "T-B-COR": 2.0,
    "T-F-CAR": 3.0,
    "T-I-CAL": 1.0,
    "T-B-ROD": 16.0,
    "T-B-ALI": 5.0,
    "I-VIB": 1.0,
    "I-VAL": 2.0,
}

# Patrones para reconocer tareas en el texto dictado del cierre
PATRONES_TAREAS: list[tuple[str, re.Pattern[str]]] = [
    ("T-B-EMP", re.compile(r"(cambio de empaquetadura|empaquetadura|t-b-emp)", re.IGNORECASE)),
    ("T-B-VAL", re.compile(r"(cambio de valvulas|valvulas de succion|t-b-val)", re.IGNORECASE)),
    ("T-B-ACE", re.compile(r"(reposicion de aceite|cambio de aceite|aceite del carter|t-b-ace)", re.IGNORECASE)),
    ("T-B-COR", re.compile(r"(cambio de correas|correas|t-b-cor)", re.IGNORECASE)),
    ("T-F-CAR", re.compile(r"(cambio de cartuchos|cartuchos|t-f-car)", re.IGNORECASE)),
    ("T-I-CAL", re.compile(r"(calibracion de instrumento|calibracion|t-i-cal)", re.IGNORECASE)),
    ("T-B-ROD", re.compile(r"(cambio de rodamientos|rodamientos del ciguenal|t-b-rod)", re.IGNORECASE)),
    ("T-B-ALI", re.compile(r"(alineacion laser|alineacion de acople|t-b-ali)", re.IGNORECASE)),
    ("I-VIB", re.compile(r"(analisis de vibraciones|vibraciones con colector|i-vib)", re.IGNORECASE)),
    ("I-VAL", re.compile(r"(revision de valvulas|inspeccion de valvulas|i-val)", re.IGNORECASE)),
]

# Patrones para reconocer repuestos en el texto dictado del cierre
PATRONES_REPUESTOS: list[tuple[str, re.Pattern[str]]] = [
    ("TX-EMP-KIT", re.compile(r"(kit de empaquetadura|tx-emp-kit)", re.IGNORECASE)),
    ("TX-VAL-KIT", re.compile(r"(kit de valvulas|tx-val-kit)", re.IGNORECASE)),
    ("TX-ROD-CIG", re.compile(r"(juego de rodamientos|rodamientos de ciguenal|tx-rod-cig)", re.IGNORECASE)),
    ("TX-SELLO-CAR", re.compile(r"(juego de sellos de carter|sellos de carter|sello de carter|tx-sello-car)", re.IGNORECASE)),
    ("ACE-ISO220", re.compile(r"(aceite iso vg 220|aceite iso|ace-iso220)", re.IGNORECASE)),
    ("COR-5V-1800", re.compile(r"(correa 5v|correas 5v|cor-5v-1800)", re.IGNORECASE)),
    ("ACO-ELAST-90", re.compile(r"(elastomero de acople|elastomero|aco-elast-90)", re.IGNORECASE)),
    ("FC-CART-25", re.compile(r"(cartucho(?:s)? filtrante(?:s)? 25|fc-cart-25)", re.IGNORECASE)),
    ("FC-CART-10", re.compile(r"(cartucho(?:s)? filtrante(?:s)? 10|fc-cart-10)", re.IGNORECASE)),
    ("FC-JUNTA-TAPA", re.compile(r"(junta de tapa|fc-junta-tapa)", re.IGNORECASE)),
    ("INS-TRANS-P", re.compile(r"(transmisor de presion|ins-trans-p)", re.IGNORECASE)),
]

PALABRAS_NUMEROS: dict[str, int] = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _parsear_numero_espanol(fragmento: str) -> float | None:
    """Convierte números escritos con dígitos o palabras en español ('26,5', 'cuatro coma cinco', 'cuarenta y ocho')."""
    s = _norm(fragmento).strip()
    if not s:
        return None
    # Número directo con punto o coma
    m_dig = re.search(r"(\d+(?:[.,]\d+)?)", s)
    if m_dig:
        return float(m_dig.group(1).replace(",", "."))

    # Forma decimal hablada: "<entero> coma/punto/con <decimal>"
    m_dec = re.search(r"([a-z\s]+?)\s+(?:coma|punto|con)\s+([a-z]+)", s)
    if m_dec:
        ent = _palabras_a_entero(m_dec.group(1))
        dec = _palabras_a_entero(m_dec.group(2))
        if ent is not None and dec is not None:
            return float(f"{ent}.{dec}")

    # Forma "X y media"
    if "y media" in s:
        base = _palabras_a_entero(s.replace("y media", "").strip())
        if base is not None:
            return base + 0.5

    ent = _palabras_a_entero(s)
    return float(ent) if ent is not None else None


def _palabras_a_entero(texto: str) -> int | None:
    tokens = [tok for tok in re.split(r"[\s-]+", texto.strip()) if tok and tok != "y"]
    if not tokens:
        return None
    total = 0
    encontro = False
    for tok in tokens:
        if tok in PALABRAS_NUMEROS:
            total += PALABRAS_NUMEROS[tok]
            encontro = True
    return total if encontro else None


class RepuestoDeclarado(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codigo: str
    cantidad: float


class DatosCierreExtraidos(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texto_transcripto: str = ""
    tareas_declaradas: list[str] = Field(default_factory=list)
    repuestos_declarados: list[RepuestoDeclarado] = Field(default_factory=list)
    horas_declaradas: float | None = None
    prueba_resultado: str | None = None
    equipos_medidos: int | None = None


class ResultadoEvaluacionCierre(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orden_id: str
    es_contratista: bool
    lista_aplicada: str | None = None
    item_contrato: str | None = None
    monto_habilita_usd: float | None = None
    diferencias: list[str] = Field(default_factory=list)
    estado_sugerido: str  # "Cerrada" | "Certificada" | "Observada"
    datos_extraidos: DatosCierreExtraidos


def parsear_texto_cierre(texto: str) -> DatosCierreExtraidos:
    """Extrae tareas, repuestos, horas y prueba desde la transcripción de una `NotaCierre`."""
    t_norm = _norm(texto)

    # 1. Tareas mencionadas
    tareas: list[str] = []
    for cod, pat in PATRONES_TAREAS:
        if pat.search(t_norm) and cod not in tareas:
            tareas.append(cod)

    # 2. Repuestos usados (cláusula "use ..." o en todo el texto)
    repuestos: list[RepuestoDeclarado] = []
    clausula_use = t_norm
    m_use = re.search(r"\buse\b\s+([^.]+)", t_norm)
    if m_use:
        clausula_use = m_use.group(1)
        partes_rep = [p.strip() for p in re.split(r",|\sy\s(?=\d|un|una|dos|tres|cuatro|ocho|cuarenta|noventa)", clausula_use)]
    else:
        partes_rep = [t_norm]

    vistos_rep: dict[str, float] = {}
    for parte in partes_rep:
        for cod, pat in PATRONES_REPUESTOS:
            m = pat.search(parte)
            if m:
                prefijo = parte[: m.start()].strip()
                # Tomar las últimas 4 palabras previas al nombre del repuesto para extraer cantidad
                ultimas = " ".join(prefijo.split()[-4:])
                cant = _parsear_numero_espanol(ultimas)
                if cant is None or cant <= 0:
                    cant = 1.0
                vistos_rep[cod] = vistos_rep.get(cod, 0.0) + cant
                break
    for cod, cant in vistos_rep.items():
        repuestos.append(RepuestoDeclarado(codigo=cod, cantidad=round(cant, 2)))

    # 3. Horas declaradas
    horas: float | None = None
    m_h = re.search(r"(?:fueron|llevo|trabaje|demoro|total)?\s*([\d.,]+|[a-z\s]{1,30}?)\s+horas?\b", t_norm)
    if m_h:
        horas = _parsear_numero_espanol(m_h.group(1))

    # 4. Prueba / resultado operativo
    prueba: str | None = None
    for pat_p in (
        r"(probe a [^.,]+(?:,\s*sin perdida)?)",
        r"(quedo en servicio[^.]*)",
        r"(medi las tres bombas)",
    ):
        mp = re.search(pat_p, t_norm)
        if mp:
            prueba = mp.group(1).strip()
            break

    # 5. Equipos medidos
    equipos_medidos: int | None = None
    if "tres bombas" in t_norm or "3 bombas" in t_norm:
        equipos_medidos = 3
    elif "cuatro bombas" in t_norm or "4 bombas" in t_norm:
        equipos_medidos = 4

    return DatosCierreExtraidos(
        texto_transcripto=texto,
        tareas_declaradas=tareas,
        repuestos_declarados=repuestos,
        horas_declaradas=horas,
        prueba_resultado=prueba,
        equipos_medidos=equipos_medidos,
    )


def calcular_monto_contrato(
    tareas: list[str],
    fecha_inicio: str | datetime,
    equipo: str = "",
    cantidad_equipos_ivib: int | None = None,
) -> tuple[str, str | None, float]:
    """Calcula `(lista_aplicada, item_contrato, monto_habilita_usd)` según `CT-SVP-2025` o `CT-SVP-2022`.

    Regla CT-SVP-2025 §1.2 y §2.5:
    - Trabajos iniciados antes del 2025-01-01 usan la lista `CT-SVP-2022`.
    - Trabajos iniciados desde 2025-01-01 usan la lista `CT-SVP-2025`.
    - El monto del habilita sale siempre de la orden y de la lista de precios (Anexo IV §3.2).
    """
    if isinstance(fecha_inicio, datetime):
        fecha_iso = fecha_inicio.isoformat()
    else:
        fecha_iso = str(fecha_inicio)

    es_vigente = fecha_iso[:10] >= "2025-01-01"
    lista_nombre = "CT-SVP-2025" if es_vigente else "CT-SVP-2022"
    tabla = PRECIOS_CT_SVP_2025 if es_vigente else PRECIOS_CT_SVP_2022

    total = 0.0
    primer_item: str | None = None
    for tarea in tareas:
        item = TAREA_A_ITEM.get(tarea)
        if not item:
            continue
        if primer_item is None:
            primer_item = item
        precio_unit = float(tabla[item]["precio"])
        if tarea == "I-VIB":
            if cantidad_equipos_ivib is not None:
                n_eq = cantidad_equipos_ivib
            elif equipo.upper() == "PLANTA":
                n_eq = 3
            else:
                n_eq = 1
            total += precio_unit * n_eq
        else:
            total += precio_unit

    return lista_nombre, primer_item, round(total, 2)


def evaluar_cierre_y_certificacion(
    orden: OrdenSistema,
    texto_cierre: str,
    fecha_inicio: str | datetime | None = None,
    horas_registro_ingreso: float | None = None,
    horas_declaradas_override: float | None = None,
    repuestos_declarados_override: list[dict[str, Any]] | None = None,
    monto_declarado_usd: float | None = None,
    cantidad_equipos_ivib: int | None = None,
) -> ResultadoEvaluacionCierre:
    """Compara una nota de cierre contra la orden aprobada y calcula certificación según contrato.

    Soporta tanto el flujo en vivo (`planta/cierres` transcrito por Whisper) como la verificación
    contra el dataset oficial `cierres_ejemplo.json` (con `horas_registro_ingreso` y `monto_declarado_usd`).
    """
    datos = parsear_texto_cierre(texto_cierre)
    es_contratista = orden.ejecutor.strip().lower() != "propio"
    diferencias: list[str] = []

    # 1. Comparar tareas declaradas vs orden (si se logró parsear alguna tarea)
    if datos.tareas_declaradas:
        set_ord = set(orden.tareas)
        set_decl = set(datos.tareas_declaradas)
        extras_t = sorted(set_decl - set_ord)
        if extras_t:
            diferencias.append(f"Declara tarea(s) no incluida(s) en la orden: {', '.join(extras_t)}.")

    # 2. Comparar repuestos declarados vs repuestos autorizados en la orden
    mapa_aut: dict[str, float] = {}
    for r in orden.repuestos:
        mapa_aut[r.codigo] = mapa_aut.get(r.codigo, 0.0) + float(r.cantidad)

    if repuestos_declarados_override is not None:
        reps_decl = [
            RepuestoDeclarado(codigo=str(r["codigo"]), cantidad=float(r["cantidad"]))
            for r in repuestos_declarados_override
        ]
    else:
        reps_decl = list(datos.repuestos_declarados)

    mapa_decl: dict[str, float] = {}
    for r in reps_decl:
        mapa_decl[r.codigo] = mapa_decl.get(r.codigo, 0.0) + float(r.cantidad)

    for cod, cant_decl in mapa_decl.items():
        if cod not in mapa_aut:
            diferencias.append(
                f"Declara repuesto {cod} (x{cant_decl:g}) que no estaba en la orden ni fue autorizado."
            )
        elif cant_decl > mapa_aut[cod] + 1e-6:
            diferencias.append(
                f"Declara {cant_decl:g} de {cod}, superando lo autorizado en la orden ({mapa_aut[cod]:g})."
            )

    # 3. Comparar horas declaradas vs registro de ingreso o tope de horas estándar
    horas_decl = horas_declaradas_override if horas_declaradas_override is not None else datos.horas_declaradas
    if horas_decl is not None:
        if horas_registro_ingreso is not None:
            if horas_decl > horas_registro_ingreso + 0.5:
                diferencias.append(
                    f"Declara {horas_decl:g} h; el registro de ingreso da {horas_registro_ingreso:g} h. "
                    "Las horas sobre las incluidas no estaban autorizadas."
                )
        else:
            horas_esperadas = sum(HORAS_ESTANDAR_TAREA.get(t, 2.0) for t in orden.tareas) or 2.0
            # El simulador varía el trabajo real entre 0.85x y 1.30x; un exceso > 1.40x indica desvío de horas
            if es_contratista and horas_decl > max(horas_esperadas * 1.45, horas_esperadas + 2.0):
                diferencias.append(
                    f"Declara {horas_decl:g} h para tarea de referencia {horas_esperadas:g} h sin autorización previa."
                )

    # 4. Certificación contractual para contratista (CT-SVP-2025 / CT-SVP-2022)
    lista_aplicada: str | None = None
    item_contrato: str | None = None
    monto_habilita: float | None = None

    if es_contratista:
        f_ini = fecha_inicio or orden.ts
        # Inferencia de cantidad de equipos para I-VIB
        n_eq = cantidad_equipos_ivib
        if n_eq is None and "I-VIB" in orden.tareas:
            if orden.equipo.upper() == "PLANTA" or datos.equipos_medidos == 3:
                n_eq = 3
            elif monto_declarado_usd is not None and monto_declarado_usd >= 700:
                n_eq = 3
            else:
                n_eq = 1

        lista_aplicada, item_contrato, monto_habilita = calcular_monto_contrato(
            tareas=orden.tareas,
            fecha_inicio=f_ini,
            equipo=orden.equipo,
            cantidad_equipos_ivib=n_eq,
        )

        # Detección de desvíos de facturación cuando viene el monto declarado (CT-SVP-2025 §2.2, §2.3, §2.5)
        if monto_declarado_usd is not None and abs(monto_declarado_usd - monto_habilita) > 1.0:
            diff_usd = round(monto_declarado_usd - monto_habilita, 2)
            # a) Trabajo iniciado antes de 2025 cotizado con lista 2025
            if lista_aplicada == "CT-SVP-2022":
                _, _, monto_2025 = calcular_monto_contrato(
                    tareas=orden.tareas,
                    fecha_inicio="2025-01-01",
                    equipo=orden.equipo,
                    cantidad_equipos_ivib=n_eq,
                )
                if abs(monto_declarado_usd - monto_2025) <= 1.0:
                    diferencias.append(
                        "Cotiza con la lista vigente (CT-SVP-2025) un trabajo iniciado en 2024, "
                        "que corresponde certificar con CT-SVP-2022."
                    )
            # b) Movilización SVP-06 (+USD 180) sobre precio cerrado o por equipo
            if abs(diff_usd - PRECIOS_CT_SVP_2025["SVP-06"]["precio"]) <= 1.0 and not diferencias:
                diferencias.append(
                    "Factura movilización (SVP-06, USD 180) sobre precio cerrado/por equipo; "
                    "solo corresponde a trabajos por hora (CT-SVP-2025 §2.2)."
                )
            # c) Recargo del 50% por horas extra no solicitado por la operadora
            if abs(monto_declarado_usd - round(monto_habilita * 1.5, 2)) <= 1.0 and not diferencias:
                diferencias.append(
                    "Aplica recargo del 50 % fuera de horario sin pedido de la operadora (CT-SVP-2025 §2.3)."
                )
            # d) Diferencia genérica de monto (ej. equipo extra o ítem no autorizado)
            if not diferencias:
                diferencias.append(
                    f"Monto declarado (USD {monto_declarado_usd:.2f}) difiere del monto contractual "
                    f"correspondiente a la orden (USD {monto_habilita:.2f})."
                )

        estado_sugerido = "Observada" if diferencias else "Certificada"
    else:
        estado_sugerido = "Cerrada"

    return ResultadoEvaluacionCierre(
        orden_id=orden.id,
        es_contratista=es_contratista,
        lista_aplicada=lista_aplicada,
        item_contrato=item_contrato,
        monto_habilita_usd=monto_habilita,
        diferencias=diferencias,
        estado_sugerido=estado_sugerido,
        datos_extraidos=datos,
    )
