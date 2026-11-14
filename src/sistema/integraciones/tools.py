"""6 Tools de solo lectura con Pydantic estricto (extra="forbid"):
1. consultar_equipos
2. consultar_historial_sensores
3. consultar_historial_ordenes
4. consultar_inventario_y_bom
5. consultar_catalogo_tareas
6. buscar_en_manuales
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from sistema.integraciones.planta_api import PlantaAPIClient, PlantaAPIError


class StrictModel(BaseModel):
    """Modelo base con validación estricta que prohíbe campos extra (requisito 1.3)."""

    model_config = ConfigDict(extra="forbid")


# =====================================================================
# 1. TOOL: EQUIPOS (GET /api/planta)
# =====================================================================


class ConsultarEquiposInput(StrictModel):
    consulta: str | None = Field(
        default=None,
        description="TAG exacto (ej. 'B-102') o nombre informal dicho en campo (ej. 'la dos', 'el de la izquierda', 'la de reserva'). None devuelve todos.",
    )


class LimiteSensorOut(StrictModel):
    variable: str
    unidad: str
    alarma_baja: float | None = None
    alarma_alta: float | None = None
    disparo: float | None = None


class EquipoOut(StrictModel):
    tag: str
    tipo: str
    descripcion: str
    fabricante: str | None = None
    modelo: str | None = None
    nombres_informales: list[str] = Field(default_factory=list)
    unidad_manometro_local: str | None = None
    rol_inicial: str | None = None
    sensores: list[LimiteSensorOut] = Field(default_factory=list)


class ConsultarEquiposOutput(StrictModel):
    equipos: list[EquipoOut]
    ambiguo: bool = False
    mensaje: str | None = None
    pregunta_aclaracion: str | None = Field(
        default=None,
        description="Pregunta para pedir al operador/usuario que especifique el equipo cuando es ambiguo o indeterminado",
    )
    opciones_aclaracion: list[str] = Field(
        default_factory=list,
        description="Opciones formateadas (ej. 'B-101 (la uno)') para ofrecer en la tablet",
    )


# =====================================================================
# 2. TOOL: HISTORIAL DE SENSORES (GET /api/historian/{tag}/{variable})
# =====================================================================


class ConsultarHistorialSensoresInput(StrictModel):
    tag: str = Field(min_length=1, description="TAG del equipo (ej. 'B-101', 'F-201B')")
    variable: str = Field(
        min_length=1,
        description="Variable del sensor (ej. 'presion_descarga', 'presion_succion', 'temperatura_aceite', 'vibracion', 'presion_diferencial', 'caudal')",
    )
    desde: str | None = Field(default=None, description="Fecha/hora ISO 8601 inicial opcional")
    hasta: str | None = Field(default=None, description="Fecha/hora ISO 8601 final opcional")
    ultimas_muestras: int = Field(default=60, ge=1, le=2000, description="Cantidad de muestras recientes a resumir")


class MuestraSensorOut(StrictModel):
    ts: str
    valor: float | None


class ConsultarHistorialSensoresOutput(StrictModel):
    tag: str
    variable: str
    cantidad_muestras: int
    minimo_historico: float | None = None
    maximo_historico: float | None = None
    promedio_reciente: float | None = None
    ultimo_valor: float | None = None
    tendencia: str = Field(description="'estable', 'subiendo', 'bajando' o 'sin_datos'")
    muestras_recientes: list[MuestraSensorOut] = Field(default_factory=list)


# =====================================================================
# 3. TOOL: HISTORIAL DE ÓRDENES (GET /api/historial-ordenes)
# =====================================================================


class ConsultarHistorialOrdenesInput(StrictModel):
    equipo: str | None = Field(default=None, description="TAG del equipo (ej. 'B-101', 'F-201B')")
    desde: str | None = Field(default=None, description="Fecha ISO 8601 desde")
    hasta: str | None = Field(default=None, description="Fecha ISO 8601 hasta")
    limite: int = Field(default=20, ge=1, le=200, description="Cantidad máxima de órdenes a devolver (las más recientes)")


class OrdenHistoricaOut(StrictModel):
    id: str
    fecha: str
    equipo: str
    tarea: str | None = None
    falla: str | None = None
    nota: str | None = None
    repuestos: list[str] = Field(default_factory=list)
    horas: float | None = None
    origen: str | None = None


class ConsultarHistorialOrdenesOutput(StrictModel):
    total_encontradas: int
    ordenes: list[OrdenHistoricaOut]


# =====================================================================
# 4. TOOL: INVENTARIO, STOCK Y LISTA DE MATERIALES (BOM)
# =====================================================================


class ConsultarInventarioInput(StrictModel):
    equipo: str | None = Field(
        default=None,
        description="TAG del equipo para filtrar únicamente los repuestos de su lista de materiales (BOM)",
    )
    codigo_repuesto: str | None = Field(
        default=None,
        description="Código exacto del repuesto (ej. 'TX-EMP-KIT', 'FC-CART-10')",
    )


class RepuestoInventarioOut(StrictModel):
    codigo: str
    nombre: str
    unidad: str
    costo_usd: float
    stock_yacimiento: int
    stock_central: int


class ConsultarInventarioOutput(StrictModel):
    equipo: str | None = None
    lista_materiales_equipo: list[str] = Field(default_factory=list)
    repuestos: list[RepuestoInventarioOut]


# =====================================================================
# 5. TOOL: CATÁLOGO DE TAREAS
# =====================================================================


class ConsultarCatalogoTareasInput(StrictModel):
    equipo: str | None = Field(default=None, description="TAG del equipo (ej. 'B-101', 'F-201A')")
    codigo_tarea: str | None = Field(default=None, description="Código de tarea (ej. 'T-B-EMP', 'T-F-CAR', 'T-B-ROD')")
    solo_estandar: bool = Field(default=False, description="Si es True, devuelve únicamente tareas estándar")


class RepuestoTareaOut(StrictModel):
    codigo: str
    cantidad: float


class TareaCatalogoOut(StrictModel):
    codigo: str
    nombre: str
    estandar: bool
    equipos: list[str] = Field(default_factory=list)
    detiene: bool
    horas: float
    ejecutor: str
    repuestos: list[RepuestoTareaOut] = Field(default_factory=list)


class ConsultarCatalogoTareasOutput(StrictModel):
    tareas: list[TareaCatalogoOut]


# =====================================================================
# 6. TOOL: BÚSQUEDA EN MANUALES (CON CITAS Y FILTRO DE VIGENCIA)
# =====================================================================


class BuscarManualesInput(StrictModel):
    consulta: str = Field(min_length=2, description="Pregunta o términos a buscar en los manuales de la planta")
    equipo: str | None = Field(default=None, description="TAG o modelo de equipo opcional (ej. 'B-101', 'TX-350', 'FC-48')")
    top_k: int = Field(default=5, ge=1, le=20, description="Cantidad máxima de secciones a devolver")
    incluir_no_vigentes: bool = Field(
        default=False,
        description="Por defecto False para excluir la revisión vencida TX350-C y secciones que no aplican a PIA-3",
    )


class SeccionManualOut(StrictModel):
    cita: str = Field(description="Cita canónica de la sección, ej. 'TX350-D §3.2'")
    documento: str
    seccion: str
    titulo: str
    vigente: bool
    puntaje: float
    texto: str


class BuscarManualesOutput(StrictModel):
    consulta: str
    resultados: list[SeccionManualOut]


# =====================================================================
# IMPLEMENTACIÓN DE LAS 6 TOOLS DE SOLO LECTURA
# =====================================================================


def _normalizar_texto(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower().strip())
    return "".join(c for c in t if not unicodedata.combining(c))


def _tokenizar_bm25(texto: str) -> list[str]:
    parada = {
        "a", "al", "algo", "como", "con", "cual", "cuales", "cuando", "de", "del",
        "el", "ella", "en", "es", "esta", "este", "esto", "hay", "la", "las", "le",
        "lo", "los", "mas", "me", "mi", "muy", "no", "o", "otra", "otras", "para",
        "pero", "por", "que", "se", "si", "sin", "sobre", "su", "sus", "un", "una",
        "uno", "y", "ya",
    }
    norm = _normalizar_texto(texto)
    return [t[:6] for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", norm) if t not in parada]


def _formatear_opciones_equipos(equipos: list[EquipoOut]) -> list[str]:
    """Genera etiquetas legibles con TAG y primer nombre informal para mostrar en la tablet."""
    return [
        f"{e.tag} ({e.nombres_informales[0]})" if e.nombres_informales else e.tag
        for e in equipos
    ]


def _resolver_equipos_por_consulta(consulta: str, todos: list[EquipoOut]) -> ConsultarEquiposOutput:
    """Función auxiliar que resuelve equipos contra /api/planta y solicita aclaración si es ambiguo o indeterminado."""
    q = _normalizar_texto(consulta)
    articulos = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al"}
    q_tokens = [
        tok.rstrip("s") if len(tok) > 4 and tok.endswith("s") else tok
        for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", q)
        if tok not in articulos
    ]
    q_nucleo = " ".join(w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", q) if w not in articulos)

    # 1. Búsqueda por identificador específico: TAG exacto o alias de nombres_informales
    especificos: list[EquipoOut] = []
    for eq in todos:
        tag_norm = _normalizar_texto(eq.tag)
        nombres_norm = [_normalizar_texto(n) for n in eq.nombres_informales]
        nucleos_alias = [
            " ".join(w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", n) if w not in articulos)
            for n in nombres_norm
        ]
        if (
            q == tag_norm
            or tag_norm in q.split()
            or q in nombres_norm
            or (q_nucleo and q_nucleo in nucleos_alias)
            or any(len(n) >= 4 and re.search(rf"\b{re.escape(n)}\b", q) for n in nombres_norm)
        ):
            especificos.append(eq)

    if len(especificos) == 1:
        return ConsultarEquiposOutput(equipos=especificos, ambiguo=False)

    if len(especificos) > 1:
        tags_str = ", ".join(e.tag for e in especificos)
        return ConsultarEquiposOutput(
            equipos=especificos,
            ambiguo=True,
            mensaje=f"Referencia ambigua ('{consulta}'): coincide con múltiples equipos ({tags_str}).",
            pregunta_aclaracion=f"Mencionaste '{consulta}', ¿podrías especificar a cuál equipo te referís ({tags_str})?",
            opciones_aclaracion=_formatear_opciones_equipos(especificos),
        )

    # 2. Búsqueda por atributos compartidos (tipo, descripción, modelo, fabricante)
    candidatos_puntaje: list[tuple[int, EquipoOut]] = []
    for eq in todos:
        atributos_texto = " ".join(
            filter(None, [eq.tipo, eq.descripcion, eq.modelo, eq.fabricante, " ".join(eq.nombres_informales)])
        )
        attr_tokens = {
            tok.rstrip("s") if len(tok) > 4 and tok.endswith("s") else tok
            for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", _normalizar_texto(atributos_texto))
            if tok not in articulos
        }
        coincidencias_tok = sum(1 for tok in q_tokens if tok in attr_tokens)
        if coincidencias_tok > 0:
            candidatos_puntaje.append((coincidencias_tok, eq))

    # 3. Si no se pudo determinar a qué equipo se hace referencia, pedimos que se especifique mejor
    if not candidatos_puntaje:
        tags_todos = ", ".join(e.tag for e in todos)
        return ConsultarEquiposOutput(
            equipos=[],
            ambiguo=True,
            mensaje=f"No se pudo determinar a qué equipo se hace referencia con '{consulta}'.",
            pregunta_aclaracion=(
                f"No pude determinar a qué equipo te referís con '{consulta}'. "
                f"¿Podrías especificar cuál es el equipo? ({tags_todos})"
            ),
            opciones_aclaracion=_formatear_opciones_equipos(todos),
        )

    max_hits = max(score for score, _ in candidatos_puntaje)
    mejores = [eq for score, eq in candidatos_puntaje if score == max_hits]
    if len(mejores) == 1:
        return ConsultarEquiposOutput(equipos=mejores, ambiguo=False)

    tags_str = ", ".join(e.tag for e in mejores)
    return ConsultarEquiposOutput(
        equipos=mejores,
        ambiguo=True,
        mensaje=f"Referencia ambigua ('{consulta}'): puede referirse a {tags_str}.",
        pregunta_aclaracion=f"¿A cuál de estos equipos te referís con '{consulta}' ({tags_str})?",
        opciones_aclaracion=_formatear_opciones_equipos(mejores),
    )


class PlantaTools:
    """Conjunto de herramientas de solo lectura con manejo de errores, reintentos y contratos estrictos."""

    def __init__(self, api_client: PlantaAPIClient | None = None):
        self.api = api_client or PlantaAPIClient()

    # ----------------------------------------------------------------- 1. Equipos
    def consultar_equipos(self, datos: ConsultarEquiposInput) -> ConsultarEquiposOutput:
        planta = self.api.obtener_planta()
        sensores_por_tag: dict[str, list[LimiteSensorOut]] = {}
        for s in planta.get("sensores", []):
            sensores_por_tag.setdefault(s["tag"], []).append(
                LimiteSensorOut(
                    variable=s["variable"],
                    unidad=s["unidad"],
                    alarma_baja=s.get("alarma_baja"),
                    alarma_alta=s.get("alarma_alta"),
                    disparo=s.get("disparo"),
                )
            )

        todos: list[EquipoOut] = [
            EquipoOut(
                tag=eq["tag"],
                tipo=eq["tipo"],
                descripcion=eq["descripcion"],
                fabricante=eq.get("fabricante"),
                modelo=eq.get("modelo"),
                nombres_informales=eq.get("nombres", []),
                unidad_manometro_local=(eq.get("manometro_local") or {}).get("unidad"),
                rol_inicial=eq.get("rol_inicial"),
                sensores=sensores_por_tag.get(eq["tag"], []),
            )
            for eq in planta.get("equipos", [])
        ]

        if not datos.consulta:
            return ConsultarEquiposOutput(equipos=todos)

        return _resolver_equipos_por_consulta(datos.consulta, todos)

    # ----------------------------------------------------------------- 2. Historial de Sensores
    def consultar_historial_sensores(self, datos: ConsultarHistorialSensoresInput) -> ConsultarHistorialSensoresOutput:
        muestras = self.api.obtener_historian(datos.tag, datos.variable, desde=datos.desde, hasta=datos.hasta)
        valores_validos = [float(m["valor"]) for m in muestras if m.get("valor") is not None]

        if not valores_validos:
            return ConsultarHistorialSensoresOutput(
                tag=datos.tag,
                variable=datos.variable,
                cantidad_muestras=len(muestras),
                tendencia="sin_datos",
                muestras_recientes=[],
            )

        recientes_raw = muestras[-datos.ultimas_muestras :]
        valores_recientes = [float(m["valor"]) for m in recientes_raw if m.get("valor") is not None]

        # Cálculo de tendencia comparando primera mitad vs segunda mitad de la ventana reciente
        tendencia = "estable"
        if len(valores_recientes) >= 6:
            mitad = len(valores_recientes) // 2
            prom_inicio = sum(valores_recientes[:mitad]) / mitad
            prom_fin = sum(valores_recientes[mitad:]) / len(valores_recientes[mitad:])
            delta = prom_fin - prom_inicio
            umbral = max(0.05, abs(prom_inicio) * 0.03)
            if delta > umbral:
                tendencia = "subiendo"
            elif delta < -umbral:
                tendencia = "bajando"

        return ConsultarHistorialSensoresOutput(
            tag=datos.tag,
            variable=datos.variable,
            cantidad_muestras=len(muestras),
            minimo_historico=round(min(valores_validos), 3),
            maximo_historico=round(max(valores_validos), 3),
            promedio_reciente=round(sum(valores_recientes) / len(valores_recientes), 3) if valores_recientes else None,
            ultimo_valor=round(valores_validos[-1], 3),
            tendencia=tendencia,
            muestras_recientes=[
                MuestraSensorOut(ts=str(m["ts"]), valor=float(m["valor"]) if m.get("valor") is not None else None)
                for m in recientes_raw[-15:]
            ],
        )

    # ----------------------------------------------------------------- 3. Historial de Órdenes
    def consultar_historial_ordenes(self, datos: ConsultarHistorialOrdenesInput) -> ConsultarHistorialOrdenesOutput:
        ordenes_raw = self.api.obtener_historial_ordenes(
            equipo=datos.equipo, desde=datos.desde, hasta=datos.hasta
        )
        recientes = ordenes_raw[-datos.limite :]
        salida: list[OrdenHistoricaOut] = []
        for o in recientes:
            repuestos_raw = o.get("repuestos") or []
            repuestos_str = [
                r if isinstance(r, str) else f"{r.get('codigo', '')}x{r.get('cantidad', 1)}"
                for r in repuestos_raw
            ]
            salida.append(
                OrdenHistoricaOut(
                    id=str(o.get("id", "")),
                    fecha=str(o.get("fecha") or o.get("fecha_aviso") or ""),
                    equipo=str(o.get("equipo", "")),
                    tarea=o.get("tarea"),
                    falla=o.get("falla"),
                    nota=o.get("nota") or o.get("descripcion") or o.get("texto"),
                    repuestos=repuestos_str,
                    horas=float(o["horas"]) if o.get("horas") is not None else None,
                    origen=o.get("origen"),
                )
            )
        return ConsultarHistorialOrdenesOutput(total_encontradas=len(ordenes_raw), ordenes=salida)

    # ----------------------------------------------------------------- 4. Inventario, Stock y BOM
    def consultar_inventario_y_bom(self, datos: ConsultarInventarioInput) -> ConsultarInventarioOutput:
        mat = self.api.obtener_materiales()
        bom_map: dict[str, list[str]] = mat.get("lista_materiales", {})
        repuestos_raw: list[dict[str, Any]] = mat.get("repuestos", [])

        codigos_permitidos: set[str] | None = None
        bom_equipo: list[str] = []
        if datos.equipo:
            bom_equipo = list(bom_map.get(datos.equipo, []))
            codigos_permitidos = set(bom_equipo)

        salida: list[RepuestoInventarioOut] = []
        for r in repuestos_raw:
            cod = r["codigo"]
            if codigos_permitidos is not None and cod not in codigos_permitidos:
                continue
            if datos.codigo_repuesto and cod.upper() != datos.codigo_repuesto.upper():
                continue
            stock = r.get("stock") or {}
            salida.append(
                RepuestoInventarioOut(
                    codigo=cod,
                    nombre=r["nombre"],
                    unidad=r["unidad"],
                    costo_usd=float(r["costo_usd"]),
                    stock_yacimiento=int(stock.get("yacimiento", 0)),
                    stock_central=int(stock.get("central", 0)),
                )
            )

        return ConsultarInventarioOutput(
            equipo=datos.equipo,
            lista_materiales_equipo=bom_equipo,
            repuestos=salida,
        )

    # ----------------------------------------------------------------- 5. Catálogo de Tareas
    def consultar_catalogo_tareas(self, datos: ConsultarCatalogoTareasInput) -> ConsultarCatalogoTareasOutput:
        mat = self.api.obtener_materiales()
        salida: list[TareaCatalogoOut] = []

        for t in mat.get("tareas_estandar", []):
            if datos.equipo and datos.equipo not in t.get("equipos", []):
                continue
            if datos.codigo_tarea and t["codigo"].upper() != datos.codigo_tarea.upper():
                continue
            salida.append(
                TareaCatalogoOut(
                    codigo=t["codigo"],
                    nombre=t["nombre"],
                    estandar=True,
                    equipos=list(t.get("equipos", [])),
                    detiene=bool(t["detiene"]),
                    horas=float(t["horas"]),
                    ejecutor=str(t["ejecutor"]),
                    repuestos=[
                        RepuestoTareaOut(codigo=r["codigo"], cantidad=float(r["cantidad"]))
                        for r in t.get("repuestos", [])
                    ],
                )
            )

        if not datos.solo_estandar:
            for t in mat.get("tareas_no_estandar", []):
                if datos.codigo_tarea and t["codigo"].upper() != datos.codigo_tarea.upper():
                    continue
                salida.append(
                    TareaCatalogoOut(
                        codigo=t["codigo"],
                        nombre=t["nombre"],
                        estandar=False,
                        equipos=list(t.get("equipos", ["B-101", "B-102", "B-103"])),
                        detiene=bool(t["detiene"]),
                        horas=float(t["horas"]),
                        ejecutor=str(t["ejecutor"]),
                        repuestos=[],
                    )
                )

        return ConsultarCatalogoTareasOutput(tareas=salida)

    # ----------------------------------------------------------------- 6. Búsqueda en Manuales
    def buscar_en_manuales(self, datos: BuscarManualesInput) -> BuscarManualesOutput:
        from sistema.rag.buscador import BuscadorRAG
        from sistema.rag.indexador import IndexadorRAG

        fragmentos = self.api.obtener_fragmentos_corpus()
        # Filtramos órdenes si se buscan secciones de manuales/contratos, o delegamos al buscador híbrido
        candidatos = [f for f in fragmentos if f.get("tipo") != "orden"]
        if not candidatos:
            return BuscarManualesOutput(consulta=datos.consulta, resultados=[])

        indexador = IndexadorRAG(fragmentos_crudos=candidatos)
        buscador = BuscadorRAG(indexador=indexador)
        encontrados = buscador.buscar_fragmentos(
            pregunta=datos.consulta,
            k=datos.top_k,
            filtro_equipo=datos.equipo,
            solo_vigentes=not datos.incluir_no_vigentes,
        )

        resultados: list[SeccionManualOut] = [
            SeccionManualOut(
                cita=str(f["cita"]),
                documento=str(f["doc"]),
                seccion=str(f["numero"]),
                titulo=str(f["titulo"]),
                vigente=bool(f["vigente"]),
                puntaje=float(f["puntaje"]),
                texto=str(f["texto"]),
            )
            for f in encontrados
        ]
        return BuscarManualesOutput(consulta=datos.consulta, resultados=resultados)

    # ----------------------------------------------------------------- Exportación a LangChain Tools
    def obtener_langchain_tools(self) -> list[StructuredTool]:
        """Devuelve las 6 tools empaquetadas como StructuredTool de LangChain con args_schema estricto."""
        return [
            StructuredTool.from_function(
                func=lambda **kwargs: self.consultar_equipos(ConsultarEquiposInput(**kwargs)).model_dump(),
                name="consultar_equipos",
                description="Consulta equipos de la planta, nombres informales de campo, manómetros locales y límites de alarma.",
                args_schema=ConsultarEquiposInput,
            ),
            StructuredTool.from_function(
                func=lambda **kwargs: self.consultar_historial_sensores(ConsultarHistorialSensoresInput(**kwargs)).model_dump(),
                name="consultar_historial_sensores",
                description="Consulta el historial de un sensor (45 días), su rango histórico (mín/máx) y su tendencia.",
                args_schema=ConsultarHistorialSensoresInput,
            ),
            StructuredTool.from_function(
                func=lambda **kwargs: self.consultar_historial_ordenes(ConsultarHistorialOrdenesInput(**kwargs)).model_dump(),
                name="consultar_historial_ordenes",
                description="Consulta el historial de órdenes de trabajo cerradas y del simulador por equipo y fechas.",
                args_schema=ConsultarHistorialOrdenesInput,
            ),
            StructuredTool.from_function(
                func=lambda **kwargs: self.consultar_inventario_y_bom(ConsultarInventarioInput(**kwargs)).model_dump(),
                name="consultar_inventario_y_bom",
                description="Consulta el stock de repuestos en el almacén del yacimiento y central, costos en USD y lista de materiales (BOM) por equipo.",
                args_schema=ConsultarInventarioInput,
            ),
            StructuredTool.from_function(
                func=lambda **kwargs: self.consultar_catalogo_tareas(ConsultarCatalogoTareasInput(**kwargs)).model_dump(),
                name="consultar_catalogo_tareas",
                description="Consulta el catálogo de tareas estándar y no estándar de mantenimiento, duración, ejecutor y si detiene el equipo.",
                args_schema=ConsultarCatalogoTareasInput,
            ),
            StructuredTool.from_function(
                func=lambda **kwargs: self.buscar_en_manuales(BuscarManualesInput(**kwargs)).model_dump(),
                name="buscar_en_manuales",
                description="Busca en los manuales vigentes de la planta (TX350-D, FC48-3, MBP-4, CT-SVP-2025) y devuelve secciones con su cita exacta.",
                args_schema=BuscarManualesInput,
            ),
        ]
