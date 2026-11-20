# Informe Técnico — Entregas 1 y 2 (Planta de Inyección de Agua PIA-3)

Este informe consolida el presupuesto de recursos de la PC de planta, la selección y benchmark real de modelos locales y en la nube, las mejoras aplicadas tras la evaluación en vivo de la **Entrega 1**, y los resultados oficiales obtenidos en la **Entrega 2** sobre los escenarios exigidos por el Anexo de Entregas.

---

## 1. Presupuesto de Recursos en la PC de Planta y Modelos Elegidos

### 1.1. Distribución de Contenedores (`academia.planta=pc`)

Límite normativo total de la PC de planta: **4.0 CPUs y 8.0 GB de RAM** (`docker-compose.yml`):

| Servicio | Imagen | Límite CPU | Límite RAM | Rol |
|---|---|---:|---:|---|
| **`postgres`** | `pgvector/pgvector:pg16` | `0.5 CPU` | `1.0 GB` | Persistencia durable (conversaciones, eventos, órdenes, checkpoints, cola, trazas) e índice vectorial `pgvector` |
| **`ollama`** | `Dockerfile.ollama` (`qwen2.5:3b` pre-horneado) | `2.0 CPU` | `4.0 GB` | Inferencia LLM local 100% offline; sobrevive a `docker compose down -v` entre escenarios |
| **`sistem_agent`** | `Dockerfile` (Python 3.11 + `faster-whisper`) | `1.5 CPU` | `3.0 GB` | Coordinador MQTT/HTTP, STT local, LangGraph, Reglas en código, Bloque G y FastAPI UI (`:8090`) |
| **TOTAL** | — | **`4.0 CPU`** | **`8.0 GB`** | Cumple exactamente la restricción de hardware de planta |

- **Hardware de prueba**: Windows 11 Pro (x86_64) con Docker Desktop (WSL2 Linux containers) bajo los límites de recursos declarados en `docker-compose.yml` sobre la red aislada `planta-lan`.

### 1.2. Modelos Elegidos por Tarea

| Tarea | Modelo / Motor | Tamaño / Cuantización | Dónde Corre | Justificación |
|---|---|---|---|---|
| **1. Transcripción STT (Recorridas y Cierres)** | `faster-whisper` (`small`) | ~244M · `int8` en CPU (~480 MB) | Local (`sistem_agent`, `/opt/whisper`, `local_files_only=True`) | Precargado al arranque; 100% offline ante cortes de enlace; 0.8–1.6 s por audio con `initial_prompt` de jerga PIA-3 |
| **2. Detección S1–S4, Límites, Sensores y Reglas D** | Código puro (`DetectorPlanta`, `reglas_aprobacion.py`, `RevisorOrden`) | 0 parámetros LLM | Local (`sistem_agent`) | Garantiza `< 30 s` en seguridad (real: ~1–10 s con STT), 0 aprobaciones indebidas, inmunidad a *prompt injection* y cálculo exacto de $\Delta P$ corregida (`FC48-3 §3.2`) |
| **3. Paso Agéntico de Análisis de Anomalías** | **`qwen2.5:3b`** vía Ollama (`/api/chat`, `format="json"`) | ~3B · Q4_K_M (~1.9 GB) | Local (`ollama:11434`, 100% offline) | Pre-horneado en `Dockerfile.ollama`; excelente seguimiento de JSON estructurado en español dentro de 2 CPU / 4 GB RAM, respaldado por consolidador determinista ante SLA estricto de `10 s` |
| **4. Análisis Multimodal de Fotos (VLM)** | Primario: **`google/gemma-4-31b-it`** · Fallback: **`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`** | Cloud Free Tier (NVIDIA NIM) | Nube vía `HTTP_PROXY=http://enlace:3128` | Validado en vivo contra fotos del simulador (`rotura_descarga`, `empaquetadura`); encola automáticamente cuando el enlace está caído |
| **5. Recuperación Documental e Historial (RAG)** | `BuscadorRAG` híbrido (BM25 + `pgvector` `vector(256)`) | < 15 MB RAM | Local (`postgres` + `sistem_agent`) | `mrr = 0.962`, `recall@5 = 1.000`, `distractor@5 = 0.000` (excluye Rev. C y otras plantas) |
| **6. Certificación Contractual (Bloque G)** | `evaluar_cierre_y_certificacion` (`CT-SVP-2025` / `CT-SVP-2022`) | Código puro | Local (`sistem_agent`) | Validado **12/12 exacto** sobre `cierres_ejemplo.json` (vigencia 2022 vs 2025, `SVP-06`, horas extra, repuestos no autorizados) |

---

## 2. Benchmark Real de Modelos en la Nube (NVIDIA NIM Free Tier)

Al probar el catálogo real disponible en `https://integrate.api.nvidia.com/v1/models` con cuenta gratuita, se comprobó que modelos antiguos como `llama-3.2-11b-vision-instruct` y `qwen2.5-vl` ya no existen en la API, y que `gemma-3-12b-it` / `gemma-3-4b-it` / `cosmos-reason2-8b` devuelven `HTTP 404` en la capa gratuita.

Se midieron los modelos multimodales activos con las fotos reales del simulador (`rotura_descarga/04.jpg` y `empaquetadura/01.jpg`):

| Modelo NVIDIA NIM (Free Tier) | Estado HTTP | Latencia Medida | Tokens (In / Out) | Precisión en `rotura_descarga` y `empaquetadura` | Decisión |
|---|:---:|---:|---:|:---:|---|
| **`google/gemma-4-31b-it`** | `200 OK` | `1.96 – 7.28 s` | `390 / 55` | **100% (2/2)** · JSON limpio, detecta daño estructural grave | **Primario (`VLM_MODEL`)** |
| **`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`** *(con `enable_thinking=False`)* | `200 OK` | **`0.72 – 0.97 s`** | `905 / 58–83` | **100% (2/2)** · Muy baja latencia y alta precisión | **Fallback (`VLM_FALLBACK_MODEL`)** |
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | `200 OK` | `1.85 s` | `904 / 64` | 50% (1/2) · Clasificó `rotura_descarga` como `desconocida` | Descartado |
| `mistralai/mistral-large-4-1t-a52b` | Timeout | `> 30 s` | — | Timeout en pruebas | Descartado |
| `google/gemma-3-12b-it` / `gemma-3-4b-it` / `cosmos-reason2-8b` | `404` | — | — | No habilitados en cuenta free | No disponibles |

### Consumo y Costo Operativo Consolidado

| Componente | Latencia Medida | Tokens (In / Out) | Costo Real (USD) |
|---|---:|---:|---:|
| STT local (`faster-whisper small int8`) | `0.8 – 1.6 s` / audio | 0 | `$0.00` |
| Seguridad S1–S4 y Parte por Voz | `< 2 ms` (post-STT) | 0 | `$0.00` |
| RAG Híbrido (`BuscadorRAG` + `pgvector`) | `71.2 ms` / consulta | 0 | `$0.00` |
| Análisis LLM local (`ollama qwen2.5:3b`) | SLA máx. `10 s` (fallback determinista si supera budget CPU) | Local (trazado en `trazas_llm`) | `$0.00` |
| Análisis VLM nube (`gemma-4-31b-it` / `nemotron-3-nano-omni`) | `0.7 – 7.3 s` / foto | `390–905 in / 55–83 out` | `$0.00` (Free Tier) |

---

## 3. Resultados de la Entrega 1 (RAG y Corrección en Vivo v3)

### 3.1. Evaluación RAG (71 preguntas de `retrieval.json` sobre 701 fragmentos)

| Métrica | Baseline BM25 (`corpus/evaluar.py`) | Nuestro `BuscadorRAG` (`pgvector` + BM25 + Metadatos) | Mejora |
|---|---:|---:|---:|
| **`mrr`** | `0.573` | **`0.962`** | **`+0.389`** |
| **`hit@1`** | `0.352` | **`0.930`** | **`+0.578`** |
| **`hit@3`** | `0.775` | **`1.000`** | **`+0.225`** |
| **`recall@5`** | `0.664` | **`1.000`** | **`+0.336`** |
| **`distractor@5`** | `0.099` | **`0.000`** | **`-0.099` (0 trampas)** |

### 3.2. Ajustes y Resultado en Vivo Entrega 1 (`corregir.py` v3)

Tras la primera corrida de Entrega 1, se aplicaron 4 mejoras clave:
1. **`local_files_only=True` + `precargar()` en `TranscriptorLocal`**: evitó que `faster-whisper` intentara consultar HuggingFace cuando el enlace estaba caído en `fuga-sin-enlace`.
2. **Segmentación por comas y normalización fonética en `ProcesadorParteVoz`**: protegiendo decimales y `"parada, en reserva"`, elevó la confirmación al primer intento (`A2`) en notas con valores numéricos al **100%**.
3. **Pantalla explícita en `AlertaSeguridad`**: el aviso de tablet (`espera_respuesta=False`) incluye `Pantalla(filas=[FilaParte(...)])`, llevando `A1` al **100%**.
4. **Optimización de capas del `Dockerfile`**: `pip install` y la descarga de `faster-whisper` se ejecutan antes de `COPY . .`, reduciendo los rebuilds de desarrollo de ~3 min a ~5 s.

- **Puntaje verificado Entrega 1 (`martes-tranquilo` + `fuga-sin-enlace`)**: **`91.3 / 100 · APRUEBA`** (0 faltas eliminatorias, 0 mensajes inválidos, 100% en `A1`, `A4a`, `A4b`, `B1`, `B2`, `E1–E3`, `F1–F2`).

---

## 4. Resultados Oficiales de la Entrega 2 (`corregir.py`)

Se evaluó el sistema completo sobre los **4 escenarios oficiales de la Entrega 2** (`tormenta-perfecta`, `sin-luz`, `contratista` y `fuga-sin-enlace`, según el Anexo Entrega 2 pág. 4) y sobre los **5 escenarios acumulados de Entrega 1 + Entrega 2** (sumando `martes-tranquilo`).

### 4.1. Evolución de Iteraciones en Entrega 2

| Iteración | Principales Mejoras Aplicadas |
|---|---|
| **v1** | Primera corrida completa; `D1=100%`, `G1=100%`; detectados timeouts de Ollama en cola (`E2=50%`) y falsas alarmas de bomba parada `B-103`. |
| **v2 – v3** | Eliminadas falsas alarmas de bombas paradas (`presion_descarga < 15 kg/cm²`), precarga de Ollama en hilo paralelo (evitando perder notas tras reinicio en `sin-luz`), y normalización de homófonos de Whisper (`el A1`, `lados`, `stein`, `filto no ve`, `fito`, `golpequeo`). |
| **v4** | Corregida prioridad `F02` vs `F03` (evitando clasificar rodamiento normal de `66 °C` como `F03`) y reseteo de `_episodio_activo` $\rightarrow$ **`D2=100%`**. |
| **v5** | Consulta de `/api/estado` en vivo dentro de `_nodo_reglas` (`D1=100%`), salida JSON compacta de Ollama con timeout total estricto y latido inmediato ante encolado offline $\rightarrow$ **`E2=100%`**. |
| **Final (v5 + `sin-luz` FC48-3)** | Corrección de $\Delta P$ por caudal (`FC48-3 §3.2`: $\Delta P_{\text{corr}} = \Delta P / (Q/2400)^2$) usando `FT-401` + SLA `10 s` en Ollama $\rightarrow$ **`sin-luz = 100.0 / 100`** (`F3=100%`). |

### 4.2. Puntaje Final Oficial por Criterio

- **4 escenarios oficiales de Entrega 2** (`tormenta-perfecta`, `sin-luz`, `contratista`, `fuga-sin-enlace`): **`95.9 / 100 · APRUEBA`** (`0` eliminatorias, `0` mensajes inválidos)
- **5 escenarios acumulados Entrega 1 + Entrega 2** (`+ martes-tranquilo`): **`96.0 / 100 · APRUEBA`** (`0` eliminatorias, `0` mensajes inválidos)

| Criterio | Peso | Entrega 2 (4 escenarios) | Acumulado E1+E2 (5 escenarios) | Puntos (E2 / E1+E2) |
|---|---:|---:|---:|---:|
| **A. Parte por voz** | | | | |
| `A1` Cobertura | 5 | `97%` | `98%` | `4.86` / `4.88` |
| `A2` Borrador al primer intento | 10 | `79%` | `79%` | `7.91` / `7.88` |
| `A3` Parte final correcto | 8 | `82%` | `83%` | `6.56` / `6.67` |
| `A4a` Pregunta cuando hace falta | 4 | **`100%`** | **`100%`** | **`4.00`** / **`4.00`** |
| `A4b` No pregunta de más | 3 | **`100%`** | **`100%`** | **`3.00`** / **`3.00`** |
| **B. Detección en planta** | | | | |
| `B1` Sin novedad / Escalado | 10 | `98%` | `99%` | `9.83` / `9.86` |
| `B2` Alerta de seguridad | 10 | **`100%`** | **`100%`** | **`10.00`** / **`10.00`** |
| **D. Órdenes de trabajo** | | | | |
| `D1` Reglas de aprobación | 15 | **`100%`** | **`100%`** | **`15.00`** / **`15.00`** |
| `D2` Orden adecuada a la falla | 10 | **`100%`** | **`100%`** | **`10.00`** / **`10.00`** |
| **E. Operación sin enlace** | | | | |
| `E1` Sigue andando sin enlace | 3 | **`100%`** | **`100%`** | **`3.00`** / **`3.00`** |
| `E2` Pendientes visibles | 3 | **`100%`** | **`100%`** | **`3.00`** / **`3.00`** |
| `E3` Sin duplicados | 4 | **`100%`** | **`100%`** | **`4.00`** / **`4.00`** |
| **F. Robustez** | | | | |
| `F1` Mensajes válidos | 2 | **`100%`** | **`100%`** | **`2.00`** / **`2.00`** |
| `F2` Latido | 1 | **`100%`** | **`100%`** | **`1.00`** / **`1.00`** |
| `F3` Sobrevive al corte de luz (`sin-luz`) | 2 | **`100%`** | **`100%`** | **`2.00`** / **`2.00`** |
| **G. Cierre y certificación** | | | | |
| `G1` Orden ejecutada (`planta/cierres`) | 3 | **`100%`** | **`100%`** | **`3.00`** / **`3.00`** |
| `G2` Cierre propio *(UI humana `:8090`)* | 2 | *(revisión manual)* | *(revisión manual)* | – |
| `G3` Certificación contratista *(UI humana `:8090`)* | 5 | *(revisión manual)* | *(revisión manual)* | – |

- **Faltas eliminatorias**: **0** (`aprobacion_indebida = 0`, `sin_alerta_de_seguridad = 0`, `sistema_mudo = 0`).
- **Validación Bloque G (`cierres_ejemplo.json`)**: **12 / 12 casos exactos** en `lista_aplicada` (`CT-SVP-2022` vs `CT-SVP-2025`), `item_contrato`, `monto_habilita_usd`, `estado` (`Certificada` vs `Observada`) y detección de diferencias.
- **Suite de tests automatizados (`pytest`)**: **48 / 48 tests pasando** dentro de la imagen Docker.

> **Nota técnica sobre `puntuar.py` del repositorio base**: durante la corrección se observó que en `academia2026-tp/correccion/puntuar.py:312-314` la expresión `ok = (...) or (... and r["payload"].get("motivo"))` devuelve el `str` del campo `motivo` cuando una orden está correctamente en `PendienteAprobacion`, provocando `TypeError: unsupported operand type(s) for +=: 'int' and 'str'` en `bien += ok`. Sin modificar en absoluto el repositorio base `academia2026-tp`, la puntuación de los archivos JSON generados por `corregir.py` se computó envolviendo `bool(ok)` externamente sobre la misma lógica y rúbrica oficiales.

---

## 5. Registro Consolidado de Decisiones Técnicas

1. **Separación estricta entre Workflow Fijo y Paso Agéntico (LangGraph)**: el LLM local propone hipótesis y borrador de orden, pero nunca escribe ni aprueba directamente; toda salida pasa obligatoriamente por `RevisorOrden` (citas vigentes + BOM + catálogo) y por `evaluar_reglas_aprobacion` (código puro con estado en vivo de `/api/estado`).
2. **Pre-horneado offline de modelos en Docker**: tanto `faster-whisper small int8` (`/opt/whisper` en `Dockerfile`) como `qwen2.5:3b` (`Dockerfile.ollama`) quedan embebidos en las imágenes Docker, asegurando operación 100% offline y resistencia al `docker compose down -v` que ejecuta `corregir.py` entre escenarios.
3. **Detección predictiva de colmatación corregida por caudal (`FC48-3 §3.2`)**: al normalizar la presión diferencial de filtros por el caudal medido en `FT-401` ($\Delta P_{\text{corr}} = \Delta P / (Q / 2400)^2$), se elimina la onda sinusoidal diaria de caudal y se detecta limpiamente `F07` en `F-201B` antes del corte de energía de `sin-luz`, permitiendo validar `F3 = 100%` sin falsos positivos en filtros sanos.
4. **Aislamiento de tráfico LAN vs. Proxy HTTP (`PLANTA_HTTP_PROXY`)**: todos los clientes internos (`PlantaAPIClient`, `ClienteLLMLocal`, `AgentePlanta`) utilizan `trust_env=False` para `sim:8100`, `enlace:8081` y `ollama:11434`, reservando `HTTP_PROXY=http://enlace:3128` únicamente para las llamadas salientes a NVIDIA NIM (`vision.py`).
5. **Trazabilidad y Durabilidad por `correlation_id`**: cada incidente enlaza `evento_id -> orden_id -> cierre_id` en `checkpoints_workflow`, `ordenes_trabajo`, `cola_eventos_analisis` y `trazas_llm`, permitiendo auditar desde la API (`:8090`) qué modelo, herramientas, citas y reglas intervinieron en cada decisión.
