# Política de Memoria y Persistencia en PostgreSQL

## 1. Qué se recuerda (Esquema de Persistencia)

El sistema persiste en PostgreSQL 16 (`pgvector/pgvector:pg16`, volumen `postgres_data`) tres estructuras operativas:

1. **Memoria de Conversación por Nota (`conversaciones_notas`)**:
   - `nota_id` (PK), `recorrida_id`, `ts_nota`, `operador`, `audio_url`, `fotos`.
   - `texto_transcripto`: Texto obtenido por el modelo STT local (`faster-whisper`).
   - `estado`: `"Borrador"` mientras espera confirmación o corrección en la tablet; `"Confirmado"` cuando el operador lo valida.
   - `intento`: Contador de turnos de interacción con la tablet (corte en 3 intentos).
   - `ultimo_mensaje_tablet_id`: Índice de correlación rápida cuando llega `RespuestaOperador` en `planta/respuestas`.
   - `filas` (`JSONB`): Lista actual de objetos `FilaParte`.
   - `ambiguedad_pendiente` (`JSONB`): Detalle de la ambigüedad consultada al operador (`tipo: "equipo"` o `"valor"`).

2. **Registro Idempotente de Eventos (`eventos_planta`)**:
   - `evento_id` (PK idempotente: `EV-{nota_id}-{tag}` o `EV-SEG-{nota_id}`).
   - `sincronizado` (`BOOLEAN`): `FALSE` cuando el evento `Escalado` o `AlertaSeguridad` ocurre durante un corte de enlace (`enlace_ok = False`), permitiendo informar el contador exacto de `pendientes_sincronizar` en `sistema/estado` y sincronizar en orden sin duplicar cuando vuelve el enlace.

3. **Base Vectorial de Documentos (`documentos_rag`)**:
   - Fragmentos de manuales vigentes (`TX350-D`, `FC48-3`, `MBP-4`, `CT-SVP-2025`) e historial con sus embeddings y metadatos (`equipo`, `fabricante`, `documento`, `seccion`, `cita`, `vigente`).

## 2. Por cuánto tiempo se recuerda (Política de Retención)

- **Memoria episódica de conversación de una nota (`Borrador`)**: Activa durante la ventana de diálogo con la tablet (hasta su confirmación o el límite de 3 intentos). Una vez pasada a `"Confirmado"`, se conserva en PostgreSQL durante **45 días simulados** (alineado con la ventana del historiador de sensores de la planta) para auditoría y trazabilidad.
- **Eventos y Órdenes pendientes**: Persistidos en disco en el volumen `postgres_data` sin expiración mientras estén abiertos, garantizando que sobrevivan a cortes de energía (`sin-luz`) y reinicios abruptos de la PC de planta (`academia.planta=pc`).
