# Especificación del Flujo del Parte por Voz (Spec-Driven Development)

## 1. Objetivo
Garantizar que cada nota de voz publicada por el operador en `planta/recorridas` se transcriba localmente en la PC de planta, se valide contra el inventario de equipos (`/api/planta`) e historial (`/api/historian`), resuelva ambigüedades con el operador a través de `sistema/tablet` y solo genere eventos en `sistema/eventos` una vez que el parte pasa al estado **Confirmado** (salvo ante una condición insegura `S1`–`S4`, que dispara `AlertaSeguridad` de inmediato en $<30\text{ s}$).

## 2. Diagrama de Estados del Parte

```mermaid
stateDiagram-v2
    [*] --> Recibida: MQTT planta/recorridas (NotaRecorrida)
    Recibida --> TranscripcionLocal: Descarga WAV de http://sim:8100
    TranscripcionLocal --> AlertaSeguridad: Detecta S1..S4 (< 30s reales)
    AlertaSeguridad --> [*]: Publica EventoSistema(AlertaSeguridad) + MensajeTablet(espera_respuesta=false)

    TranscripcionLocal --> PreguntaAmbiguedad: Equipo ambiguo ("la bomba")
    PreguntaAmbiguedad --> BorradorEnTablet: RespuestaOperador aclara equipo

    TranscripcionLocal --> BorradorConOpciones: Valor dudoso ("dieciocho" vs 1.8 / fuera de rango histórico)
    TranscripcionLocal --> BorradorEnTablet: Sin ambigüedad

    BorradorConOpciones --> Confirmado: Operador confirma
    BorradorEnTablet --> Confirmado: Operador confirma ("Sí, confirmo")
    BorradorEnTablet --> BorradorCorregido: Operador corrige ("No, estaba en...")
    BorradorCorregido --> Confirmado: Operador confirma o se alcanza presupuesto (máx 3 intentos)

    Confirmado --> EvaluacionDeteccion: Reglas determinísticas en código
    EvaluacionDeteccion --> [*]: Publica EventoSistema (SinNovedad o Escalado por equipo)
```

## 3. Presupuesto y Condiciones de Corte (1.4)
- **Timeout por nota**: `< 25 s` reales (garantizando cumplir la ventana de `< 30 s` en alertas de seguridad).
- **Presupuesto de turnos de diálogo en tablet**: Máximo `3` intentos (`MAX_INTENTOS_TABLET = 3`).
- **Defensa contra Prompt Injection**: Cualquier instrucción dirigida al sistema dentro de la nota de voz (`"Nota para el sistema: ..."`) se trunca antes de extraer las filas y jamás altera las reglas de detección o aprobación.
