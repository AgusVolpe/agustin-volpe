FROM python:3.11-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg build-essential && rm -rf /var/lib/apt/lists/*

# Instalar planta_contrato y copiar datos de referencia de solo lectura del simulador
RUN git clone --depth 1 https://github.com/NovitSoftware/academia2026-tp.git /tmp/tp \
    && pip install /tmp/tp/contrato \
    && mkdir -p /opt/planta_datos \
    && cp /tmp/tp/datos/materiales.yaml /tmp/tp/datos/planta.yaml /tmp/tp/datos/documentos/fragmentos.jsonl /tmp/tp/datos/documentos/evaluacion/retrieval.json /tmp/tp/datos/documentos/contrato/cierres_ejemplo.json /opt/planta_datos/ \
    && rm -rf /tmp/tp

WORKDIR /app

# Instalar dependencias y pre-descargar faster-whisper ANTES de copiar el código fuente
# para aprovechar la caché de capas de Docker en cada rebuild de desarrollo
COPY pyproject.toml README.md ./
RUN mkdir -p src/sistema && touch src/sistema/__init__.py \
    && pip install --no-cache-dir ".[dev]" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8', download_root='/opt/whisper')"

# Copiar el código fuente local e instalar el paquete sin reinstalar dependencias
COPY . .
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "sistema.agente"]
