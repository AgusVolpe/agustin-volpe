FROM python:3.11-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg build-essential && rm -rf /var/lib/apt/lists/*

# Instalar planta_contrato y copiar datos de referencia de solo lectura del simulador
RUN git clone --depth 1 https://github.com/NovitSoftware/academia2026-tp.git /tmp/tp \
    && pip install /tmp/tp/contrato \
    && mkdir -p /opt/planta_datos \
    && cp /tmp/tp/datos/materiales.yaml /tmp/tp/datos/planta.yaml /tmp/tp/datos/documentos/fragmentos.jsonl /tmp/tp/datos/documentos/evaluacion/retrieval.json /opt/planta_datos/ \
    && rm -rf /tmp/tp

# Copiar el código local e instalar el paquete junto con dependencias de test
WORKDIR /app
COPY . .
RUN pip install ".[dev]"

# Pre-descargar el modelo faster-whisper en la imagen para que funcione 100% offline sin depender del enlace
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8', download_root='/opt/whisper')"

# Definir el comando de arranque
CMD ["python", "-m", "sistema.agente"]