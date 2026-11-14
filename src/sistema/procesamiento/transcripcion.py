"""Transcripción local de notas de voz (WAV 16kHz mono -> texto en español).
Usa faster-whisper en CPU (int8) con vocabulario de planta para funcionar 100% sin enlace a internet."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from sistema.config import settings

log = logging.getLogger(__name__)

PROMPT_VOCABULARIO_PLANTA = (
    "Recorrida en planta de inyección de agua: la uno, la dos, la tres, la de la punta, "
    "la del medio, la del fondo, la de reserva, la bomba, el filtro A, el filtro B, "
    "el primario, el secundario, el de la izquierda, el de la derecha, el pulmón, "
    "ciento doce kilos, mil seiscientos psi, bar, de descarga, me dio uno coma ocho de diferencial, "
    "dieciocho de diferencial, sigue goteando por la empaquetadura, pierde bastante por la empaquetadura, "
    "chorrea por la empaquetadura, el rodamiento me dio sesenta grados con la pistola, "
    "el visor de aceite está bajo, casi no tiene aceite en el visor, las correas están flojas, chillan, "
    "hace un golpeteo metálico, suena como si tuviera piedras adentro, tiene un ruido raro de rodamiento, "
    "está parada, en reserva, parada por mantenimiento, disparada, no arranca, "
    "está rebalsando a la pileta, sale agua a presión, olor a gas, sale humo del motor, sangrando."
)


class TranscriptorLocal:
    """Transcriptor STT local basado en faster-whisper sobre CPU."""

    def __init__(self, model_size: str | None = None, download_root: str = "/opt/whisper"):
        self.model_size = model_size or settings.whisper_model_size
        self.download_root = download_root
        self._modelo = None

    def _asegurar_modelo(self):
        if self._modelo is None:
            from faster_whisper import WhisperModel

            root = self.download_root if Path(self.download_root).exists() else None
            log.info("Cargando modelo faster-whisper '%s' en CPU (int8)...", self.model_size)
            self._modelo = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=3,
                download_root=root,
            )
        return self._modelo

    def transcribir_wav_bytes(self, wav_bytes: bytes) -> str:
        """Transcribe bytes WAV 16kHz mono a texto."""
        if not wav_bytes:
            return ""
        modelo = self._asegurar_modelo()
        segmentos, _ = modelo.transcribe(
            io.BytesIO(wav_bytes),
            language="es",
            beam_size=2,
            best_of=2,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=PROMPT_VOCABULARIO_PLANTA,
            vad_filter=True,
        )
        texto = " ".join(seg.text.strip() for seg in segmentos if seg.text.strip())
        log.info("Audio transcripto: '%s'", texto)
        return texto
