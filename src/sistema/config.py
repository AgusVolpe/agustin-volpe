"""Configuración centralizada (Pydantic Settings): MQTT, API sim, Proxy, Postgres y modelos."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Identificación y planta
    sistema_nombre: str = "agustin-volpe"
    mqtt_host: str = "broker"
    mqtt_port: int = 1883
    sim_url: str = "http://sim:8100"
    enlace_control_url: str = "http://enlace:8081"

    # Base de datos
    database_url: str = "postgresql+psycopg://planta:planta_secret@postgres:5432/planta_db"

    # Modelos locales (PC de planta - 100% offline)
    ollama_base_url: str = "http://ollama:11434"
    llm_local_model: str = "qwen2.5:3b"
    whisper_model_size: str = "small"

    # Modelos en la nube (NVIDIA NIM - Visión multimodal saliendo por enlace:3128)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    vlm_model: str = "google/gemma-4-31b-it"
    vlm_fallback_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    nvidia_timeout_s: float = 25.0

    # Interfaz humana (FastAPI para planificador, supervisor e inspector)
    api_ui_port: int = 8090

    # Tiempos y presupuestos (1.4 Control)
    latido_intervalo_s: int = 25          # < 30s para margen seguro con el corrector (máx 60s)
    timeout_global_nota_s: float = 25.0   # < 30s reales para alertas de seguridad
    max_intentos_tool: int = 3
    timeout_tool_s: float = 5.0

    @property
    def postgres_dsn(self) -> str:
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


settings = Settings()


def get_settings() -> Settings:
    return settings