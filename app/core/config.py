from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Database ----------------------------------------------------------
    # Preferred: a full SQLAlchemy/Postgres URL (Supabase gives you one), e.g.
    # postgresql://postgres.<ref>:<password>@aws-1-<region>.pooler.supabase.com:5432/postgres
    DATABASE_URL: str = ""

    # Legacy SQL Server settings, used only when DATABASE_URL is empty.
    DB_SERVER: str = ""
    DB_NAME: str = ""
    DB_DRIVER: str = ""
    # When set, SQL authentication is used instead of Windows integrated auth,
    # which cannot work from macOS or Linux.
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None

    # ---- Auth --------------------------------------------------------------
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ---- Voice (Vonage) ----------------------------------------------------
    VONAGE_APPLICATION_ID: str = ""
    VONAGE_PRIVATE_KEY_PATH: str = ""
    VONAGE_NUMBER: str = ""
    VOICE_TO_NUMBER: str = ""
    VOICE_ANSWER_URL: str = ""

    PUBLIC_BASE_URL: str = "http://localhost:8000"

    # ---- LLM ---------------------------------------------------------------
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
