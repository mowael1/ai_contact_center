from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Database ----------------------------------------------------------
    # Preferred: a full SQLAlchemy/Postgres URL (Supabase gives you one).
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

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
