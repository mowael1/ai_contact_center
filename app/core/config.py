from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Supabase Postgres connection string, e.g.
    # postgresql://postgres.<ref>:<password>@aws-1-<region>.pooler.supabase.com:5432/postgres
    DATABASE_URL: str
    
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    VONAGE_APPLICATION_ID: str = ""
    VONAGE_PRIVATE_KEY_PATH: str = ""
    VONAGE_NUMBER: str = ""
    VOICE_TO_NUMBER: str = ""
    VOICE_ANSWER_URL: str = ""
    
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    
    @property
    def database_uri(self) -> str:
        """SQLAlchemy URL, forcing the psycopg (v3) driver."""
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
    
settings = Settings()
