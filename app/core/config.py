from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DB_SERVER: str
    DB_NAME: str
    DB_DRIVER: str
    
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    VONAGE_APPLICATION_ID: str
    VONAGE_PRIVATE_KEY_PATH: str
    VONAGE_NUMBER: str
    VOICE_TO_NUMBER: str
    VOICE_ANSWER_URL: str
    
    PUBLIC_BASE_URL: str
    
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
    
settings = Settings()
