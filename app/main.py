from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="AI Contact Center API",
    version="1.0.0"
)

app.include_router(
    api_router,
    prefix="/api/v1"
)

@app.get("/")
def root():
    return {
        "message": "AI Contact Center API is running"
    }