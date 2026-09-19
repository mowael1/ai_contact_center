from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

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
def frontend():
    frontend_path = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "index.html"
    )

    return FileResponse(frontend_path)


@app.get("/health")
def health():
    return {
        "message": "AI Contact Center API is running"
    }