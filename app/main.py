from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from fastapi.staticfiles import StaticFiles



BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"

app = FastAPI(
    title="AI Contact Center API",
    version="1.0.0"
)

app.mount(
    "/audio",
    StaticFiles(
        directory=str(AUDIO_DIR)
    ),
    name="audio",
)


app.include_router(
    api_router,
    prefix="/api/v1"
)


app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "frontend")),
    name="static",
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