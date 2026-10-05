from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from fastapi.staticfiles import StaticFiles



BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
FRONTEND_PATH = BASE_DIR.parent / "frontend" / "index.html"

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
    version = FRONTEND_PATH.stat().st_mtime_ns

    return RedirectResponse(
        url=f"/static/index.html?v={version}",
        status_code=307,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Clear-Site-Data": '"cache"',
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health():
    return {
        "message": "AI Contact Center API is running"
    }
