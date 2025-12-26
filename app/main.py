from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os

from app.config import settings
from app.state import get_bundle
from app.routers import query as query_router

app = FastAPI(title="RAG Hybrid API", version="1.0.0")

origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"

# --------------------------
#  API ROUTING FIX (CORE FIX)
# --------------------------

# Global default client API
#app.include_router(query_router.router, prefix="/api")

# Dynamic tenant API:
# /maran/api/ask
# /rsms/api/history
app.include_router(query_router.router, prefix="/{client_id}/api")


# --------------------------
#  DOCUMENT SERVING
# --------------------------

DOCS_BASE_PATH = Path(os.getenv("BASE_DIR", r"C:\sms"))

@app.get("/{client_id}/docs/{filename:path}")
def serve_document(client_id: str, filename: str):
    base = (DOCS_BASE_PATH / client_id / "documents").resolve()
    full = (base / filename).resolve()

    if not str(full).startswith(str(base)) or not full.is_file():
        logger.error(f"Document not found: {full}")
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info(f"Serving: {full}")
    return FileResponse(str(full))


# --------------------------
#  SPA ROUTER (MUST BE LAST)
# --------------------------

def _index_file() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

@app.get("/", response_class=HTMLResponse)
def serve_root():
    return _index_file()

# IMPORTANT: ALWAYS KEEP THIS LAST
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve index.html for any unknown route (after API routes)."""
    return _index_file()
