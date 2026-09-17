"""Registry service (port 4000): the single source of truth for each paper's
pipeline status. Every other stage polls it for work and reports back.

    GET  /v1/health
    GET  /v1/papers?status=embedded
    GET  /v1/papers/{stem}                 -> {..., "status": null} if unknown
    PUT  /v1/papers/{stem}/status          {status, domain?, published_at?, error?, force?}
    GET  /v1/domains/{domain}/checkpoint   -> newest published_at for that domain
    GET  /v1/stats                         -> counts, stage durations, errors
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator

from common.logsetup import get_logger
from config import STATUS_TYPE as StatusType
from registry import FileRegistry

log = get_logger("registry")
db = FileRegistry()
registry = FastAPI(title="registry_manager", version="1")


class StatusUpdate(BaseModel):
    status: StatusType
    domain: Optional[str] = None
    published_at: Optional[datetime] = None
    error: Optional[str] = None
    force: bool = False

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v


@registry.get("/v1/health")
def health():
    return {"status": "ok"}


@registry.get("/v1/papers")
def list_papers(status: Optional[str] = Query(None)):
    return {"papers": db.list_papers(status)}


@registry.get("/v1/papers/{stem}")
def get_paper(stem: str):
    return db.get_paper(stem) or {"filename": stem, "status": None}


@registry.put("/v1/papers/{stem}/status")
def set_status(stem: str, update: StatusUpdate):
    if update.status == "error" and not update.error:
        raise HTTPException(422, "status 'error' requires an 'error' message")
    ok = db.update_status(stem, update.status, domain=update.domain,
                          published_at=update.published_at, error_msg=update.error,
                          force=update.force)
    if ok:
        log.info("%s -> %s%s", stem, update.status, f" ({update.error[:80]})" if update.error else "")
    else:
        log.warning("rejected %s -> %s (unknown paper?)", stem, update.status)
    return {"success": ok}


@registry.get("/v1/domains/{domain}/checkpoint")
def checkpoint(domain: str):
    return {"domain": domain, "last_checkpoint": db.get_last_domain_date(domain)}


@registry.get("/v1/stats")
def stats():
    return db.stats()


if __name__ == "__main__":
    uvicorn.run("main:registry", host="127.0.0.1", port=4000, reload=False, log_level="warning")
