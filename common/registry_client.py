"""HTTP client for the registry service (port 4000) used by every stage.

API (see registry_manager/main.py):
    GET  /v1/health
    GET  /v1/papers?status=...            list papers
    GET  /v1/papers/{stem}                one paper (status null if unknown)
    PUT  /v1/papers/{stem}/status         {status, domain?, published_at?, error?}
    GET  /v1/domains/{domain}/checkpoint  newest published_at for a domain
    GET  /v1/stats                        counts per status + stage durations
"""

from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from common.paths import REGISTRY_URL


class RegistryUnavailable(Exception):
    """The registry service can't be reached."""


class RegistryClient:
    def __init__(self, base_url: str = REGISTRY_URL, timeout: float = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low level ------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = requests.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except requests.RequestException as e:
            raise RegistryUnavailable(f"registry unreachable at {self.base_url}: {e}") from e
        if r.status_code >= 500:
            raise RegistryUnavailable(f"registry error {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        return r.json()

    # -- API ------------------------------------------------------------
    def health(self) -> bool:
        try:
            return self._request("GET", "/v1/health").get("status") == "ok"
        except (RegistryUnavailable, requests.RequestException):
            return False

    def get_paper(self, stem: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/papers/{quote(stem, safe='')}")

    def get_status(self, stem: str) -> Optional[str]:
        return self.get_paper(stem).get("status")

    def list_papers(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"status": status} if status else None
        return self._request("GET", "/v1/papers", params=params).get("papers", [])

    def set_status(self, stem: str, status: str, **fields) -> bool:
        body = {"status": status, **{k: v for k, v in fields.items() if v is not None}}
        return bool(self._request("PUT", f"/v1/papers/{quote(stem, safe='')}/status",
                                  json=body).get("success"))

    def report_error(self, stem: str, message: str) -> bool:
        return self.set_status(stem, "error", error=str(message)[:1000])

    def checkpoint(self, domain: str) -> Optional[str]:
        return self._request("GET", f"/v1/domains/{quote(domain, safe='')}/checkpoint").get("last_checkpoint")

    def stats(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/stats")
