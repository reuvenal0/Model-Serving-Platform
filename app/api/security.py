"""Loopback-only operational API keys; no credentials enter diagnostics."""

from hmac import compare_digest
from ipaddress import ip_address
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from urllib.parse import urlsplit

key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Loopback only. Operator key reads; administrator key also mutates routing.",
)


async def require_operator(
    request: Request, key: Annotated[str | None, Security(key_header)]
) -> str:
    try:
        address = ip_address(request.client.host) if request.client else None
        local = address is not None and (
            address.is_loopback
            or bool(
                getattr(address, "ipv4_mapped", None)
                and address.ipv4_mapped.is_loopback
            )
        )
    except ValueError:
        local = False
    # Never trust forwarding headers as evidence of a local peer.
    if not local:
        raise HTTPException(403)
    settings = request.app.state.security_settings
    if len(request.headers.getlist("x-api-key")) != 1:
        raise HTTPException(401)
    candidate = key.encode("utf-8") if key is not None and len(key) <= 256 else b""
    admin = compare_digest(candidate, (settings.admin_api_key or "").encode("ascii"))
    operator = compare_digest(
        candidate, (settings.operator_api_key or "").encode("ascii")
    )
    if candidate and admin and settings.admin_api_key is not None:
        return "admin"
    if candidate and operator and settings.operator_api_key is not None:
        return "operator"
    raise HTTPException(401)


async def require_admin(role: Annotated[str, Depends(require_operator)]) -> None:
    if role != "admin":
        raise HTTPException(403)


async def require_documentation_access(request: Request) -> None:
    """Opt-in browser access is separate from operational authorization."""
    settings = request.app.state.security_settings
    if not settings.enable_dev_swagger:
        await require_operator(request, request.headers.get("x-api-key"))
        return
    try:
        peer = ip_address(request.client.host) if request.client else None
        local = peer is not None and (
            peer.is_loopback
            or bool(getattr(peer, "ipv4_mapped", None) and peer.ipv4_mapped.is_loopback)
        )
    except ValueError:
        local = False
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    origin = request.headers.get("origin")
    try:
        origin_host = urlsplit(origin).hostname if origin is not None else None
    except ValueError:
        raise HTTPException(403) from None
    if (
        not local
        or request.url.hostname not in allowed_hosts
        or request.headers.get("sec-fetch-site") == "cross-site"
        or (origin is not None and origin_host not in allowed_hosts)
    ):
        raise HTTPException(403)
