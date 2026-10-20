"""Explicit credentials and loopback peer for existing HTTP contract tests.

Security-specific tests use the original Starlette client and exercise fail-closed
settings directly. No authentication dependency is bypassed here.
"""
from dataclasses import replace
from secrets import token_urlsafe
from fastapi.testclient import TestClient as BaseTestClient


class TestClient(BaseTestClient):
    __test__ = False

    def __init__(self, app, **kwargs):
        key = token_urlsafe(32)
        app.state.security_settings = replace(app.state.security_settings, admin_api_key=key)
        kwargs.setdefault("client", ("127.0.0.1", 50000))
        kwargs.setdefault("headers", {"X-API-Key": key})
        super().__init__(app, **kwargs)
