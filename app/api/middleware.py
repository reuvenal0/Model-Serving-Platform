"""Enforce the real HTTP body byte limit before JSON/Pydantic parsing."""

from starlette.types import ASGIApp, Receive, Scope, Send
import logging

from app.api.errors import error_response


class SanitizedErrorMiddleware:
    """Contain unexpected HTTP exceptions before Starlette re-raises to servers.

    Domain/HTTP errors are handled by the existing inner exception handlers.
    Never attach a backend exception or its context to an internal diagnostic.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception:
            logging.getLogger(__name__).error("Unhandled API error: request could not be completed")
            if started:
                # Headers cannot be replaced; abort without the private cause.
                raise RuntimeError("HTTP response could not be completed") from None
            await error_response(500, "internal_error", "Request could not be completed")(scope, receive, send)


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit():
                await error_response(422, "invalid_request", "Invalid Content-Length header")(scope, receive, send)
                return
            significant = lengths[0].lstrip(b"0") or b"0"
            maximum = str(self.max_body_bytes).encode("ascii")
            if len(significant) > len(maximum) or (len(significant) == len(maximum) and significant > maximum):
                await error_response(413, "request_too_large", "Request body exceeds configured limit")(scope, receive, send)
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                await error_response(413, "request_too_large", "Request body exceeds configured limit")(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
