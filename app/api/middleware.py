"""Enforce the real HTTP body byte limit before JSON/Pydantic parsing."""

from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.errors import error_response


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
