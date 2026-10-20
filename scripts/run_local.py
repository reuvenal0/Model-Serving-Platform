"""Supported local-development launcher: loopback only, no trusted proxies."""

import uvicorn
from app.config import Settings
from app.main import create_app


def main() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host="127.0.0.1",
        port=8000,
        workers=1,
        proxy_headers=False,
        timeout_graceful_shutdown=50,
    )


if __name__ == "__main__":
    main()
