"""Supported local-development launcher: loopback only, no trusted proxies."""

import uvicorn
import os
from pathlib import Path
from app.config import Settings
from app.main import create_app


def load_local_env() -> None:
    """Read only cwd/.env as literal assignments; never execute or interpolate it."""
    path = Path.cwd() / ".env"
    if not path.is_file():
        return
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key.isascii() or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid .env assignment on line {number}")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Invalid .env quoting on line {number}")
            value = value[1:-1]
        values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


def main() -> None:
    load_local_env()
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
