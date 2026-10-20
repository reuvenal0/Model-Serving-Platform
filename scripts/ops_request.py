"""Authenticated management through genuine container-local loopback only."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the management credential to another origin.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="Relative operational path, e.g. /runtime/admission"
    )
    parser.add_argument("--method", choices=["GET", "PUT", "POST"], default="GET")
    parser.add_argument(
        "--role",
        choices=["operator", "admin"],
        help="Choose a configured key without printing it",
    )
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--data", help="JSON body; never supply credentials here")
    data.add_argument(
        "--data-stdin",
        action="store_true",
        help="Read JSON from stdin; safe for PowerShell native quoting",
    )
    args = parser.parse_args()
    if (
        not args.path.startswith("/")
        or args.path.startswith("//")
        or any(c in args.path for c in ("#", "?", "\\", "\r", "\n"))
    ):
        parser.error("Use an absolute API path without a URL, query, or fragment")
    key = (
        os.getenv(args.role.upper() + "_API_KEY")
        if args.role
        else os.getenv("ADMIN_API_KEY") or os.getenv("OPERATOR_API_KEY")
    )
    if not key:
        parser.exit(2, "No operational API key is configured in this container.\n")
    try:
        if args.data_stdin:
            args.data = sys.stdin.read(1048577)
            if len(args.data) > 1048576:
                raise ValueError("Management JSON exceeds limit")
        payload = (
            json.dumps(json.loads(args.data)).encode()
            if args.data is not None
            else None
        )
        request = Request(
            "http://127.0.0.1:8000" + args.path,
            data=payload,
            method=args.method,
            headers={"X-API-Key": key, "Content-Type": "application/json"},
        )
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=30) as response:
            print("HTTP", response.status)
            print(json.dumps(json.load(response), indent=2))
    except HTTPError as error:
        # Do not print URLs, request headers, or unverified response bodies.
        print("HTTP", error.code, "operational request rejected", file=sys.stderr)
        raise SystemExit(1) from None
    except (URLError, TimeoutError, ValueError):
        print("Operational request could not be completed", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
