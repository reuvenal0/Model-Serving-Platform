"""Verify the actual running development API; never delete or undeploy data."""

import argparse
import json
import math

import httpx


def predictions_match(body: dict, expected: list[float], version: str = "v1") -> bool:
    values = body.get("predictions")
    latency = body.get("latency_ms")
    return (
        body.get("model_name") == "demo-linear" and body.get("version") == version
        and type(latency) in (float, int) and math.isfinite(latency) and latency >= 0
        and isinstance(values, list) and len(values) == len(expected)
        and all(type(value) in (int, float) and math.isclose(value, target, rel_tol=1e-8, abs_tol=1e-8)
                for value, target in zip(values, expected))
    )


def verify(base_url: str, deployment_id: str, include_routing: bool = False) -> bool:
    checks = [
        ("GET", "/health", None, lambda body: body.get("status") == "ok"),
        ("GET", f"/deployments/{deployment_id}/health", None,
         lambda body: body.get("ready") is True and body.get("status") == "running"
         and body.get("deployment_id") == deployment_id),
        ("POST", "/predict/demo-linear", {"version": "v1", "inputs": [[4.0]]},
         lambda body: predictions_match(body, [9.0])),
        ("POST", "/predict/demo-linear/batch", {"version": "v1", "inputs": [[4.0], [5.0]]},
         lambda body: predictions_match(body, [9.0, 11.0])),
    ]
    if include_routing:
        checks.extend([
            ("GET", "/routing/demo-linear", None,
             lambda body: body.get("model_name") == "demo-linear"
             and type(body.get("revision")) is int and body["revision"] >= 1
             and isinstance(body.get("weights"), dict) and bool(body["weights"])
             and all(type(weight) is int and 0 <= weight <= 100 for weight in body["weights"].values())
             and sum(body["weights"].values()) == 100),
            ("POST", "/predict/demo-linear", {"inputs": [[4.0]]},
             lambda body: body.get("version") in ("v1", "v2")
             and predictions_match(body, [9.0 if body["version"] == "v1" else 13.0], body["version"])),
        ])
    passed = True
    with httpx.Client(base_url=base_url, timeout=30.0, trust_env=False) as client:
        for method, path, payload, validate in checks:
            try:
                response = client.request(method, path, json=payload) if payload else client.request(method, path)
                print(method, path, "HTTP", response.status_code)
                try:
                    body = response.json()
                except ValueError:
                    print("Non-JSON response:", response.text[:500])
                    valid = False
                else:
                    print(json.dumps(body, indent=2))
                    valid = response.status_code == 200 and isinstance(body, dict) and validate(body)
            except httpx.RequestError as error:
                print(method, path, "HTTP unavailable:", error)
                valid = False
            print("PASS" if valid else "FAIL")
            passed = passed and valid
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--deployment-id", default="demo-api-deployment")
    parser.add_argument("--routing", action="store_true", help="Also verify configured v1/v2 automatic routing without mutating it")
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.base_url, args.deployment_id, args.routing) else 1)


if __name__ == "__main__":
    main()
