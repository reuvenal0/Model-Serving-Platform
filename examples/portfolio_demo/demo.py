"""Bundled ONNX demo via the existing authenticated loopback HTTP API only."""

import argparse
import base64
from collections import Counter
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from scripts.ops_request import NoRedirect

MODEL = "demo-linear-onnx"
DESCRIPTION = "Portfolio ONNX demo: v1 y=2x+1; v2 y=3x+1."
ROOT = Path(__file__).parent
RESOURCES = {"cpu": 1.0, "memory_mb": 128}
BASELINE = {"v1": 100, "v2": 0}
ALLOCATIONS = [BASELINE, {"v1": 90, "v2": 10}, {"v1": 50, "v2": 50}]


def api(method, path, body=None, *, missing_ok=False):
    """No proxies, redirects, remote URL option, or credentials in diagnostics."""
    key = os.environ.get("ADMIN_API_KEY")
    if not key:
        raise RuntimeError("Configure ADMIN_API_KEY before starting the container.")
    request = Request(
        "http://127.0.0.1:8000" + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404 and missing_ok:
            return None
        raise RuntimeError(f"Demo API request rejected (HTTP {error.code}).") from None
    except (URLError, TimeoutError, ValueError):
        raise RuntimeError("Demo API request could not be completed.") from None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def version_body(version):
    content = (ROOT / f"{version}.onnx").read_bytes()
    tensor = {"shape": [None, 1], "dtype": "number"}
    return {
        "version": version, "onnx_base64": base64.b64encode(content).decode("ascii"),
        "input_schema": {"portfolio_sha256": sha256(content).hexdigest(),
                         "inference": tensor},
        "output_schema": {"inference": {"tensors": {"Y": tensor}}},
    }


def setup(call=api):
    """Preflight reserved identities before creating anything; never overwrite."""
    model = next((item for item in call("GET", "/models") if item["name"] == MODEL), None)
    require(model is None or model["description"] == DESCRIPTION,
            "Reserved model name has incompatible metadata; nothing was overwritten.")
    versions = {item["version"]: item for item in call("GET", f"/models/{MODEL}/versions")} if model else {}
    require(set(versions) <= {"v1", "v2"}, "Unexpected demo versions; refusing changes.")
    deployments = {item["id"]: item for item in call("GET", "/deployments")}
    routing = call("GET", f"/routing/{MODEL}", missing_ok=True) if model else None
    require(routing is None or routing["weights"] in ALLOCATIONS,
            "Existing operator routing is outside demo allocations; refusing changes.")
    for version in ("v1", "v2"):
        body = version_body(version)
        if version in versions:
            existing = versions[version]
            require(existing["framework"] == "onnx" and existing["status"] == "registered"
                    and all(existing[field] == body[field] for field in ("input_schema", "output_schema")),
                    f"Incompatible {version} metadata; refusing to replace its artifact.")
        deployment = deployments.get(f"portfolio-onnx-{version}")
        if deployment:
            require(deployment["model_name"] == MODEL and deployment["version"] == version
                    and deployment["resources"] == RESOURCES and deployment["replicas"] == 1
                    and deployment["status"] == "running",
                    "Reserved deployment is incompatible or stopped; refusing changes.")
    require(all(item["id"] in {"portfolio-onnx-v1", "portfolio-onnx-v2"}
                for item in deployments.values() if item["model_name"] == MODEL),
            "Unexpected deployments for the demo model; refusing changes.")
    if model is None:
        call("POST", "/models", {"name": MODEL, "description": DESCRIPTION})
    for version in ("v1", "v2"):
        if version not in versions:
            call("POST", f"/models/{MODEL}/versions", version_body(version))
        deployment_id = f"portfolio-onnx-{version}"
        if deployment_id not in deployments:
            call("POST", "/deployments", {"deployment_id": deployment_id,
                 "model_name": MODEL, "version": version, "resources": RESOURCES})
        require(call("GET", f"/deployments/{deployment_id}/health")["ready"],
                "Demo deployment did not pass readiness.")
    if routing is None:
        call("PUT", f"/routing/{MODEL}", {"weights": BASELINE, "expected_revision": 0})
    print("Setup ready: two ONNX versions; matching resources reused without replacement.")


def predict(call, version=None):
    body = {"inputs": [[4.0]]}
    if version is not None:
        body["version"] = version
    result = call("POST", f"/predict/{MODEL}", body)
    require(result["version"] in ("v1", "v2"), "Unexpected prediction destination.")
    require(version is None or result["version"] == version, "Explicit version was not respected.")
    expected = 9.0 if result["version"] == "v1" else 13.0
    require(isclose(result["predictions"]["Y"][0][0], expected, abs_tol=1e-6),
            "Unexpected ONNX prediction.")
    return result


def verify(call=api):
    for version in ("v1", "v2"):
        print("Explicit prediction:", json.dumps(predict(call, version)))
    current = call("GET", f"/routing/{MODEL}")
    require(current["weights"] == BASELINE, "Final routing must be 100% v1.")
    require(predict(call)["version"] == "v1", "Restored routing did not select v1.")
    print("Final routing:", json.dumps(current))
    print("Metrics:", json.dumps(call("GET", f"/metrics/{MODEL}")))
    print("Logs:", json.dumps(call("GET", f"/inference-logs?model_name={MODEL}&limit=5")))


def run(call=api):
    setup(call)
    for version in ("v1", "v2"):
        print("Explicit prediction:", json.dumps(predict(call, version)))
    current = call("GET", f"/routing/{MODEL}")
    if current["weights"] != BASELINE:
        current = call("PUT", f"/routing/{MODEL}",
                       {"weights": BASELINE, "expected_revision": current["revision"]})
    target = current["revision"]
    for weights in ALLOCATIONS[1:]:
        current = call("PUT", f"/routing/{MODEL}",
                       {"weights": weights, "expected_revision": current["revision"]})
        selected = [predict(call)["version"] for _ in range(12)]
        print("Configured weights:", weights, "selected:", selected, "observed:", dict(Counter(selected)))
    restored = call("POST", f"/routing/{MODEL}/rollback",
                    {"target_revision": target, "expected_revision": current["revision"]})
    require(restored["weights"] == BASELINE and restored["restored_from"] == target,
            "Rollback did not restore the baseline as a new revision.")
    print("Routing history:", json.dumps(call("GET", f"/routing/{MODEL}/history")))
    verify(call)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run", "verify"))
    args = parser.parse_args()
    try:
        {"setup": setup, "run": run, "verify": verify}[args.command]()
    except RuntimeError as error:
        # Only our bounded diagnostics, never backend bodies or exception traces.
        parser.exit(1, f"Portfolio demo failed: {error}\n")
    except OSError:
        parser.exit(1, "Portfolio demo could not read its bundled artifacts.\n")


if __name__ == "__main__":
    main()
