"""Fresh-clone acceptance: clean source copy, unique Compose project, disposable volumes."""

import json
import argparse
import os
from pathlib import Path
import shutil
import secrets
import subprocess
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def source_copy(destination: Path) -> None:
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    for name in files:
        source = ROOT / name
        if not name or not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assert not (destination / "model_serving.db").exists()
    assert not (destination / "artifacts").exists()
    assert not (destination / ".venv").exists()
    assert not list(destination.glob(".env*"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="After proving empty startup, explicitly seed and verify the documented demo",
    )
    args = parser.parse_args()
    project = "msp-fresh-" + uuid4().hex
    image = project + ":test"
    env = dict(os.environ, APP_PORT="0", TRUST_SKLEARN_ARTIFACTS="false")
    for key in (
        "ADMIN_API_KEY",
        "OPERATOR_API_KEY",
        "CONTAINER_DATABASE_URL",
        "CONTAINER_ARTIFACT_STORAGE_DIR",
    ):
        env.pop(key, None)
    opener = build_opener(ProxyHandler({}))
    with TemporaryDirectory(prefix="msp-fresh-source-") as directory:
        root = Path(directory)
        source_copy(root)
        override = root / "smoke-override.json"
        override.write_text(json.dumps({"services": {"api": {"image": image}}}))
        command = [
            "docker",
            "compose",
            "--project-name",
            project,
            "--file",
            str(root / "compose.yaml"),
            "--file",
            str(override),
        ]

        def compose(*args, timeout=900, input_data=None):
            result = subprocess.run(
                command + list(args),
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                input=input_data,
                timeout=timeout,
            )
            if result.returncode:
                raise RuntimeError(
                    "Isolated fresh Compose verification failed: " + args[0]
                )
            return result.stdout.strip()

        try:
            compose("config", "--quiet")
            compose("up", "--build", "--detach", "--wait", "--wait-timeout", "120")
            container = compose("ps", "--quiet", "api")
            port = compose("port", "api", "8000").strip()
            with opener.open("http://" + port + "/health", timeout=5) as response:
                assert response.status == 200
            for path in ("/docs", "/openapi.json", "/metrics"):
                try:
                    opener.open("http://" + port + path, timeout=5)
                    raise AssertionError("Published management access was allowed")
                except HTTPError as error:
                    assert error.code == 403
            code = (
                "import sqlite3,json,os; c=sqlite3.connect('file:/data/db/model_serving.db?mode=ro',uri=True); "
                "t=[r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'\")]; "
                "assert len(t)==7 and 'schema_migrations' not in t; assert c.execute('SELECT COUNT(*) FROM models').fetchone()[0]==0; "
                "assert os.getuid()==10001; print('PASS: final schema, empty registry, non-root volumes')"
            )
            print(compose("exec", "--no-TTY", "api", "python", "-c", code), flush=True)
            print(
                "Fresh source copy -> docker compose up --build -> Healthy; no keys or existing data: PASS",
                flush=True,
            )
            if args.demo:
                compose("stop", "api")
                env.update(
                    ADMIN_API_KEY=secrets.token_urlsafe(32),
                    OPERATOR_API_KEY=secrets.token_urlsafe(32),
                    TRUST_SKLEARN_ARTIFACTS="true",
                )
                compose(
                    "run",
                    "--rm",
                    "--no-deps",
                    "api",
                    "python",
                    "-m",
                    "examples.seed_development_model",
                )
                compose("up", "--detach", "--force-recreate", "--wait", "api")

                def management(path, *extra, input_data=None):
                    result = compose(
                        "exec",
                        "--no-TTY",
                        "api",
                        "python",
                        "scripts/ops_request.py",
                        path,
                        *extra,
                        input_data=input_data,
                    )
                    assert result.startswith("HTTP 200")
                    return json.loads(result.split("\n", 1)[1])

                from urllib.request import Request

                def predict():
                    base = "http://" + compose("port", "api", "8000")
                    request = Request(
                        base + "/predict/demo-linear",
                        data=b'{"version":"v1","inputs":[[4.0]]}',
                        headers={"Content-Type": "application/json"},
                    )
                    with opener.open(request, timeout=10) as response:
                        assert abs(json.load(response)["predictions"][0] - 9.0) < 1e-8

                predict()
                management(
                    "/routing/demo-linear",
                    "--method",
                    "PUT",
                    "--data-stdin",
                    input_data='{"weights":{"v1":100},"expected_revision":0}',
                )
                first_ids = {
                    log["request_id"]
                    for log in management("/inference-logs", "--role", "operator")[
                        "logs"
                    ]
                }
                compose("up", "--detach", "--force-recreate", "--wait", "api")
                assert management("/runtime/resources")["cache"]["loads"] == 1
                assert management("/deployments/demo-api-deployment/health")["ready"]
                assert management("/routing/demo-linear", "--role", "operator")[
                    "weights"
                ] == {"v1": 100}
                assert first_ids <= {
                    log["request_id"]
                    for log in management("/inference-logs", "--role", "operator")[
                        "logs"
                    ]
                }
                predict()
                assert (
                    management("/metrics/demo-linear", "--role", "operator")[
                        "total_requests"
                    ]
                    == 2
                )
                container = compose("ps", "--quiet", "api")
                print(
                    "Explicit documented demo seeding, roles, routing, logs, restart and prediction 9.0: PASS",
                    flush=True,
                )
            compose("stop", "api", timeout=90)
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.ExitCode}}", container],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            assert result.stdout.strip() in ("0", "143")
            assert "Application shutdown complete." in compose(
                "logs", "--no-color", "api"
            )
        finally:
            # Explicit UUID project in an isolated source directory, never the
            # normal project. Only this test's newly allocated volumes are removed.
            compose("down", "--volumes", "--remove-orphans", timeout=120)
            subprocess.run(
                ["docker", "image", "rm", image], capture_output=True, timeout=60
            )
    print("Fresh installation resources removed.")


if __name__ == "__main__":
    main()
