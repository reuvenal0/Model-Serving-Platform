"""Build and smoke-test only uniquely named temporary containers and volumes.

Run with Docker's Linux engine. No persistent development paths are mounted and
no API key appears in a CLI argument, image layer, or diagnostic. Default mode
builds an owned image; --image uses an existing image without deleting it.
"""

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class DockerSmoke:
    def __init__(self, image: str | None = None) -> None:
        self.token = uuid4().hex
        self.image = image or f"msp-smoke:{self.token}"
        self.owns_image = image is None
        self.built_image = False
        self.volumes = [
            f"msp-smoke-db-{self.token}",
            f"msp-smoke-artifacts-{self.token}",
        ]
        self.created_volumes: list[str] = []
        self.containers: list[str] = []
        self.env = dict(
            os.environ,
            ADMIN_API_KEY=secrets.token_urlsafe(32),
            OPERATOR_API_KEY=secrets.token_urlsafe(32),
        )
        self.opener = build_opener(ProxyHandler({}))

    def docker(
        self, *args: str, input_data: bytes | None = None, timeout: int = 180
    ) -> str:
        result = subprocess.run(
            ["docker", *args],
            input=input_data,
            capture_output=True,
            env=self.env,
            timeout=timeout,
            cwd=ROOT,
        )
        if result.returncode:
            # CLI output can contain operator configuration; do not echo it.
            raise RuntimeError(
                f"Docker {args[0]} failed; inspect Docker availability/configuration separately"
            )
        return result.stdout.decode("utf-8").strip()

    def volume_args(self) -> list[str]:
        return [
            "--mount",
            f"type=volume,src={self.volumes[0]},dst=/data/db",
            "--mount",
            f"type=volume,src={self.volumes[1]},dst=/data/artifacts",
        ]

    def storage_environment(self, trusted: bool) -> list[str]:
        return [
            "--env",
            "DATABASE_URL=sqlite:////data/db/smoke.db",
            "--env",
            "ARTIFACT_STORAGE_DIR=/data/artifacts",
            "--env",
            "MSP_CONTAINER_SMOKE=1",
            "--env",
            f"TRUST_SKLEARN_ARTIFACTS={str(trusted).lower()}",
            "--env",
            "ADMIN_API_KEY",
            "--env",
            "OPERATOR_API_KEY",
        ]

    def start(self, trusted: bool) -> tuple[str, str]:
        name = f"msp-smoke-api-{len(self.containers)}-{self.token}"
        self.containers.append(name)
        self.docker(
            "run",
            "-d",
            "--name",
            name,
            "--platform",
            "linux/amd64",
            "--init",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,nosuid,size=64m",
            "--publish",
            "127.0.0.1::8000",
            *self.volume_args(),
            *self.storage_environment(trusted),
            self.image,
        )
        ports = json.loads(
            self.docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", name)
        )["8000/tcp"]
        assert len(ports) == 1 and ports[0]["HostIp"] == "127.0.0.1"
        base = "http://127.0.0.1:" + ports[0]["HostPort"]
        deadline = time.monotonic() + 90
        while True:
            try:
                if self.request(base, "/health")[0] == 200:
                    break
            except (URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Container liveness did not become available")
            time.sleep(0.2)
        assert (
            self.docker("exec", name, "python", "-c", "import os; print(os.getuid())")
            == "10001"
        )
        health_deadline = time.monotonic() + 30
        while (
            self.docker("inspect", "--format", "{{.State.Health.Status}}", name)
            != "healthy"
        ):
            if time.monotonic() >= health_deadline:
                raise RuntimeError("Docker health check did not become healthy")
            time.sleep(0.2)
        return name, base

    def request(
        self,
        base: str,
        path: str,
        *,
        payload: dict | None = None,
        key: bool = False,
        forwarded: bool = False,
    ) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-API-Key"] = self.env["ADMIN_API_KEY"]
        if forwarded:
            headers["X-Forwarded-For"] = "127.0.0.1"
        request = Request(
            base + path,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def management(self, name: str, path: str) -> dict:
        output = self.docker("exec", name, "python", "scripts/ops_request.py", path)
        status, body = output.split("\n", 1)
        assert status.strip() == "HTTP 200"
        return json.loads(body)

    def stop(self, name: str) -> None:
        self.docker("stop", "--time", "60", name)
        # Uvicorn restores SIGTERM after graceful cleanup; Tini reports 143.
        assert self.docker("inspect", "--format", "{{.State.ExitCode}}", name) in (
            "0",
            "143",
        )
        assert "Application shutdown complete." in self.container_logs(name)
        self.docker("rm", name)
        self.containers.remove(name)

    def container_logs(self, name: str) -> str:
        result = subprocess.run(
            ["docker", "logs", name], capture_output=True, timeout=30
        )
        if result.returncode:
            raise RuntimeError("Container diagnostics could not be retrieved")
        return (result.stdout + result.stderr).decode("utf-8", errors="replace")

    def check_network(self, name: str, base: str) -> None:
        for path in (
            "/metrics",
            "/runtime/resources",
            "/runtime/admission",
            "/routing/container-smoke",
            "/docs",
            "/openapi.json",
        ):
            assert self.request(base, path)[0] == 403
            assert self.request(base, path, key=True, forwarded=True)[0] == 403
        # A genuine container-local peer still needs credentials.
        code = (
            "import urllib.request,urllib.error; "
            "\ntry: urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8000/metrics')"
            "\nexcept urllib.error.HTTPError as e: print(e.code)"
        )
        assert self.docker("exec", name, "python", "-c", code) == "401"
        code = (
            "import http.client,os; "
            "\nfor keys in [('wrong-key',),(os.environ['ADMIN_API_KEY'],'wrong-key')]:"
            "\n c=http.client.HTTPConnection('127.0.0.1',8000); c.putrequest('GET','/metrics')"
            "\n for key in keys: c.putheader('X-API-Key',key)"
            "\n c.endheaders(); r=c.getresponse(); assert r.status==401; r.read(); c.close()"
            "\nprint('PASS')"
        )
        assert self.docker("exec", name, "python", "-c", code) == "PASS"
        code = (
            "import os; os.environ['APP_ENV']='development'; os.environ['ENABLE_DEV_SWAGGER']='true'; "
            "os.environ['MSP_SECURE_CONTAINER']='false'; from app.main import create_app; "
            "\ntry: create_app()"
            "\nexcept ValueError: print('REJECTED')"
            "\nelse: raise RuntimeError('Unsafe container Swagger configuration was allowed')"
        )
        assert self.docker("exec", name, "python", "-c", code) == "REJECTED"
        assert self.management(name, "/runtime/admission")["active"] == 0
        for role, expected in [("ADMIN_API_KEY", 404), ("OPERATOR_API_KEY", 403)]:
            code = (
                "import json,os,urllib.request,urllib.error; "
                "r=urllib.request.Request('http://127.0.0.1:8000/routing/smoke-never-created',"
                "data=json.dumps({'weights':{'v1':100},'expected_revision':0}).encode(),method='PUT',"
                f"headers={{'X-API-Key':os.environ[{role!r}],'Content-Type':'application/json'}}); "
                "\ntry: urllib.request.build_opener(urllib.request.ProxyHandler({})).open(r)"
                "\nexcept urllib.error.HTTPError as e: print(e.code)"
            )
            assert self.docker("exec", name, "python", "-c", code) == str(expected)

    def native_onnx(self, name: str) -> None:
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "one"], ["Y"])],
            "smoke",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])],
            [numpy_helper.from_array(np.array([1], dtype=np.float32), "one")],
        )
        artifact = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        ).SerializeToString()
        code = (
            "import sys,numpy as np,onnxruntime as ort; "
            "session=ort.InferenceSession(sys.stdin.buffer.read(),providers=['CPUExecutionProvider']); "
            "assert np.allclose(session.run(None,{'X':np.array([[4.]],dtype=np.float32)})[0],[[5.]]); print('PASS')"
        )
        assert (
            self.docker("exec", "-i", name, "python", "-c", code, input_data=artifact)
            == "PASS"
        )

    def exercise(self) -> None:
        if self.docker("info", "--format", "{{.OSType}}") != "linux":
            raise RuntimeError("A running Linux Docker engine is required")
        if self.owns_image:
            self.docker(
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                self.image,
                ".",
                timeout=900,
            )
            self.built_image = True
        print(
            "Image built"
            if self.built_image
            else "Using supplied image (build skipped)",
            flush=True,
        )
        config = json.loads(
            self.docker("image", "inspect", "--format", "{{json .Config}}", self.image)
        )
        assert config["User"] == "10001:10001"
        assert "TRUST_SKLEARN_ARTIFACTS=false" in config["Env"]
        assert not any(
            item.startswith(("ADMIN_API_KEY=", "OPERATOR_API_KEY="))
            for item in config["Env"]
        )
        for volume in self.volumes:
            self.docker(
                "volume", "create", "--label", f"msp.smoke={self.token}", volume
            )
            self.created_volumes.append(volume)
        empty, base = self.start(False)
        self.check_network(empty, base)
        assert self.management(empty, "/metrics")["total_requests"] == 0
        print(
            "Empty initialization, non-root volume permissions, health and network policy: PASS",
            flush=True,
        )
        self.stop(empty)
        seed_name = f"msp-smoke-seed-{self.token}"
        self.containers.append(seed_name)
        seeded = json.loads(
            self.docker(
                "run",
                "--rm",
                "--name",
                seed_name,
                "--platform",
                "linux/amd64",
                *self.volume_args(),
                *self.storage_environment(True),
                self.image,
                "python",
                "scripts/container_fixture.py",
                "seed",
            )
        )
        self.containers.remove(seed_name)
        assert (
            len(seeded["tables"]) == 7 and "schema_migrations" not in seeded["tables"]
        )
        previous_log_ids = set()
        for attempt in range(2):
            name, base = self.start(True)
            self.check_network(name, base)
            current_logs = self.management(name, "/inference-logs")["logs"]
            assert current_logs and previous_log_ids <= {
                log["request_id"] for log in current_logs
            }
            # Check the cache BEFORE any endpoint can trigger readiness loading.
            assert self.management(name, "/runtime/resources")["cache"]["loads"] == 1
            assert (
                self.management(name, "/deployments/smoke-running/health")["ready"]
                is True
            )
            interrupted = self.management(name, "/deployments/smoke-interrupted/health")
            assert interrupted["status"] == "failed" and interrupted["ready"] is False
            assert self.management(name, "/routing/container-smoke")["weights"] == {
                "v1": 100
            }
            for path, inputs, expected in [
                ("/predict/container-smoke", [[4.0]], [9.0]),
                ("/predict/container-smoke/batch", [[4.0], [5.0]], [9.0, 11.0]),
            ]:
                status, body = self.request(
                    base, path, payload={"version": "v1", "inputs": inputs}
                )
                assert status == 200 and len(body["predictions"]) == len(expected)
                assert all(
                    math.isclose(a, b, abs_tol=1e-8)
                    for a, b in zip(body["predictions"], expected)
                )
            assert (
                self.request(
                    base, "/predict/container-smoke", payload={"inputs": [[4.0]]}
                )[0]
                == 200
            )
            fingerprint = json.loads(
                self.docker(
                    "exec",
                    name,
                    "python",
                    "scripts/container_fixture.py",
                    "fingerprint",
                )
            )
            assert fingerprint == seeded
            self.native_onnx(name)
            assert (
                self.management(name, "/metrics/container-smoke")["total_requests"] >= 4
            )
            print(
                f"Recreation {attempt + 1}: sklearn/ONNX predictions, persisted metadata/artifacts/history, startup recovery: PASS",
                flush=True,
            )
            previous_log_ids = {
                log["request_id"]
                for log in self.management(name, "/inference-logs")["logs"]
            }
            logs = self.container_logs(name)
            assert (
                self.env["ADMIN_API_KEY"] not in logs
                and self.env["OPERATOR_API_KEY"] not in logs
            )
            self.stop(name)

    def cleanup(self) -> list[str]:
        # Only unique names allocated by this instance; never compose down -v,
        # prune, bind-mount cleanup, or removal of user-provided images.
        commands = [["docker", "rm", "--force", name] for name in self.containers]
        commands += [
            ["docker", "volume", "rm", volume] for volume in self.created_volumes
        ]
        if self.built_image:
            commands.append(["docker", "image", "rm", self.image])
        failures = []
        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, timeout=75)
                if result.returncode:
                    failures.append("Owned smoke resource cleanup was not confirmed")
            except (OSError, subprocess.TimeoutExpired):
                failures.append("Owned smoke resource cleanup was not confirmed")
        return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", help="Use an already built image; never remove that image"
    )
    args = parser.parse_args()
    smoke = DockerSmoke(args.image)
    completed = False
    try:
        smoke.exercise()
        completed = True
    finally:
        failures = smoke.cleanup()
        if failures:
            print(
                "Smoke cleanup incomplete; inspect only resources labeled msp.smoke",
                file=sys.stderr,
            )
            if completed:
                raise RuntimeError("Smoke cleanup was not confirmed")
    print("Container smoke verification passed; owned temporary resources removed.")


if __name__ == "__main__":
    main()
