"""Packaging/security contracts checked without requiring a Docker engine."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from scripts.check_project import layer_errors
from scripts.container_fixture import require_isolated_storage
from scripts.container_smoke import DockerSmoke
from scripts.lock_dependencies import wheel_entry
from scripts.ops_request import NoRedirect

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def compose():
    return yaml.safe_load((ROOT / "compose.yaml").read_text())


def test_compose_has_one_loopback_service_and_private_persistent_volumes(compose):
    assert set(compose["services"]) == {"api"}
    api = compose["services"]["api"]
    assert api["ports"] == ["127.0.0.1:${APP_PORT:-8000}:8000"]
    assert api["volumes"] == ["metadata:/data/db", "artifacts:/data/artifacts"]
    assert set(compose["volumes"]) == {"metadata", "artifacts"}
    assert api["platform"] == "linux/amd64"
    assert api["init"] is True
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["stop_grace_period"] == "60s"
    assert api["pids_limit"] == 256
    assert api["restart"] == "unless-stopped"
    assert "CONTAINER_MEMORY" in api["mem_limit"] and "CONTAINER_CPUS" in api["cpus"]


def test_compose_secrets_are_runtime_only_and_trust_defaults_off(compose):
    env = compose["services"]["api"]["environment"]
    assert env["ADMIN_API_KEY"] is None and env["OPERATOR_API_KEY"] is None
    assert env["TRUST_SKLEARN_ARTIFACTS"].endswith(":-false}")
    assert (
        "/data/db/" in env["DATABASE_URL"]
        and "/data/artifacts" in env["ARTIFACT_STORAGE_DIR"]
    )
    assert "build" in compose["services"]["api"]
    assert set(compose["services"]["api"]["build"]) == {"context"}
    assert "/health" in compose["services"]["api"]["healthcheck"]["test"][-1]


def test_dockerfile_pins_base_and_has_safe_single_worker_factory_command():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "python:3.12.13-slim-bookworm@sha256:" in dockerfile
    assert "AS builder" in dockerfile and "AS runtime" in dockerfile
    assert "--require-hashes" in dockerfile and "--only-binary=:all:" in dockerfile
    assert "USER 10001:10001" in dockerfile and "STOPSIGNAL SIGTERM" in dockerfile
    assert "TRUST_SKLEARN_ARTIFACTS=false" in dockerfile
    assert "ADMIN_API_KEY" not in dockerfile and "OPERATOR_API_KEY" not in dockerfile
    command = json.loads(
        next(line[4:] for line in dockerfile.splitlines() if line.startswith("CMD "))
    )
    assert command[:5] == [
        "python",
        "-m",
        "uvicorn",
        "app.main:create_app",
        "--factory",
    ]
    assert command[command.index("--workers") + 1] == "1"
    assert "--no-proxy-headers" in command
    assert int(command[command.index("--timeout-graceful-shutdown") + 1]) < 60
    assert "COPY . " not in dockerfile


@pytest.mark.parametrize(
    "pattern",
    [
        "**",
        ".env",
        "**/.env",
        "**/.env.*",
        "**/*.db",
        "**/*.artifact",
        "**/artifacts/",
        "**/.git/",
        "**/.venv/",
    ],
)
def test_build_context_excludes_private_and_generated_data(pattern):
    text = (ROOT / ".dockerignore").read_text().splitlines()
    assert pattern in text or pattern == ".env" and "**/.env" in text


def lock_entries(name):
    entries = {}
    for line in (ROOT / "requirements" / f"{name}.lock").read_text().splitlines():
        if line and not line.startswith("#"):
            package, digest = line.split(" --hash=sha256:")
            distribution, version = package.split("==")
            assert len(digest) == 64 and int(digest, 16) >= 0
            assert distribution not in entries
            entries[distribution] = version
    return entries


def test_locks_pin_complete_separate_sets_and_keep_runtime_baseline():
    runtime, dev, build = (lock_entries(name) for name in ("runtime", "dev", "build"))
    assert runtime.items() <= dev.items()
    assert not {"pytest", "onnx", "ruff", "pyyaml", "httpx"}.intersection(runtime)
    assert {"pytest", "onnx", "ruff", "pyyaml", "httpx"} <= dev.keys()
    assert {"setuptools", "wheel", "pip"} <= build.keys()
    assert runtime["scikit-learn"] == "1.9.1" and runtime["onnxruntime"] == "1.31.0"
    assert runtime["numpy"] == "2.5.4"


def test_ci_has_pinned_actions_read_only_permissions_and_no_secret_dependency():
    document = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(document["on"]) == {"push", "pull_request"}
    assert document["permissions"] == {"contents": "read"}
    assert set(document["jobs"]) == {"python", "container"}
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "secrets." not in text and "pull_request_target" not in text
    assert "python -m pytest -q" in text and "python scripts/container_smoke.py" in text
    assert "python -m ruff check" in text and "python scripts/check_project.py" in text
    for job in document["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].split("@")[1]) == 40


def test_smoke_names_credentials_and_mounts_are_isolated():
    smoke = DockerSmoke("caller-provided-image:local")
    other = DockerSmoke("caller-provided-image:local")
    assert set(smoke.volumes).isdisjoint(other.volumes)
    assert not smoke.owns_image
    assert smoke.env["ADMIN_API_KEY"] != smoke.env["OPERATOR_API_KEY"]
    arguments = smoke.volume_args() + smoke.storage_environment(True)
    assert not any(str(ROOT) in argument for argument in arguments)
    assert smoke.env["ADMIN_API_KEY"] not in " ".join(arguments)
    assert smoke.env["OPERATOR_API_KEY"] not in " ".join(arguments)
    assert "DATABASE_URL=sqlite:////data/db/smoke.db" in arguments
    assert "TRUST_SKLEARN_ARTIFACTS=true" in arguments


def test_smoke_cleanup_never_deletes_a_supplied_image_or_unknown_volume(monkeypatch):
    smoke = DockerSmoke("existing-image")
    smoke.containers = ["owned-container"]
    smoke.created_volumes = ["owned-volume"]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return Mock(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    smoke.cleanup()
    assert calls == [
        ["docker", "rm", "--force", "owned-container"],
        ["docker", "volume", "rm", "owned-volume"],
    ]


def test_fixture_refuses_development_database_before_opening_any_resource(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./model_serving.db")
    monkeypatch.setenv("MSP_CONTAINER_SMOKE", "1")
    with pytest.raises(ValueError, match="isolated"):
        require_isolated_storage()


def test_layer_checker_detects_infrastructure_leak(tmp_path):
    folder = tmp_path / "app" / "domain"
    folder.mkdir(parents=True)
    (folder / "bad.py").write_text("from sqlalchemy import create_engine\n")
    assert len(layer_errors(tmp_path)) == 1


def test_management_transport_never_follows_redirects():
    assert (
        NoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "https://untrusted.example"
        )
        is None
    )


def test_management_cli_refuses_missing_credentials_without_network(
    monkeypatch, capsys
):
    from scripts.ops_request import main

    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["ops_request.py", "/metrics"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "No operational API key" in capsys.readouterr().err


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com",
        "//example.com",
        "/metrics?secret=x",
        "/metrics#fragment",
        "/a\\b",
        "/a\nb",
    ],
)
def test_management_cli_rejects_unsupported_destinations(monkeypatch, path):
    from scripts.ops_request import main

    monkeypatch.setattr(sys, "argv", ["ops_request.py", path])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_wheel_lock_hashes_distribution_metadata(tmp_path):
    import zipfile

    path = tmp_path / "sample.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "demo.dist-info/METADATA", "Name: Demo_Package\nVersion: 1.2\n"
        )
    assert wheel_entry(path) == (
        "demo-package",
        "1.2",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize("status", ["0", "143"])
def test_graceful_container_stop_accepts_sigterm_only_after_completed_cleanup(
    monkeypatch, status
):
    smoke = DockerSmoke("provided")
    smoke.containers = ["owned"]
    commands = []

    def docker(*args, **kwargs):
        commands.append(args)
        return status if args[0] == "inspect" else ""

    monkeypatch.setattr(smoke, "docker", docker)
    monkeypatch.setattr(
        smoke, "container_logs", lambda name: "Application shutdown complete."
    )
    smoke.stop("owned")
    assert smoke.containers == [] and commands[-1] == ("rm", "owned")


@pytest.mark.parametrize(
    "status,logs",
    [("137", "Application shutdown complete."), ("143", "shutdown never completed")],
)
def test_forced_or_incomplete_container_shutdown_is_not_a_pass(
    monkeypatch, status, logs
):
    smoke = DockerSmoke("provided")
    smoke.containers = ["owned"]
    monkeypatch.setattr(smoke, "docker", lambda *args, **kwargs: status)
    monkeypatch.setattr(smoke, "container_logs", lambda name: logs)
    with pytest.raises(AssertionError):
        smoke.stop("owned")


def test_power_shell_stdin_json_and_explicit_operator_key(monkeypatch, capsys):
    import io
    from scripts import ops_request

    admin, operator = "a" * 32, "b" * 32
    monkeypatch.setenv("ADMIN_API_KEY", admin)
    monkeypatch.setenv("OPERATOR_API_KEY", operator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ops_request.py",
            "/routing/demo-linear",
            "--method",
            "PUT",
            "--role",
            "operator",
            "--data-stdin",
        ],
    )
    monkeypatch.setattr(
        sys, "stdin", io.StringIO('{"weights":{"v1":100},"expected_revision":0}')
    )
    response = Mock()
    response.status = 200
    response.read.return_value = b'{"ok":true}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    def opened(request, timeout):
        assert request.get_header("X-api-key") == operator
        assert json.loads(request.data)["weights"] == {"v1": 100}
        return response

    monkeypatch.setattr(ops_request, "build_opener", lambda *args: Mock(open=opened))
    ops_request.main()
    output = capsys.readouterr().out
    assert admin not in output and operator not in output and "HTTP 200" in output
