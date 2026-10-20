# Model Serving Platform

**Reuven**

A Python model-serving platform for registering and deploying versioned ML models, serving predictions, splitting traffic between versions, and monitoring inference. Built with FastAPI, SQLAlchemy, SQLite, ONNX Runtime and scikit-learn as a layered modular monolith for local use.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start (Docker)](#quick-start-docker)
- [Supported Model Formats](#supported-model-formats)
- [Deploy a Model (Step by Step)](#deploy-a-model-step-by-step)
- [Example Inference Calls](#example-inference-calls)
- [Management API](#management-api)
- [Run Tests](#run-tests)
- [Troubleshooting](#troubleshooting)
- [Known Limitations & Documentation](#known-limitations--documentation)

## Features

| Area | Implemented |
| --- | --- |
| Model registry | Model metadata, versioning, persistent artifacts, ONNX upload and artifact download |
| Deployments | Readiness checks, stop, logical scaling and startup recovery |
| Inference | Explicit-version or weighted-routed predictions, vectorized batch requests |
| A/B testing | Weighted routing, canary changes, revision history and manual rollback |
| Monitoring | Inference logs, throughput, error rates and latency percentiles |
| Engineering | Layered interfaces, SQLite, Docker Compose, pytest and GitHub Actions |

## Project Structure

```text
app/
  api/                 # HTTP routes, request schemas and authorization
  core/                # Application wiring and shared errors
  domain/              # Entities and business invariants
  interfaces/          # Persistence, storage and loading contracts
  services/            # Registry, deployment, inference, routing, monitoring
  infrastructure/      # SQLite, file storage, sklearn and ONNX implementations
examples/
  portfolio_demo/      # demo.py, v1.onnx, v2.onnx, extended walkthrough
  seed_development_model.py
scripts/                # Operational and verification utilities
tests/                  # Unit and integration tests
requirements/           # Pinned dependency files
.github/workflows/ci.yml
Dockerfile
compose.yaml
DECISIONS.md
AI_USAGE.md
```

## Quick Start (Docker)

**Prerequisites:** Git, Docker with Compose v2 (Linux containers, amd64) and Python 3. On Linux/macOS, use `python3` in place of `python` if needed. The following commands use Python rather than platform-specific shell syntax, so they work in PowerShell, Bash and Zsh.

```sh
git clone https://github.com/reuvenal0/Model-Serving-Platform.git
cd Model-Serving-Platform
```

Create a private `.env` with cryptographically random API keys. This command **refuses to overwrite** an existing file (keep your existing `.env` if you have already initialized the project):

```sh
python -c "import os,secrets; f=os.fdopen(os.open('.env',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w',encoding='utf-8'); f.write('ADMIN_API_KEY='+secrets.token_urlsafe(32)+'\nOPERATOR_API_KEY='+secrets.token_urlsafe(32)+'\nAPP_PORT=8001\nTRUST_SKLEARN_ARTIFACTS=false\n'); f.close()"
docker compose -p portfolio-onnx up --build --detach --wait
python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8001/health')),indent=2))"
```

Expected health response:

```json
{"status": "ok"}
```

Docker Compose reads `.env`; inherited environment variables override it. If your shell already has conflicting API key, port, or storage settings, clear those overrides before starting. Keep `.env` private and out of Git. The first Docker build requires internet access. Startup creates an empty database; **models are installed by the setup command below**, not automatically at startup. Data is stored in project-scoped Docker volumes.

**Security:** Inference and `/health` are public on the configured host-loopback port. Management endpoints require an API key **and a genuine loopback connection inside the container**. Host-published Docker management requests are rejected even with a valid key; use the provided `docker compose exec` commands.

## Supported Model Formats

| Format | Support | Notes |
| --- | --- | --- |
| ONNX (`.onnx`) | HTTP registration and inference | Bundled sample models; supported dense tensor types |
| scikit-learn Pickle (`.pkl`) | Trusted offline registration and inference | Requires explicit trust opt-in; never unpickle untrusted artifacts |
| Custom Python (`.py`) | Not implemented | Arbitrary uploaded Python code is not executed |

ONNX upload uses bounded Base64 JSON (`onnx_base64`), not multipart form upload. Neither ONNX nor Pickle runs in an isolated model process.

## Deploy a Model (Step by Step)

The bundled demo contains two ONNX versions of `demo-linear-onnx`:

| Version | Function | Prediction for `x = 4` |
| --- | --- | --- |
| [v1](examples/portfolio_demo/v1.onnx) | `y = 2x + 1` | `9.0` |
| [v2](examples/portfolio_demo/v2.onnx) | `y = 3x + 1` | `13.0` |

**1. Register, deploy and check readiness.** The `setup` command uses authenticated management APIs, registers both bundled versions, creates deployments and initializes 100/0 routing. Matching existing resources are reused.

```sh
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo setup
```

**2. Check each deployment's readiness** (administrator access):

```sh
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /deployments/portfolio-onnx-v1/health --role admin
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /deployments/portfolio-onnx-v2/health --role admin
```

**3. Run the inference examples below**, then demonstrate weighted routing, metrics and rollback:

```sh
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo run
```

The demo switches to 90/10, then 50/50, reports observed requests and metrics, and manually rolls back to 100% v1. Routing is probabilistic, so small observed samples will not necessarily match weights exactly. Repeated `run` executions append monitoring records and routing history.

**4. Verify restart recovery** without removing persisted data:

```sh
docker compose -p portfolio-onnx up --detach --force-recreate --wait api
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo verify
```

When finished, `docker compose -p portfolio-onnx down` stops the stack **without removing its volumes**.

## Example Inference Calls

Run after `setup`, with the API still running. This Python standard-library snippet makes **three real HTTP requests** and prints the complete JSON, including nested prediction values. It works in PowerShell, Bash and Zsh, unlike PowerShell's default table display which hides arrays.

```sh
python -c "import json,urllib.request as http
base='http://127.0.0.1:8001/predict/demo-linear-onnx'
for suffix,version,inputs in [('', 'v1', [[4.0]]), ('', 'v2', [[4.0]]), ('/batch', 'v1', [[4.0],[5.0]])]:
    body=json.dumps({'version':version,'inputs':inputs}).encode()
    request=http.Request(base+suffix,data=body,headers={'Content-Type':'application/json'})
    with http.urlopen(request,timeout=30) as response:
        print(json.dumps(json.load(response),indent=2))"
```

Expected outputs, in order (`latency_ms` is illustrative and will vary):

**Single prediction — v1**

```json
{"model_name":"demo-linear-onnx","version":"v1","predictions":{"Y":[[9.0]]},"latency_ms":1.2}
```

**Single prediction — v2**

```json
{"model_name":"demo-linear-onnx","version":"v2","predictions":{"Y":[[13.0]]},"latency_ms":1.3}
```

**Batch prediction — v1**

```json
{"model_name":"demo-linear-onnx","version":"v1","predictions":{"Y":[[9.0],[11.0]]},"latency_ms":1.4}
```

Omit `version` in a prediction request to use the current weighted routing policy. The response identifies which version served the request. Batch prediction is one vectorized call, not an asynchronous job.

## Management API

Operations use container-local HTTP, an `X-API-Key`, and the appropriate role. Example read-only calls:

```sh
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /models --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /metrics/demo-linear-onnx --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /inference-logs --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /routing/demo-linear-onnx --role operator
```

| Method | Endpoint | Purpose / access |
| --- | --- | --- |
| `GET` | `/health` | Public liveness |
| `POST` | `/predict/{model_name}`; `/predict/{model_name}/batch` | Public inference |
| `GET` / `POST` | `/models` | Read / admin: list / register |
| `GET` / `POST` | `/models/{name}/versions` | Read / admin: list / register ONNX version |
| `GET` | `/models/{name}/versions/{version}/artifact` | Read: download artifact |
| `DELETE` | `/models/{name}/versions/{version}` | Admin: delete unreferenced version |
| `GET` / `POST` | `/deployments` | Read / admin: list / deploy |
| `POST` | `/deployments/{deployment_id}/stop`; `/deployments/{deployment_id}/scale` | Admin: stop / update logical replicas |
| `GET` | `/deployments/{deployment_id}/health` | Admin: load readiness |
| `GET` / `PUT` | `/routing/{model_name}` | Read / admin: inspect / update weights |
| `GET` | `/routing/{model_name}/history` | Read: revision history |
| `POST` | `/routing/{model_name}/rollback` | Admin: restore historical weights |
| `GET` | `/metrics`; `/metrics/{model_name}`; `/inference-logs` | Read: monitoring |

Version registration accepts `version`, `onnx_base64` and optional schema fields; scaling accepts `{"replicas":3}`. Routing changes use revision checks. Logical replicas do **not** create additional worker processes.

## Run Tests

Use Python 3.12–3.14 locally. Create a virtual environment and install development dependencies:

```sh
python -m venv .venv
```

Activate it using `.\.venv\Scripts\Activate.ps1` (PowerShell), or `source .venv/bin/activate` (Linux/macOS), and then run:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

 Tests cover registration, deployments, correctness, routing, monitoring, error handling, authorization and concurrency. CI lives in [.github/workflows/ci.yml](.github/workflows/ci.yml). The previous documented baseline was **995 tests passed**; rerun pytest for the count at your revision.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| API is healthy but prediction returns 404 | Run `demo setup` first; startup does not register models |
| Management request is rejected from host | Expected security behavior; use `docker compose exec` |
| A new shell cannot access management APIs | Check credentials in the active Compose container; do not generate new keys for an existing installation |
| Host port 8001 is occupied | Choose another `APP_PORT` before startup and update example URLs |
| SQLite reports an incompatible historical schema | Use a *new* Compose project / volume or perform an intentional migration; **do not delete existing volumes to silence errors** |

## Known Limitations & Documentation

Models share one API process. There is no per-model crash isolation, enforced per-model CPU/RAM budget, native-call cancellation, async job API, Custom Python loader, multipart upload, autoscaling or automatic error-triggered rollback. Logs are best-effort; historical SQLite schema migrations are unsupported. This is a local portfolio MVP, **not a production-ready public service**.

- [Design decisions](DECISIONS.md)
- [AI tool usage](AI_USAGE.md)
- [Extended ONNX and trusted sklearn walkthrough](examples/portfolio_demo/README.md)
