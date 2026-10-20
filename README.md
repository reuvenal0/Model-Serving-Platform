# Model Serving Platform

## Overview

A Python model-serving platform with versioned registration, deployment lifecycle,
ONNX/sklearn inference, weighted routing and monitoring. It uses FastAPI, SQLite
and local artifacts in a layered modular monolith, designed as a portfolio MVP.

## Features & Supported Formats

| Area | Implemented | Limitations |
| --- | --- | --- |
| Registry | Model metadata, versions, persistent artifacts | ONNX base64 JSON upload; no multipart upload |
| Execution | ONNX dense tensors; trusted sklearn Pickle offline | No Custom Python or model sandbox |
| Deployment | Readiness, stop, restart recovery | Replicas/resources are logical, not isolated workers |
| Inference | Explicit or routed version; vectorized batches | No asynchronous jobs or hard execution timeout |
| Routing | Weighted canary, revision history, manual rollback | No automatic rollout or error-triggered rollback |
| Monitoring | Persisted logs, rates, latency percentiles, RSS/cache | Best-effort logs; no retention service |
| Security | Operator/admin keys and genuine loopback management | Public inference; not production-ready |

## Quick Start (Docker)

Recommended: Git, Docker Desktop using Linux containers (amd64), PowerShell 5.1+.
Run in one shell; Docker/CI use pinned Python 3.12 dependencies.

```powershell
git clone https://github.com/reuvenal0/Model-Serving-Platform.git
Set-Location Model-Serving-Platform
Copy-Item .env.example .env
function New-DemoKey {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes); [Convert]::ToBase64String($bytes) }
    finally { $rng.Dispose() }
}
$env:ADMIN_API_KEY = New-DemoKey
$env:OPERATOR_API_KEY = New-DemoKey
$env:TRUST_SKLEARN_ARTIFACTS = 'false'
$env:APP_PORT = '8001'
docker compose -p portfolio-onnx up --build --detach --wait
Invoke-RestMethod http://127.0.0.1:8001/health
```

Startup creates an empty schema; it does not install models. Keep the generated
keys private and this shell open. Process variables override Compose `.env` values;
never publish resolved credentials. Leave container storage path overrides unset
so this project uses its own volumes. First build requires network access.

## Deploy & Run Demo

```powershell
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo setup
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo run
# Recreate the API with the same volumes; startup revalidates deployments.
docker compose -p portfolio-onnx up --detach --force-recreate --wait api
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo verify
# Stop containers, retaining data:
docker compose -p portfolio-onnx down
```

Bundled [v1](examples/portfolio_demo/v1.onnx) computes `2x+1`; [v2](examples/portfolio_demo/v2.onnx)
computes `3x+1`. At `x=4`, predictions are 9 and 13. The demo registers/deploys both,
samples 90/10 then 50/50 routing, shows metrics, and rolls back to 100% v1.
Observed random counts need not match the weights. Setup reuses matching resources
and rejects incompatible records; rerunning the demonstration appends logs/history.
No Pickle trust, training or artifact generation is needed.

## API Examples

After setup, public inference is available through the host port:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/predict/demo-linear-onnx -Method Post -ContentType application/json -Body '{"version":"v1","inputs":[[4.0]]}'
Invoke-RestMethod http://127.0.0.1:8001/predict/demo-linear-onnx/batch -Method Post -ContentType application/json -Body '{"version":"v1","inputs":[[4.0],[5.0]]}'
```

Single response (latency is illustrative and varies):

```json
{"model_name":"demo-linear-onnx","version":"v1","predictions":{"Y":[[9.0]]},"latency_ms":1.2}
```

The batch response has the same fields with `"predictions":{"Y":[[9.0],[11.0]]}`.
Omit `version` to use routing; the response identifies the version actually used.
Batch means one vectorized tensor call, not an asynchronous job.

Management requires `X-API-Key` **and a genuine loopback peer**. Host-published
Docker requests are not container-local, even with valid keys. Use authenticated
commands inside the container; Swagger never bypasses this boundary:

```powershell
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /models --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /metrics/demo-linear-onnx --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /inference-logs --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /routing/demo-linear-onnx --role operator
```

Operators read; administrators also register models/versions, deploy/stop and
change/roll back routing. `setup`/`run` use the administrator key without printing it.
Version registration is `POST /models/{name}/versions` with `version`,
`onnx_base64` and optional schema objects; artifact paths are never accepted.
Operators/admins can download bytes with `GET /models/{name}/versions/{version}/artifact`.
Downloads obey `MAX_RUNTIME_ARTIFACT_BYTES`; missing artifacts return 404.
Admins can `DELETE /models/{name}/versions/{version}` (204); deployment or routing
history references block deletion (409). `POST /deployments/{deployment_id}/scale`
accepts `{"replicas":3}` and returns updated metadata: replicas remain logical,
with no worker creation. Multipart upload is not supported.

## Run Tests

Local Python 3.12–3.14 (Docker is the reproducible environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

Last full-suite baseline: **991 passed**. CI configuration is in [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Documentation & Limitations

- [Design decisions](DECISIONS.md) and [AI usage / developer TODOs](AI_USAGE.md).
- [Assignment compliance and API gaps](docs/ASSIGNMENT_COMPLIANCE.md).
- [Complete ONNX walkthrough and optional trusted sklearn demo](examples/portfolio_demo/README.md).
- [Release audit, evidence and limitations](docs/RELEASE_AUDIT.md).

One API process owns SQLite and in-process model runtimes. There are no isolated
workers or hard per-model CPU/RAM limits. Request/admission/artifact limits provide
partial protection; native work cannot be forcibly cancelled. Historical SQLite
schema upgrades are unsupported; incompatible databases fail rather than migrate.
Do not expose this local-development management interface publicly.
