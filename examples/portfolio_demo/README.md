# Official portfolio demo: ONNX, routing, monitoring and recovery

Two bundled, deterministic **147-byte ONNX graphs** implement `y = 2x + 1`
(v1) and `y = 3x + 1` (v2). Input `[[4.0]]` returns `{"Y":[[9.0]]}`
or `{"Y":[[13.0]]}`. Normal demo execution uses only Python's standard library
as an HTTP client and the existing server. No training, Pickle, `onnx` package,
artifact generation, or network downloads are needed to run it. Building the
base image/dependencies initially requires the usual network access.

## Docker / Windows PowerShell 5.1+

From the repository root, use a **separate Compose project** and generated keys.
Keep this shell open; do not print credentials or publish resolved Compose config.

```powershell
function New-DemoKey {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes); [Convert]::ToBase64String($bytes) }
    finally { $rng.Dispose() }
}
$env:ADMIN_API_KEY = New-DemoKey
$env:OPERATOR_API_KEY = New-DemoKey
$env:TRUST_SKLEARN_ARTIFACTS = 'false'
$env:APP_PORT = '8001' # Avoid an existing development server on 8000.
# Build once after a fresh clone, or reuse a current image containing this demo.
docker compose -p portfolio-onnx up --build --detach --wait
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo setup
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo run
```

`setup` registers `demo-linear-onnx` v1/v2 and ready deployments
`portfolio-onnx-v1`/`portfolio-onnx-v2`, then initializes 100/0 routing.
Matching identities, schemas (including bundled artifact SHA-256), deployment
resources and RUNNING states are reused. Artifacts are never overwritten.
Mismatched identities, stopped/failed deployments or non-demo routing allocations
cause a clear failure; there is no silent repair. Existing artifact bytes cannot
be downloaded through management APIs; setup compares their recorded fingerprint
and metadata, while the run verifies predictions. Use these reserved names only
for this demo. The isolated project must not override `CONTAINER_DATABASE_URL`
or `CONTAINER_ARTIFACT_STORAGE_DIR` to point at other data.

`run` repeats safe setup, verifies explicit versions, sets 90/10 routing, sends
12 versionless requests, sets 50/50 and sends another 12. It prints destinations,
observed counts, revision history, metrics and five recent logs. It finally
**rolls back to a 100/0 historical revision**, appending a new revision, and checks
the final configuration and prediction. Re-running `run` deliberately appends demo
routing history and inference logs. It is not a load test. Small random samples
need not match configured ratios or even include both versions; explicit calls
verify both graphs independently. Do not run concurrent operators against this
demo: revision conflicts fail rather than retrying mutations.

```powershell
# Public explicit prediction through the host port:
Invoke-RestMethod http://127.0.0.1:8001/predict/demo-linear-onnx -Method Post -ContentType application/json -Body '{"version":"v1","inputs":[[4.0]]}'
# Protected reads use real container-local loopback, never the host port:
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /metrics/demo-linear-onnx --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /inference-logs --role operator
docker compose -p portfolio-onnx exec -T api python -m scripts.ops_request /routing/demo-linear-onnx --role operator
# Recreate the API, keeping volumes; startup automatically revalidates deployments:
docker compose -p portfolio-onnx up --detach --force-recreate --wait api
docker compose -p portfolio-onnx exec -T api python -m examples.portfolio_demo.demo verify
docker compose -p portfolio-onnx down # Keeps the demo's volumes; no data deletion.
```

`verify` makes predictions and reads final routing/monitoring without registering
or replacing resources. Default metrics/log views cover the last five minutes;
historical observations remain persisted beyond that window. API keys and roles,
loopback restrictions, single-process recovery and existing request limits are
unchanged. Uploaded bytes are always ONNX; management never accepts Pickle.

## Optional trusted sklearn demo (separate project)

Reuse the existing locally generated sklearn demo, with explicit trust opt-in
only for this separate project. No Pickle upload endpoint is involved.

```powershell
$env:APP_PORT = '8002'
$env:TRUST_SKLEARN_ARTIFACTS = 'true'
# Stop the API while its existing offline seeder owns the database:
docker compose -p portfolio-sklearn run --rm --no-deps api python -m examples.seed_development_model
docker compose -p portfolio-sklearn up --detach --wait api
Invoke-RestMethod http://127.0.0.1:8002/predict/demo-linear -Method Post -ContentType application/json -Body '{"version":"v1","inputs":[[4.0]]}'
# Expected predictions: [9.0]
docker compose -p portfolio-sklearn down
$env:TRUST_SKLEARN_ARTIFACTS = 'false'
```

The existing seeder creates/trusts its own local model; never substitute untrusted
`.pkl` files. If repeating this optional setup, stop that project's API first.
