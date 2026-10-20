# Exact CPython/Linux base manifest; this packaging target is linux/amd64.
ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY requirements/build.lock requirements/runtime.lock /locks/
RUN python -m venv /opt/build \
    && /opt/build/bin/python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /locks/build.lock
COPY pyproject.toml README.md ./
COPY app ./app
RUN /opt/build/bin/python -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheel . \
    && python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /locks/runtime.lock \
    && /opt/venv/bin/python -m pip install --no-index --no-deps /wheel/*.whl \
    && /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -c "import numpy, sklearn, onnxruntime, sqlalchemy; from sklearn.linear_model import LinearRegression; assert abs(LinearRegression().fit([[1.],[2.],[3.]],[3.,5.,7.]).predict([[4.]])[0]-9.) < 1e-8; assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()"

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/db/model_serving.db \
    ARTIFACT_STORAGE_DIR=/data/artifacts \
    TRUST_SKLEARN_ARTIFACTS=false \
    MSP_SECURE_CONTAINER=true \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/db /data/artifacts /app \
    && chown app:app /data/db /data/artifacts \
    && chmod 0700 /data/db /data/artifacts
COPY --from=builder /opt/venv /opt/venv
COPY examples /app/examples
COPY scripts /app/scripts
WORKDIR /app
USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=6 \
    CMD ["python", "-c", "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); assert r.status == 200"]
CMD ["python", "-m", "uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", "--timeout-graceful-shutdown", "50"]
