ARG REELTONE_DEPENDENCY_IMAGE=python:3.11-slim
FROM ${REELTONE_DEPENDENCY_IMAGE} AS dependencies
USER root
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    REELTONE_DATA=/var/lib/reeltone
WORKDIR /opt/reeltone
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY reeltone reeltone
COPY static static
RUN useradd -U -u 11505 reeltone && install -d -o reeltone -g reeltone /var/lib/reeltone
USER reeltone
EXPOSE 8000
CMD ["python", "-m", "reeltone"]
