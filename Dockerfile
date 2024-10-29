FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV PIP_INDEX_URL=$PIP_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY run_api.py .

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/python", "run_api.py"]
CMD ["--help"]
