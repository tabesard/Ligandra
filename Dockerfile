# DrugForge — CPU image with the core (classical + graph_ga) path.
# Heavy/GPU extras (torch, transformers) are opt-in; add them in a derived image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# RDKit wheels need a couple of shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ligandra ./ligandra
COPY examples ./examples

RUN pip install --upgrade pip && pip install -e ".[ui,chembl]"

EXPOSE 8501
# Default: launch the Streamlit UI. Override CMD to run the CLI, e.g.:
#   docker run --rm ligandra ligandra run examples/local_experiment.yaml
CMD ["streamlit", "run", "ligandra/ui/app.py", "--server.address=0.0.0.0"]
