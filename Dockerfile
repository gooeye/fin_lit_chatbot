FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/

RUN pip install --upgrade pip && python - <<'PY'
from pathlib import Path
import subprocess
import tomllib

pyproject = tomllib.loads(Path("/app/pyproject.toml").read_text(encoding="utf-8"))
dependencies = pyproject.get("project", {}).get("dependencies", [])
if dependencies:
    subprocess.check_call(["pip", "install", *dependencies])
PY

COPY app.py /app/
COPY fin_lit_chatbot /app/fin_lit_chatbot

RUN pip install --no-deps .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
