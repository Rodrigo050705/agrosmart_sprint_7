FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os requisitos e força a instalação no Python global
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir streamlit && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]