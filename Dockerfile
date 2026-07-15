FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY causal_toolkit ./causal_toolkit
RUN pip install --no-cache-dir -e ".[api,dashboard]"

COPY dashboard ./dashboard
COPY tests ./tests

EXPOSE 8000 8501

# Default: run the API. Override CMD to run the dashboard instead:
#   docker run -p 8501:8501 causal-toolkit streamlit run dashboard/app.py --server.address 0.0.0.0
CMD ["uvicorn", "causal_toolkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
