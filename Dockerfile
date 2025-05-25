# Stage 1: Dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# Kopiere requirements und installiere
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: App-Container
FROM base

WORKDIR /app

# Copy code + resources
COPY .env .
COPY ingest.py app.py requirements.txt ./
COPY resources/ resources/

# DB-Ordner (wird bei Ingest gefüllt)
VOLUME ["/app/db"]

EXPOSE 8501

# Entry point: bei leerem db ingest, dann start streamlit
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["if [ ! -d db ] || [ -z \"$(ls -A db)\" ]; then python ingest.py resources/grundschutz_2023.xml --mode vectordb --output db; fi && streamlit run app.py --server.port=8501 --server.headless=true"]
