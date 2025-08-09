A Proof of Concept for extracting IT-Grundschutz requirements from the **XML** edition of the Kompendium with `ingest.py`, loading them into ChromaDB (or dumping to JSON), and then querying/navigation via a Streamlit app in `app.py`.

## Usage

### 1. Ingest Requirements from XML

```bash
python ingest.py resources/grundschutz_2023.xml \
  --mode [vectordb|json] \
  --output [DB_DIR|output.json]
```

- `resources/grundschutz_2023.xml`: Path to the Kompendium XML
- `--mode vectordb`: Parse XML → embeddings → store in ChromaDB
- `--mode json`: Parse XML → dump each requirement to JSON
- `--output db`: directory for ChromaDB (default)
- `--output filename.json`: filename for JSON dump

**Examples:**

```bash
# JSON dump of all requirements
python ingest.py resources/grundschutz_2023.xml \
  --mode json \
  --output requirements.json

# Build a ChromaDB of all requirements
python ingest.py resources/grundschutz_2023.xml \
  --mode vectordb \
  --output db
```

After running, you’ll see something like this:

```
📄 Extracted 2187 requirements from 'resources/grundschutz_2023.xml'.
✅ Ingested 2187 requirements into 'db' using OpenAI Embeddings
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

## Alternatively, run with Docker-Compose

If you’ve set up the included `Dockerfile` and `docker-compose.yml`, you can build and start everything in one step:

```bash
docker-compose up --build
```

This will automatically run the ingestion step (if needed) and then launch the Streamlit app at [http://localhost:8501](http://localhost:8501).
