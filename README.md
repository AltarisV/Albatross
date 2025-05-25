A Proof of Concept for extracting IT-Grundschutz requirements from the **XML** edition of the Kompendium with `ingest.py`, loading them into ChromaDB (or dumping to JSON), and then querying/navigation via a Streamlit app in `app.py`.

## Configuration

1. Create a `.env` file with your [OpenAI API Key](https://openai.com/blog/openai-api) in the project root (only really needed it you want to store in the vector db):

   ```env
   OPENAI_API_KEY='your_openai_api_key_here'
   ```

2. Place your XML file in **`resources/grundschutz_2023.xml`** (or adjust the path in the commands below).

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

After running, you’ll see something like:

```
📄 Extracted 2187 requirements from 'resources/grundschutz_2023.xml'.
✅ Wrote 2187 requirements to JSON file 'requirements.json'
```

(or…)

```
📄 Extracted 2187 requirements from 'resources/grundschutz_2023.xml'.
✅ Ingested 2187 requirements into 'db' using OpenAI Embeddings
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```
