A Proof of Concept for extracting data from a PDF with `ingest.py`, put it into a ChromaDB and then query and navigate the data through a streamlit app in `app.py`.


## Configuration
1. Create a `.env` file with your [OpenAI API Key](https://openai.com/blog/openai-api) in the project root and add the following line:

```env
OPENAI_API_KEY='your_openai_api_key_here'
```
2. Place your PDF file in the project root

## Usage

### 1. Ingest PDF into ChromaDB

    python ingest.py IT-Grundschutz-2023.pdf --persist_dir db

- **IT-Grundschutz-2023.pdf**: Path to your PDF file  
- **--persist_dir db**: Directory where ChromaDB files are stored (default: `db/`)

After ingest completes, you should see in your terminal:

    ✅ Ingested X document chunks into 'db' using OpenAI Embeddings

### 2. Launch the Streamlit App

    streamlit run app.py
