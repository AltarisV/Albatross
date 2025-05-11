import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from st_aggrid import AgGrid, GridOptionsBuilder

load_dotenv()


def load_db_entries(embeddings):
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
    collection = vectordb._collection
    total = collection.count()
    data = collection.get(limit=total)
    rows = []
    for doc_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        rows.append({
            "ID": doc_id,
            "Seite": meta.get("page_number"),
            "Typ": meta.get("block_type") or "Unknown",
            "Snippet": doc_text.replace("\n", " ")
        })
    df = pd.DataFrame(rows)
    return df, data["documents"]


def main():
    st.set_page_config(page_title="PDF Explorer POC", layout="wide")
    st.title("ChromaDB Kompendium Browser")

    page = st.sidebar.radio("Navigation", ["Query PDF", "Datenbank Explorer"])

    api_key = os.getenv("OPENAI_API_KEY")
    embeds = OpenAIEmbeddings(openai_api_key=api_key)

    if page == "Datenbank Explorer":
        st.header("🔎 Datenbank Explorer")
        df, docs = load_db_entries(embeds)

        with st.expander("Filter & Sortierung", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                pages = sorted(df["Seite"].unique())
                selected_pages = st.multiselect("Filter Seiten", options=pages, default=pages)
            with cols[1]:
                types = sorted(df["Typ"].unique())
                selected_types = st.multiselect("Filter Typen", options=types, default=types)
            with cols[2]:
                sort_col = st.selectbox("Sortiere nach", options=["Seite", "Typ", "ID"], index=0)
                ascending = st.checkbox("Aufsteigend", value=True)
        df_filtered = df[df["Seite"].isin(selected_pages) & df["Typ"].isin(selected_types)]
        df_sorted = df_filtered.sort_values(by=sort_col, ascending=ascending)

        gb = GridOptionsBuilder.from_dataframe(df_sorted)
        gb.configure_default_column(editable=False, sortable=True, filter=True)
        gb.configure_column("Snippet", wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gridOptions = gb.build()

        st.subheader("📊 Datenbank-Tabelle")
        grid_response = AgGrid(
            df_sorted,
            gridOptions=gridOptions,
            enable_enterprise_modules=False,
            height=600,
            fit_columns_on_grid_load=True
        )

        rows = grid_response.get('selected_rows', [])
        if rows:
            sel = rows[0]
            st.subheader("Detailansicht")
            st.markdown(f"**ID:** {sel['ID']}")
            st.markdown(f"- Seite: {sel['Seite']}  |  Typ: {sel['Typ']}")
            idx = df_sorted.index[df_sorted['ID'] == sel['ID']][0]
            st.write(docs[idx])

    else:
        st.header("Finde Informationen aus der PDF")
        query = st.text_input("Deine Frage:")
        if query:
            vectordb = Chroma(persist_directory="db", embedding_function=embeds)
            results = vectordb.similarity_search(query, k=5)
            st.subheader("🔍 Gefundene Dokument-Chunks")
            for i, doc in enumerate(results):
                page_num = doc.metadata.get("page_number", "–")
                block = doc.metadata.get("block_type", "–")
                st.markdown(f"**Chunk {i + 1}** • Seite {page_num} • Typ {block}")
                st.write(doc.page_content)
                st.write("---")


if __name__ == "__main__":
    main()
