import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from st_aggrid import AgGrid, GridOptionsBuilder

load_dotenv()


def load_db_entries(embeddings):
    # open the existing Chroma DB
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
    col = vectordb._collection
    total = col.count()
    data = col.get(limit=total)

    rows = []
    for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        rows.append({
            "ID": doc_id,
            "Req-ID": meta.get("requirement_id"),
            "XML-ID": meta.get("xml_id"),
            # show the full snippet without character limit
            "Snippet": text.replace("\n", " ")
        })
    df = pd.DataFrame(rows)
    return df, data["documents"], data["metadatas"]


def main():
    st.set_page_config(page_title="Grundschutz Kompendium Explorer", layout="wide")
    st.title("IT-Grundschutz Kompendium-Browser")

    page = st.sidebar.radio("Navigation", ["Datenbank Explorer", "Frage & Antworten"])

    api_key = os.getenv("OPENAI_API_KEY")
    embeds = OpenAIEmbeddings(openai_api_key=api_key)

    if page == "Datenbank Explorer":
        st.header("Datenbank Explorer")
        df, docs, _ = load_db_entries(embeds)

        # show how many entries we loaded
        st.markdown(f"**In Datenbank:** {len(df)} Einträge")

        # show full table
        with st.expander("Alle Einträge anzeigen", expanded=True):
            st.dataframe(df)

        # filter by requirement ID only
        with st.expander("Filter & Sortierung", expanded=False):
            reqs = sorted(x for x in df["Req-ID"].unique() if x)
            selected_reqs = st.multiselect("Filter Req-ID", options=reqs, default=reqs)
            sort_col = st.selectbox("Sortiere nach", options=["Req-ID", "ID"], index=0)
            ascending = st.checkbox("Aufsteigend", value=True)

        # apply filtering and sorting
        df_filtered = df[df["Req-ID"].isin(selected_reqs)]
        df_sorted = df_filtered.sort_values(by=sort_col, ascending=ascending)

        # AG-Grid view
        st.subheader("Gefiltert")
        gb = GridOptionsBuilder.from_dataframe(df_sorted)
        gb.configure_default_column(editable=False, sortable=True, filter=True)
        gb.configure_column("Snippet", wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gridOptions = gb.build()
        grid_resp = AgGrid(
            df_sorted,
            gridOptions=gridOptions,
            enable_enterprise_modules=False,
            height=600,
            fit_columns_on_grid_load=True
        )

        sel = grid_resp.get("selected_rows")
        if sel:
            row = sel[0]
            st.subheader(f"Detail: {row['Req-ID']}  (XML-ID: {row['XML-ID']})")
            idx = df_sorted.index[df_sorted["ID"] == row["ID"]][0]
            st.write(docs[idx])

    else:
        st.header("Vector-DB-Query")
        query = st.text_input("Stelle deine Frage:")
        if query:
            vectordb = Chroma(persist_directory="db", embedding_function=embeds)
            results = vectordb.similarity_search(query, k=5)
            for i, doc in enumerate(results, 1):
                rid = doc.metadata.get("requirement_id", "–")
                st.markdown(f"**Ergebnis {i}** • Req-ID: {rid}")
                st.write(doc.page_content)
                st.markdown("---")


if __name__ == "__main__":
    main()
