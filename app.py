import os
from dotenv import load_dotenv
import re
import streamlit as st
import pandas as pd
from io import BytesIO
import torch

from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    from st_aggrid import AgGrid, GridOptionsBuilder  # optionales Grid
except ImportError:
    AgGrid = None

load_dotenv()


def _requirement_sort_key(rid: str):
    m = re.search(r"A(\d+)$", rid)
    return int(m.group(1)) if m else rid


def _module_sort_key(mid: str):
    """
    Liefert für IDs wie "APP.2.3" ein Tuple (2,3),
    für alle anderen Fälle ein sehr großes Tuple, so dass sie ans Ende wandern.
    """
    nums = re.findall(r'\d+', mid)
    if len(nums) >= 2:
        # z.B. ["2","3"] → (2,3)
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        # z.B. "APP.12" → (12, 0)
        return int(nums[0]), 0
    else:
        # kein Zahlen­match → ans Ende sortieren
        return float('inf'), float('inf')


@st.cache_data(show_spinner=False)
def load_db(persist_dir: str, _embeddings: Embeddings):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=_embeddings)
    col = vectordb._collection
    data = col.get(limit=col.count())
    return data['documents'], data['metadatas']


def load_db_entries(persist_dir: str, embeddings: Embeddings):
    docs, metas = load_db(persist_dir, embeddings)
    rows = []
    for idx, (text, meta) in enumerate(zip(docs, metas)):
        raw_rollen = meta.get('Rollen', '')
        rollen_str = ", ".join(raw_rollen) if isinstance(raw_rollen, list) else raw_rollen

        art = meta.get('Art', 'Baustein')
        if art == "Baustein":
            title = meta.get("baustein_id", "")
            if meta.get("baustein_titel"):
                title += " – " + meta["baustein_titel"]
        elif art == "Gefahrenlage":
            title = meta.get('Gefährdungslage', '')
        else:  # Anforderung
            title = meta.get("Anforderung", meta.get("Anforderungsnummer", ""))

        gef_h_ids = meta.get("zugeordnete_gefahren", [])
        gef_h_titel = meta.get("zugeordnete_gefahren_titel", [])
        gef_h_str = ", ".join(gef_h_titel) if isinstance(gef_h_titel, list) else gef_h_titel

        rows.append({
            "ID": idx,
            "Art": art,
            "Bausteinkategorie": meta.get("bausteinkategorie_id", ""),
            "Baustein": meta.get("baustein_id", ""),
            "Titel": title,
            "Anforderungsnummer": meta.get("Anforderungsnummer", ""),
            "Gefährdungslage": meta.get("Gefährdungslage", ""),
            "Anforderungskategorie": meta.get("Anforderungskategorie", ""),
            "Rollen": rollen_str,
            "Zugeordnete Gefahren": gef_h_str,
            "Vertraulichkeit": meta.get("Vertraulichkeit", False),
            "Integrität": meta.get("Integrität", False),
            "Verfügbarkeit": meta.get("Verfügbarkeit", False),
            "Snippet": (text.replace("\n", " ")[:300] + "…") if len(text) > 300 else text
        })

    df = pd.DataFrame(rows)
    return df, docs, metas


@st.cache_data(show_spinner=False)
def build_hierarchy(docs, metas):
    hier = {}
    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        kat = meta.get('bausteinkategorie_id', 'UNKNOWN')
        b_id = meta.get('baustein_id', 'UNKNOWN')
        art = meta.get('Art', 'Baustein')
        category = hier.setdefault(kat, {})
        b = category.setdefault(b_id, {'baustein_docs': [], 'anforderungen': {}})
        if art == 'Baustein':
            b['baustein_docs'].append((idx, doc, meta))
        elif art == 'Anforderung':
            rid = meta.get('Anforderungsnummer', 'UNKNOWN')
            req = b['anforderungen'].setdefault(rid, {'chunks': [], 'meta': meta})
            req['chunks'].append((idx, doc))
    return hier


def build_excel(requirements):
    rows = []
    for req in requirements:
        meta = req['meta']
        text = "\n\n".join(chunk for _, chunk in req['chunks'])
        gef_h_ids = meta.get("zugeordnete_gefahren", [])
        gef_h_titel = meta.get("zugeordnete_gefahren_titel", [])
        gef_h_ids_str = ", ".join(gef_h_ids) if isinstance(gef_h_ids, list) else gef_h_ids
        gef_h_titel_str = ", ".join(gef_h_titel) if isinstance(gef_h_titel, list) else gef_h_titel

        rows.append({
            "Bausteinkategorie": meta.get('bausteinkategorie_id'),
            "Baustein": meta.get('baustein_id'),
            "Anforderungsnummer": meta.get('Anforderungsnummer'),
            "Anforderung": meta.get('Anforderung'),
            "Anforderungskategorie": meta.get('Anforderungskategorie'),
            "Rollen": meta.get('Rollen'),
            "Zugeordnete Gefahren (IDs)": gef_h_ids_str,
            "Zugeordnete Gefahren (Titel)": gef_h_titel_str,
            "Vertraulichkeit": meta.get("Vertraulichkeit", False),
            "Integrität": meta.get("Integrität", False),
            "Verfügbarkeit": meta.get("Verfügbarkeit", False),
            "Text": text
        })

    df = pd.DataFrame(rows)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Anforderungen")
        buf.seek(0)
    return buf


def module_matches(info, query):
    _, desc, meta = info['baustein_docs'][0]
    if query in meta.get('baustein_titel', '').lower():
        return True
    if query in desc.lower():
        return True
    for rid, req in sorted(info['anforderungen'].items(), key=lambda x: _requirement_sort_key(x[0])):
        title = req['meta'].get('Anforderung', '').lower()
        if query in title:
            return True
        for _, chunk in req['chunks']:
            if query in chunk.lower():
                return True
    return False


def add_to_cart(meta, chunks):
    entry = {'meta': meta, 'chunks': chunks}
    if entry not in st.session_state.cart:
        st.session_state.cart.append(entry)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource(show_spinner=False)
def get_vectordb(_embeddings):
    return Chroma(
        persist_directory='db',
        embedding_function=_embeddings
    )


def main():
    st.set_page_config(page_title='Kompendium Explorer',
                       layout='wide')

    st.sidebar.markdown("### Status")
    status = st.sidebar.empty()

    if 'model_loaded' not in st.session_state:
        status.info("📦 Lade Embedding-Modell…")
        embeddings = get_embeddings()
        _ = embeddings.embed_query("initialisierung")
        status.success("✅ Modell bereit")
        status.info("📡 Lade Vektor-Datenbank…")
        vectordb = get_vectordb(embeddings)
        status.success("✅ Alles bereit")
        st.session_state['embeddings'] = embeddings
        st.session_state['vectordb'] = vectordb
        st.session_state['model_loaded'] = True
    else:
        embeddings = st.session_state['embeddings']
        vectordb = st.session_state['vectordb']
        status.success("✅ App bereit")

    df, docs, metas = load_db_entries('db', embeddings)

    st.sidebar.header("Ausgewählte Anforderungen")
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    for i, item in enumerate(st.session_state.cart):
        title = item['meta'].get('Anforderung', item['meta']['Anforderungsnummer'])
        cols = st.sidebar.columns([0.8, 0.2])
        cols[0].write(f"• {title}")
        if cols[1].button("✕", key=f"rem_{i}"):
            st.session_state.cart.pop(i)

    if st.session_state.cart:
        buf = build_excel(st.session_state.cart)
        st.sidebar.download_button(
            "Download Excel",
            data=buf,
            file_name="Anforderungen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.sidebar.write("_Keine Anforderungen ausgewählt_")

    page = st.sidebar.radio('Navigation', [
        'Datenbank Explorer',
        'Semantische Suche / Q&A',
        'Drilldown der Bausteine'
    ])

    # ─── Datenbank Explorer ───
    if page == 'Datenbank Explorer':
        st.header('Datenbank Explorer')
        st.markdown(f"**Gesamt:** {len(df)} Dokumente in der Vektor-DB")
        with st.sidebar:
            st.subheader('Filter')
            art_sel = st.multiselect('Dokument-Art', sorted(df['Art'].unique()), sorted(df['Art'].unique()))
            kat_sel = st.multiselect('Bausteinkategorie', sorted(df['Bausteinkategorie'].unique()),
                                     sorted(df['Bausteinkategorie'].unique()))
            b_sel = st.multiselect('Baustein', sorted(df['Baustein'].unique()), sorted(df['Baustein'].unique()))
            katg_sel = st.multiselect('Anforderungskategorie', sorted(df['Anforderungskategorie'].unique()),
                                      sorted(df['Anforderungskategorie'].unique()))
            rollen_sel = st.multiselect('Rollen', sorted({r for row in df['Rollen'] for r in row.split(', ') if r}),
                                        sorted({r for row in df['Rollen'] for r in row.split(', ') if r}))
            gef_sel = st.multiselect(
                'Zugeordnete Gefahren',
                options=sorted(df['Zugeordnete Gefahren'].unique()),
                default=[]
            )
            sort_col = st.selectbox('Sortiere nach',
                                    ['Bausteinkategorie', 'Baustein', 'Art', 'Anforderungskategorie', 'Titel'], index=0)
            ascending = st.checkbox('Aufsteigend', True)

        mask = (
                df['Art'].isin(art_sel) &
                df['Bausteinkategorie'].isin(kat_sel) &
                df['Baustein'].isin(b_sel) &
                df['Anforderungskategorie'].isin(katg_sel) &
                ((df['Art'] != 'Anforderung') |
                 df['Rollen'].apply(lambda rs: any(r in rs for r in rollen_sel))) &
                df['Zugeordnete Gefahren'].apply(lambda s: all(g in s for g in gef_sel))
        )

        df_filt = df[mask].sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

        st.subheader('Gefilterte Dokumente')

        if AgGrid:
            gb = GridOptionsBuilder.from_dataframe(df_filt)
            gb.configure_default_column(editable=False, wrapText=True, autoHeight=True)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
            grid = AgGrid(df_filt, gridOptions=gb.build(), height=600, fit_columns_on_grid_load=True)
            sel = grid.get('selected_rows') or []
            if sel:
                idx = sel[0]['_selectedRowNodeInfo']['nodeRowIndex']
                st.markdown('---')
                st.subheader('Detail')
                st.write(docs[idx])
        else:
            st.dataframe(df_filt, height=600)

    elif page == 'Semantische Suche / Q&A':
        st.header('Semantische Suche / Q&A')
        only_anf = st.checkbox('Nur nach Anforderungen suchen', value=False)
        query = st.text_input('Suche / Frage eingeben:')
        k = st.slider('Anzahl Ergebnisse', 1, 20, 5)

        if query:
            if only_anf:
                results = vectordb.max_marginal_relevance_search(
                    query, k=k, fetch_k=k * 5, lambda_mult=0.7,
                    filter={"Art": "Anforderung"}
                )
            else:
                results = vectordb.max_marginal_relevance_search(
                    query, k=k, fetch_k=k * 5, lambda_mult=0.7
                )

            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                header = f"**{i}.** {meta.get('baustein_id', '–')} • {meta.get('Art', '–')}"
                if meta.get('Anforderung'):
                    header += f" • {meta['Anforderung']}"
                elif meta.get('Anforderungsnummer'):
                    header += f" • {meta['Anforderungsnummer']}"
                if meta.get('GefahrenID'):
                    header += f" • {meta['GefahrenID']}"
                st.markdown(header)
                st.write(doc.page_content)
                st.caption(meta)
                st.markdown('---')

    else:
        st.header('Drilldown der Bausteine')

        show_entfallen = st.checkbox(
            'Entfallene Anforderungen anzeigen',
            value=False,
            help='Anforderungen mit "ENTFALLEN" im Titel standardmäßig verbergen'
        )
        cia_sel = st.multiselect(
            'Schutzziele filtern',
            options=["Vertraulichkeit","Integrität","Verfügbarkeit"],
            default=[]
        )
        query = st.text_input("Schnellsuche (z.B. 'Server')").strip().lower()
        hier = build_hierarchy(docs, metas)

        for kat_id, modules in sorted(hier.items()):
            if query and not any(module_matches(info, query) for info in modules.values()):
                continue
            kat_title = next(
                (m.get('bausteinkategorie_titel') for m in metas if m.get('bausteinkategorie_id') == kat_id),
                kat_id
            )
            if not st.checkbox(f"{kat_id} – {kat_title}", key=f'kat_{kat_id}', value=bool(query)):
                continue

            for b_id, info in sorted(modules.items(), key=lambda x: _module_sort_key(x[0])):
                if query and not module_matches(info, query):
                    continue
                _, desc, b_meta = info['baustein_docs'][0]
                with st.expander(f"{b_id} – {b_meta.get('baustein_titel', b_id)}", expanded=bool(query)):
                    st.write(desc)
                    st.markdown("**Anforderungen:**")
                    for rid, req in sorted(info['anforderungen'].items(), key=lambda x: _requirement_sort_key(x[0])):
                        meta = req['meta']
                        meta = req['meta']
                        # Falls Schutzziel-Filter aktiv und die Anforderung keins der gewählten Ziele hat, überspringen
                        if cia_sel:
                            if ("Vertraulichkeit" in cia_sel and not meta.get("Vertraulichkeit")) \
                                    or ("Integrität" in cia_sel and not meta.get("Integrität")) \
                                    or ("Verfügbarkeit" in cia_sel and not meta.get("Verfügbarkeit")):
                                continue
                        title = meta.get('Anforderung', rid)
                        if not show_entfallen and "ENTFALLEN" in title:
                            continue
                        is_match = (
                                not query
                                or query in title.lower()
                                or any(query in chunk.lower() for _, chunk in req['chunks'])
                        )
                        if not is_match:
                            continue
                        cols = st.columns([0.7, 0.15, 0.15])
                        if query:
                            cols[0].markdown(f"- **{title}**")
                            if cols[1].button("＋", key=f"add_{kat_id}_{b_id}_{rid}", on_click=add_to_cart,
                                              args=(meta, req['chunks'])):
                                pass
                            for _, chunk in req['chunks']:
                                st.write(chunk)
                            st.caption(meta)
                            if meta.get("zugeordnete_gefahren_titel"):
                                st.markdown(f"**Zugeordnete Gefahren:** {meta['zugeordnete_gefahren_titel']}")
                        else:
                            checked = cols[0].checkbox(title, key=f"tog_{kat_id}_{b_id}_{rid}")
                            if cols[1].button("＋", key=f"add_{kat_id}_{b_id}_{rid}", on_click=add_to_cart,
                                              args=(meta, req['chunks'])):
                                pass
                            if checked:
                                for _, chunk in req['chunks']:
                                    st.write(chunk)
                                st.caption(meta)
                                if meta.get("zugeordnete_gefahren_titel"):
                                    st.markdown(f"**Zugeordnete Gefahren:** {meta['zugeordnete_gefahren_titel']}")


if __name__ == '__main__':
    main()
