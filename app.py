import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from st_aggrid import AgGrid, GridOptionsBuilder  # optionales Grid
except ImportError:
    AgGrid = None

load_dotenv()


# --- DB laden ---
def load_db(persist_dir: str, embeds: OpenAIEmbeddings):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeds)
    col = vectordb._collection
    data = col.get(limit=col.count())
    return data['documents'], data['metadatas']


# --- DataFrame für Explorer erzeugen ---
def load_db_entries(persist_dir: str, embeds: OpenAIEmbeddings):
    docs, metas = load_db(persist_dir, embeds)
    rows = []
    for idx, (text, meta) in enumerate(zip(docs, metas)):
        # Rollen: Liste oder String
        raw_roles = meta.get('roles', '')
        roles_str = ", ".join(raw_roles) if isinstance(raw_roles, list) else raw_roles

        # Titel je nach Typ: bei Modulen und Threats aus Modul-Metadaten, bei Requirements aus title-Feld
        typ = meta.get('type', 'module')
        if typ == "module":
            title = meta.get("module_id", "")
            # falls beim Ingest mitgespeichert: meta["module_title"]
            if meta.get("module_title"):
                title += " – " + meta["module_title"]
        elif typ == "threat":
            title = f"{meta.get('threat_title', '')}"
        else:  # requirement
            title = meta.get("requirement_title", meta.get("requirement_id", ""))

        rows.append({
            "ID": idx,
            "type": typ,
            "chapter": meta.get("chapter_id", ""),
            "module": meta.get("module_id", ""),
            "title": title,
            "requirement_id": meta.get("requirement_id", ""),
            "threat_title": meta.get("threat_title", ""),
            "level": meta.get("level", ""),
            "roles": roles_str,
            "snippet": (text.replace("\n", " ")[:300] + "…") if len(text) > 300 else text
        })

    df = pd.DataFrame(rows)
    return df, docs, metas


# --- Hierarchie aus Metadaten bauen ---
def build_hierarchy(docs, metas):
    hier = {}
    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        chap = meta.get('chapter_id', 'UNKNOWN')
        mod = meta.get('module_id', 'UNKNOWN')
        typ = meta.get('type', 'module')
        chapter = hier.setdefault(chap, {})
        module = chapter.setdefault(mod, {'module_docs': [], 'requirements': {}})
        if typ == 'module':
            module['module_docs'].append((idx, doc, meta))
        elif typ == 'requirement':
            rid = meta.get('requirement_id', 'UNKNOWN')
            req = module['requirements'].setdefault(rid, {'chunks': [], 'meta': meta})
            req['chunks'].append((idx, doc))
    return hier


# --- Streamlit App ---
def main():
    st.set_page_config(page_title='Kompendium Explorer', layout='wide')
    st.title('IT-Grundschutz Kompendium – Explorer')

    page = st.sidebar.radio('Navigation', [
        'Datenbank Explorer',
        'Semantische Suche / Q&A',
        'Drilldown der Bausteine'
    ])

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error('OPENAI_API_KEY fehlt in der Umgebung')
        return
    embeds = OpenAIEmbeddings(openai_api_key=api_key)

    # Datenbank Explorer
    if page == 'Datenbank Explorer':
        st.header('Datenbank Explorer')
        df, docs, metas = load_db_entries('db', embeds)
        st.markdown(f"**Gesamt:** {len(df)} Dokumente in der Vektor-DB")

        # Sidebar-Filter
        with st.sidebar:
            st.subheader('Filter')
            # Jetzt kann man nach Typ filtern (Module, Threat, Requirement)
            type_sel = st.multiselect(
                'Dokument-Typ',
                options=sorted(df['type'].unique()),
                default=sorted(df['type'].unique())
            )
            chapters_sel = st.multiselect(
                'Kapitel',
                options=sorted(df['chapter'].unique()),
                default=sorted(df['chapter'].unique())
            )
            modules_sel = st.multiselect(
                'Baustein-ID',
                options=sorted(df['module'].unique()),
                default=sorted(df['module'].unique())
            )
            level_sel = st.multiselect(
                'Level (B/S/H)',
                options=sorted(df['level'].unique()),
                default=sorted(df['level'].unique())
            )
            roles_sel = st.multiselect(
                'Rollen',
                options=sorted({r for row in df['roles'] for r in row.split(', ') if r}),
                default=sorted({r for row in df['roles'] for r in row.split(', ') if r})
            )
            sort_col = st.selectbox(
                'Sortiere nach',
                ['chapter', 'module', 'type', 'level', 'title'],
                index=0
            )
            ascending = st.checkbox('Aufsteigend', True)

        mask = (
                df['type'].isin(type_sel) &
                df['chapter'].isin(chapters_sel) &
                df['module'].isin(modules_sel) &
                df['level'].isin(level_sel) &
                (
                        (df['type'] != 'requirement')  # Module und Threats immer behalten
                        |
                        df['roles'].apply(lambda rs: any(r in rs for r in roles_sel))
                )
        )
        df_filt = df[mask].sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

        st.subheader('Gefilterte Dokumente')
        # Mit AgGrid (wenn installiert)
        if AgGrid:
            gb = GridOptionsBuilder.from_dataframe(df_filt)
            gb.configure_default_column(editable=False, wrapText=True, autoHeight=True)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
            grid = AgGrid(
                df_filt,
                gridOptions=gb.build(),
                height=600,
                fit_columns_on_grid_load=True
            )
            sel = grid.get('selected_rows') or []
            if sel:
                idx = sel[0]['_selectedRowNodeInfo']['nodeRowIndex']
                st.markdown('---')
                st.subheader('Detail')
                st.write(docs[idx])
        else:
            st.dataframe(df_filt, height=600)

    # Q&A
    elif page == 'Semantische Suche / Q&A':
        st.header('Semantische Suche / Q&A')
        only_req = st.checkbox('Nur nach Requirements suchen', value=False)
        query = st.text_input('Suche / Frage eingeben:')
        k = st.slider('Anzahl Ergebnisse', 1, 20, 5)
        if query:
            vdb = Chroma(persist_directory='db', embedding_function=embeds)

            if only_req:
                results = vdb.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=k*5,
                    lambda_mult=0.7,
                    filter={"type": "requirement"},
                )
            else:
                results = vdb.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=k*5,
                    lambda_mult=0.7,
                )

            # Jetzt hast Du genau k Docs des gewünschten Typs (oder weniger, wenn nicht genug da sind)
            for i, doc in enumerate(results, start=1):
                meta = doc.metadata
                header = f"**{i}.** {meta.get('module_id', '–')} • {meta.get('type', '–')}"
                if meta.get('requirement_title'):
                    header += f" • {meta['requirement_title']}"
                elif meta.get('requirement_id'):
                    header += f" • {meta['requirement_id']}"
                if meta.get('threat_title'):
                    header += f" • THREAT: {meta['threat_title']}"

                st.markdown(header)
                st.write(doc.page_content)
                st.caption(meta)
                st.markdown('---')

    elif page == 'Drilldown der Bausteine':
        st.header('Drilldown der Bausteine')

        # 1) Quicksearch-Feld
        query = st.text_input("Schnellsuche (z.B. 'Server')", value="").strip().lower()

        # 2) Rohdaten laden und Hierarchie bauen
        docs, metas = load_db('db', embeds)
        hier = build_hierarchy(docs, metas)

        # 3) Kapitel durchlaufen
        for chap_id, modules in sorted(hier.items()):
            # Kapitel-Titel aus Metadaten (oder fallback auf chap_id)
            chap_title = next(
                (m.get('chapter_title') for m in metas
                 if m.get('chapter_id') == chap_id and m.get('chapter_title')),
                chap_id
            )

            # Wir wollen das Kapitel nur anzeigen, wenn
            # entweder kein Query gesetzt ist ODER mindestens ein Kind passt
            # -> compute a flag:
            def module_matches(mod_info):
                # sucht in Modul-Titel und Beschreibung
                _, desc, mod_meta = mod_info['module_docs'][0]
                module_title = mod_meta.get('module_title', '')
                if query and query in module_title.lower(): return True
                if query and query in desc.lower(): return True
                # oder in einer der Anforderungen
                for rid, req in mod_info['requirements'].items():
                    title = req['meta'].get('requirement_title', '').lower()
                    if query in title: return True
                    # optional: suche auch im Chunk-Text
                    for _, chunk in req['chunks']:
                        if query in chunk.lower(): return True
                return False

            # Kapitel-Flag: mindestens ein Modul matcht?
            chap_has = (not query) or any(module_matches(info) for info in modules.values())
            if not chap_has:
                continue

            # Kapitel-Checkbox (oder Expander – hier Checkbox wie gehabt)
            if not st.checkbox(f"{chap_id} – {chap_title}", key=f"chap_{chap_id}", value=bool(query)):
                continue

            # 4) Module durchlaufen
            for mod_id, info in sorted(modules.items()):
                # Modul-Metadaten
                _, desc, mod_meta = info['module_docs'][0]
                module_title = mod_meta.get('module_title', mod_id)
                full_mod_label = f"{mod_id} – {module_title}"

                # filter Modul?
                mod_has = (not query) or module_matches(info)
                if not mod_has:
                    continue

                # Expander für das Modul, automatisch expanded wenn Query passt
                with st.expander(full_mod_label, expanded=bool(query)):
                    # Beschreibung
                    st.markdown("**Beschreibung:**")
                    st.write(desc)

                    # Anforderungen
                    st.markdown("**Anforderungen:**")
                    for rid, req in sorted(info['requirements'].items()):
                        req_meta = req['meta']
                        req_title = req_meta.get('requirement_title', rid)

                        # Prüfen, ob diese Anforderung zum Filter passt
                        req_match = False
                        if not query:
                            req_match = False
                        else:
                            low = req_title.lower()
                            if query in low:
                                req_match = True
                            else:
                                for _, chunk in req['chunks']:
                                    if query in chunk.lower():
                                        req_match = True
                                        break

                        if query:
                            # bei Query nur die gematchten Anforderungen ausgeben,
                            # und direkt den Inhalt anzeigen
                            if not req_match:
                                continue
                            st.markdown(f"- **{req_title}**")
                            for _, chunk in req['chunks']:
                                st.write(chunk)
                            st.caption(req_meta)
                        else:
                            # kein Query: wie gehabt mit Checkbox
                            if st.checkbox(req_title, key=f"req_{chap_id}_{mod_id}_{rid}"):
                                for _, chunk in req['chunks']:
                                    st.write(chunk)
                                st.caption(req_meta)

        st.markdown("---")


if __name__ == '__main__':
    main()
