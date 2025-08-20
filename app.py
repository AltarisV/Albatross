"""
app.py

Streamlit-Anwendung zur interaktiven Suche und Auswahl von IT-Grundschutz-Daten.

Features:
  - Drilldown der Bausteine inkl. Filter nach Schutzzielen
  - Semantische Suche über Vektor-Embedding-Modell (OpenAI)
  - Auswahl und Download der ausgewählten Anforderungen als Excel

Verwendung:
    streamlit run app.py
    python -m streamlit run app.py

Wichtig:
  - Dieser Branch verwendet ausschließlich OpenAI-Embeddings.
  - Setze OPENAI_API_KEY (z. B. via .env + load_dotenv()).
  - Die Chroma-DB muss mit demselben Embedding-Modell erstellt sein
    (im OpenAI-Branch also ingest.py ebenfalls auf OpenAI umstellen und DB neu aufbauen).
"""

import os
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma

# OpenAI-Embeddings (langchain-openai)
try:
    from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai openai
except ImportError as e:
    raise ImportError(
        "langchain-openai ist nicht installiert. "
        "Bitte im OpenAI-Branch installieren: pip install langchain-openai openai python-dotenv"
    ) from e

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except ImportError:
    AgGrid = None

# .env laden (hier war dein Kommentarplatz)
load_dotenv()

# Optional via ENV überschreibbar
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _requirement_sort_key(rid: str):
    """
    Erzeugt einen Sortierschlüssel aus einer Anforderungs-ID am Ende mit 'A' und Ziffern.

    Args:
        rid: Anforderungs-ID (z.B. "ABC.1.2.A12").
    Returns:
        Integer-Teilschlüssel für das Suffix, oder der Original-String bei Fehlschlag.
    """
    m = re.search(r"A(\d+)$", rid)
    return int(m.group(1)) if m else rid


def _module_sort_key(mid: str):
    """
    Liefert für Modul-IDs wie "APP.2.3" ein Tuple (2, 3) für sortierbare Vergleiche.
    IDs ohne Zahlen werden ans Ende sortiert.

    Args:
        mid: Modul-ID-String.
    Returns:
        Tuple[int, int] oder (inf, inf) für ungültige IDs.
    """
    nums = re.findall(r'\d+', mid)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), 0
    else:
        return float('inf'), float('inf')


@st.cache_data(show_spinner=False)
def load_db(persist_dir: str, _embeddings: Embeddings):
    """
    Lädt alle Dokumente und zugehörige Metadaten aus einer Chroma-Vector-Datenbank.

    Args:
        persist_dir: Verzeichnis mit persistierten Vektordaten.
        _embeddings: Embedding-Funktion für die Datenbankinstanz.
    Returns:
        Tuple[List[str], List[dict]]: Dokumenttexte und Metadaten.
    """
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=_embeddings)
    col = vectordb._collection
    data = col.get(limit=col.count())
    return data['documents'], data['metadatas']


def load_db_entries(persist_dir: str, embeddings: Embeddings):
    """
    Erstellt ein pandas DataFrame aus den geladenen DB-Einträgen für Streamlit-Tabellen.

    Args:
        persist_dir: Verzeichnis der Vektor-Datenbank.
        embeddings: Embedding-Instanz (wird an load_db weitergereicht).
    Returns:
        Tuple[pd.DataFrame, List[str], List[dict]]: Tabelle, Dokumenttexte, Metadaten.
    """
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
    """
    Gruppiert Dokumente hierarchisch nach Bausteinkategorie und -ID.

    Args:
        docs: Liste der Dokumenten-Strings.
        metas: Liste der zugehörigen Metadaten.
    Returns:
        Dict: {Kategorie: {Baustein: {"baustein_docs": [...], "anforderungen": {...}}}}
    """
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
    """
    Generiert eine Excel-Datei für die ausgewählten Anforderungen.

    Args:
        requirements: Liste von Dicts mit 'meta' und 'chunks'.
    Returns:
        BytesIO-Puffer mit der Excel-Arbeitsmappe.
    """
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
    """
    Prüft, ob die Suchanfrage in Modul-Titel, Beschreibung oder Anforderungen vorkommt.

    Args:
        info: Dict mit 'baustein_docs' und 'anforderungen'.
        query: Suchbegriff in Kleinbuchstaben.
    Returns:
        bool: True bei Treffer, sonst False.
    """
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
    """
    Fügt eine Anforderung dem Session-State-Warenkorb hinzu, falls noch nicht vorhanden.

    Args:
        meta: Metadaten der Anforderung.
        chunks: Liste von Textabschnitten der Anforderung.
    """
    entry = {'meta': meta, 'chunks': chunks}
    if entry not in st.session_state.cart:
        st.session_state.cart.append(entry)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Embeddings:
    """
    Initialisiert und cached das OpenAI-Embedding-Modell.

    Model:
      - OPENAI_EMBEDDING_MODEL (ENV) oder default 'text-embedding-3-small'
    Requires:
      - OPENAI_API_KEY (ENV / .env)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY ist nicht gesetzt. Bitte .env anlegen oder Env-Var setzen.")
        st.stop()
    # OpenAIEmbeddings liest den Key auch aus der Env; explizit ist aber klarer:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_vectordb(_embeddings: Embeddings):
    """
    Lädt und cached die lokale Chroma-Vektor-Datenbank.

    Hinweis:
        Die DB muss mit demselben Embedding-Modell erstellt worden sein.
    """
    return Chroma(
        persist_directory='db',   # Im OpenAI-Branch kannst du dies bei Bedarf auf 'db_openai' ändern.
        embedding_function=_embeddings
    )


def main():
    """
    Startet die Streamlit-App, richtet Layout und Navigation ein,
    lädt Modell und Datenbank und steuert Seitenlogik.
    """
    st.set_page_config(page_title='Kompendium Finder (OpenAI)', layout='wide')

    st.sidebar.markdown("### Status")
    status = st.sidebar.empty()

    if 'model_loaded' not in st.session_state:
        status.info("📦 Lade OpenAI-Embedding-Modell…")
        embeddings = get_embeddings()
        _ = embeddings.embed_query("initialisierung")
        status.success(f"✅ Modell bereit ({OPENAI_EMBEDDING_MODEL})")
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
        'Drilldown der Bausteine',
        'Semantische Suche'
    ])

    if page == 'Drilldown der Bausteine':
        st.header('Drilldown der Bausteine')

        show_entfallen = st.checkbox(
            'Entfallene Anforderungen anzeigen',
            value=False,
            help='Anforderungen mit "ENTFALLEN" im Titel standardmäßig verbergen'
        )
        cia_sel = st.multiselect(
            'Schutzziele filtern',
            options=["Vertraulichkeit", "Integrität", "Verfügbarkeit"],
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
                        # Schutzziel-Filter
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

    else:
        st.header('Semantische Suche')
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
                cols = st.columns([0.8, 0.2])
                cols[0].markdown(header)
                if cols[1].button("＋", key=f"add_qa_{i}", on_click=add_to_cart,
                                  args=(meta, [(None, doc.page_content)])):
                    st.success("Anforderung hinzugefügt", icon="✅")
                cols[0].write(doc.page_content)
                cols[0].caption(meta)
                st.markdown('---')


if __name__ == '__main__':
    main()
