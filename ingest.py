"""
ingest.py

Skript zur Verarbeitung von IT-Grundschutz-Daten.

Dieses Modul liest:
  - eine XML-Datei mit Baustein- und Anforderungsstruktur
  - eine Excel-Datei mit Zuordnungen von Anforderungen zu Gefährdungen und CIA-Kategorien (KRT)
  - eine CSV mit der BSI-200-3-Tabelle (Gefährdung -> CIA)

Ziel:
  - Für Anforderungen mit direktem CIA aus der KRT wird dieses übernommen.
  - Für Anforderungen ohne direktes CIA wird das CIA aus den zugeordneten Gefährdungen
    über die BSI-200-3-Tabelle abgeleitet (Union der Schutzziele über alle Gefährdungen).

Ausgabe:
  - JSON mit der angereicherten Struktur (Bausteine, Gefährdungen, Anforderungen).
  - Optional direkte Befüllung einer Chroma-Vektordatenbank mit Embeddings.

Usage:
    python ingest.py <xml_path> [--mode json|vectordb] [--output OUTPUT]

Hinweise:
  - Falls eine Anforderung bereits in der Excel (KRT) ein CIA-Feld hat,
    wird dieses übernommen. Nur wenn es leer ist, werden die Gefährdungen
    herangezogen und per BSI-200-3-Mapping auf CIA abgebildet.
  - Diese Version verwendet ausschließlich OpenAI-Embeddings. Setze OPENAI_API_KEY
    (z. B. via .env + load_dotenv). Das Embedding-Modell ist per ENV
    OPENAI_EMBEDDING_MODEL überschreibbar (Default: 'text-embedding-3-small').
"""

import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# .env laden (für OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, etc.)
load_dotenv()

# --------------------------------------------------------------------
# Konfiguration / Konstanten
# --------------------------------------------------------------------

# Regex für Requirement-IDs im Format "ABC.1.2.A3"
REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")
# Regex zur Erkennung von Modul-IDs und -Titeln: "ABC.1.2  Titel"
MODULE_RE = re.compile(r"^([A-Z]+(?:\.\d+)+)\s+")
# Kapitel-IDs im XML sind Kurzbezeichnungen aus 2–5 Buchstaben
CHAPTER_RE = re.compile(r"^[A-Z]{2,5}\b")
# XML-Namespace für DocBook-Elemente
NS = {"db": "http://docbook.org/ns/docbook"}

# Pfad zur Excel-Datei mit Zuordnungen (Mapping Requirement → Gefährdungen, CIA)
EXCEL_MAP_FILE = "resources/krt2023_Excel.xlsx"
# Gefährdung→Schutzziel-Tabelle aus BSI-Standard 200-3
BSI2023_THREAT2CIA_CSV = "resources/bsi2023_threats_to_cia.csv"

# OpenAI-Embedding-Modell (per ENV überschreibbar)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# --------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------

def _normalize_threat_id(s: str) -> str:
    """
    Normalisiert Gefährdungs-IDs auf das Format 'G 0.x'.

    Unterstützte Eingaben:
      - 'G0.1', 'G 0.01', '0.1', '0.01' → 'G 0.1'
      - 'G 0.10' bleibt 'G 0.10' (nur führende Nullen im rechten Teil werden entfernt)

    Args:
        s: Ursprüngliche Gefährdungs-ID (beliebige Schreibweise).
    Returns:
        Normalisierte Gefährdungs-ID als 'G 0.x'.
    """
    if not s:
        return s
    s = str(s).strip()
    s = s.replace("G", "").replace("g", "").strip()
    if s.startswith("0."):
        _, right = s.split(".", 1)
        try:
            right = str(int(right))  # führende Nullen entfernen
        except ValueError:
            right = right.lstrip("0") or "0"
        return f"G 0.{right}"
    s = s.replace("  ", " ")
    return f"G {s}" if not s.startswith("G ") else s


def _text_from_paras(el: ET.Element) -> str:
    """
    Extrahiert alle <para>-Texte aus einem XML-Element und gibt sie
    als durch Leerzeilen getrennten String zurück.

    Args:
        el: XML-Element, in dem nach <db:para> gesucht wird.
    Returns:
        Zusammengesetzter Text aller Absatz-Elemente.
    """
    paras = ["".join(p.itertext()).strip() for p in el.findall(".//db:para", NS)]
    return "\n\n".join(filter(None, paras))


def _find_subsection(parent: ET.Element, title: str) -> Optional[ET.Element]:
    """
    Sucht untergeordnete <section>-Elemente mit einem bestimmten <title>.

    Args:
        parent: XML-Element, unter dem gesucht wird.
        title: Titel-Text des gesuchten Abschnitts.
    Returns:
        Erstes gefundenes <section>-Element oder None.
    """
    for sec in parent.findall("db:section", NS):
        t = sec.find("db:title", NS)
        if t is not None and t.text and t.text.strip() == title:
            return sec
    return None


def _split_cia_string(x: str) -> List[str]:
    """
    Teilt einen String wie 'C,I,A' in eine Liste ['C','I','A'] auf.

    Args:
        x: String mit durch Komma getrennten Schutzzielen (C, I, A).
    Returns:
        Liste der einzelnen Buchstaben (ohne Leerzeichen).
    """
    if not x:
        return []
    return [p.strip() for p in str(x).split(",") if p and p.strip()]


# --------------------------------------------------------------------
# Datenquellen laden
# --------------------------------------------------------------------

def load_bsi2023_threat2cia(csv_path: str) -> Dict[str, List[str]]:
    """
    Lädt die BSI-200-3 Mapping-Tabelle (Gefährdung 'G 0.x' → 'C,I,A') aus CSV.

    Erwartete CSV-Spalten (Semikolon-getrennt):
      - 'threat_id': Gefährdungs-ID (z. B. 'G 0.1' oder abweichende Schreibweise)
      - 'cia': Komma-getrennte Liste von Schutzzielen (z. B. 'C,I,A' oder 'A')

    Args:
        csv_path: Pfad zur CSV.
    Returns:
        Dict[str, List[str]]: Mapping 'G 0.x' → ['C','I','A'] (Liste kann 1–3 Elemente enthalten).
    Raises:
        FileNotFoundError: falls Datei nicht existiert.
        ValueError: falls Pflichtspalten fehlen.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV '{csv_path}' nicht gefunden.")
    df = pd.read_csv(csv_path, sep=";")
    if "threat_id" not in df.columns or "cia" not in df.columns:
        raise ValueError("CSV muss Spalten 'threat_id' und 'cia' enthalten.")
    df["threat_id_norm"] = df["threat_id"].apply(_normalize_threat_id)
    return dict(zip(df["threat_id_norm"], df["cia"].apply(_split_cia_string)))


def load_excel_mapping(excel_path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Liest alle Sheets der Excel-Datei (KRT) ein und erstellt zwei Dicts:
      - mapping: { Anforderungs-ID: [Gefährdungs-ID, …] }
      - cia_map: { Anforderungs-ID: 'C,I,A' | '' }

    Annahmen:
      - Spalte 0: Anforderungs-ID
      - Spalte 2: 'CIA' (kann leer sein)
      - Ab Spalte 3: Gefährdungs-Spalten (z. B. 'G 0.1', 'G 0.2', ...),
        deren Zellen markiert sind, wenn die Gefährdung zugeordnet ist.

    Args:
        excel_path: Pfad zur Excel-Datei mit KRT-Zuordnungen.
    Returns:
        Tuple (mapping, cia_map).
    """
    mapping: Dict[str, List[str]] = {}
    cia_map: Dict[str, str] = {}
    sheets = pd.read_excel(excel_path, sheet_name=None)

    for df in sheets.values():
        if df.shape[1] < 4 or "CIA" not in df.columns:
            continue

        req_ids = df.iloc[:, 0].fillna("").astype(str)
        threat_cols = [str(c) for c in df.columns[3:]]
        flag_matrix = df.iloc[:, 3:].fillna("")
        cia_vals = df["CIA"].fillna("").astype(str)
        threat_cols_norm = [_normalize_threat_id(c) for c in threat_cols]

        for idx, rid in enumerate(req_ids):
            if not rid.strip():
                continue
            row_flags = flag_matrix.iloc[idx]
            zugeordnete = [
                col_norm
                for col_norm, orig_col, mark in zip(threat_cols_norm, threat_cols, row_flags)
                if str(mark).strip()
            ]
            mapping[rid] = zugeordnete
            cia_map[rid] = cia_vals.iloc[idx].strip()
    return mapping, cia_map


def load_threat_titles(xml_path: str) -> Dict[str, str]:
    """
    Extrahiert aus dem Kapitel "Elementare Gefährdungen" alle Gefährdungs-IDs und -Titel.

    Args:
        xml_path: Pfad zur XML-Datei im DocBook-Format.
    Returns:
        Dict[str, str]: Mapping Gefährdungs-ID (z.B. 'G 0.1') → Gefährdungs-Titel.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    titles: Dict[str, str] = {}

    for chap in root.findall("db:chapter", NS):
        ct = chap.find("db:title", NS)
        if ct is not None and ct.text and ct.text.strip() == "Elementare Gefährdungen":
            for sec in chap.findall("db:section", NS):
                t = sec.find("db:title", NS)
                if t is None or not t.text:
                    continue
                raw = t.text.strip()
                parts = raw.split(maxsplit=2)
                if len(parts) >= 2:
                    gid = f"{parts[0]} {parts[1]}"
                    titel = raw[len(gid):].strip()
                    titles[_normalize_threat_id(gid)] = titel
            break
    return titles


# --------------------------------------------------------------------
# Extraktion & Ableitung
# --------------------------------------------------------------------

def extract_structure(
        xml_path: str,
        gefahr_map: Dict[str, List[str]],
        gefahr_titel: Dict[str, str],
        cia_map: Dict[str, str],
        threat2cia: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Parst das XML, extrahiert Bausteine und Anforderungen, reichert Anforderungen
    mit Gefährdungen und CIA-Informationen an (direkt oder abgeleitet).

    Logik:
      1) Wenn eine Anforderung in der KRT ein direktes CIA hat, wird dieses übernommen.
      2) Andernfalls wird CIA via Threat-Union abgeleitet:
         Anforderung → (zugeordnete Gefährdungen) → (BSI-200-3: Gefährdung→CIA) → Union(CIA)

    Args:
        xml_path: Pfad zur XML-Datei des IT-Grundschutz-Kompendiums.
        gefahr_map: Mapping Requirement-ID → Liste von Gefährdungs-IDs (normalisierte IDs empfohlen).
        gefahr_titel: Mapping Gefährdungs-ID → Titel der Gefährdung.
        cia_map: Mapping Requirement-ID → Roh-CIA aus der KRT (z. B. 'C,I,A' oder '').
        threat2cia: Mapping Gefährdungs-ID → Liste von Schutzzielen ['C','I','A'].
    Returns:
        Liste von Bausteinkategorien mit allen Metadaten (Bausteine, Gefährdungen, Anforderungen).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bausteinkategorien: List[Dict[str, Any]] = []

    for chap in root.findall("db:chapter", NS):
        ct = chap.find("db:title", NS)
        if ct is None or not ct.text:
            continue
        title = ct.text.strip()
        # Nur Kapitel, die mit Buchstaben-ID beginnen (z. B. APP, SYS, OPS, ...)
        if not CHAPTER_RE.match(title):
            continue
        chap_id = title.split()[0]
        kat = {
            "bausteinkategorie_id": chap_id,
            "bausteinkategorie_titel": title,
            "bausteine": []
        }

        for mod in chap.findall("db:section", NS):
            mt = mod.find("db:title", NS)
            if mt is None or not mt.text:
                continue
            raw = mt.text.strip()
            m = MODULE_RE.match(raw)
            if not m:
                continue
            mod_id = m.group(1)
            mod_titel = raw[len(mod_id):].strip()

            # Haupttexte (optional vorhandene Unterabschnitte)
            beschreibung = _text_from_paras(_find_subsection(mod, "Beschreibung") or mod)
            zielsetzung = _text_from_paras(_find_subsection(mod, "Zielsetzung") or mod)
            abgrenzung = _text_from_paras(_find_subsection(mod, "Abgrenzung und Modellierung") or mod)

            # Gefährdungslage (freiwillig, für Vollständigkeit im Index)
            threats: List[Dict[str, str]] = []
            th_sec = _find_subsection(mod, "Gefährdungslage")
            if th_sec:
                for tsec in th_sec.findall("db:section", NS):
                    tt = tsec.find("db:title", NS)
                    if tt is None or not tt.text:
                        continue
                    threats.append({
                        "gefahren_id": _normalize_threat_id(tt.text.strip()),
                        "text": _text_from_paras(tsec)
                    })

            # Anforderungen
            requirements: List[Dict[str, Any]] = []
            req_root = _find_subsection(mod, "Anforderungen")
            if req_root:
                for r in req_root.findall(".//db:section", NS):
                    r_title = r.find("db:title", NS)
                    if r_title is None or not r_title.text:
                        continue
                    full = r_title.text.strip()
                    if not REQ_RE.match(full):
                        continue
                    rid = full.split()[0]
                    # Extrahiere Schutzbedarfsanforderung (Basis/Standard/Erhöhter Schutzbedarf) und Rollen
                    lvl_m = re.search(r"\((B|S|H)\)", full)
                    katg = lvl_m.group(1) if lvl_m else "?"
                    roles_m = re.search(r"\[(.+?)\]", full)
                    rollen = [x.strip() for x in roles_m.group(1).split(",")] if roles_m else []

                    # Threats der Anforderung (normalisiert) und Titel
                    zugeordnete = [_normalize_threat_id(g) for g in gefahr_map.get(rid, [])]
                    titel_list = [gefahr_titel.get(g, g) for g in zugeordnete]

                    # 1) Direktes CIA aus der KRT?
                    raw_cia = (cia_map.get(rid, "") or "").strip()
                    cia_source = "krt" if raw_cia else None

                    # 2) Falls leer → aus Threats ableiten (Union)
                    if not raw_cia and zugeordnete:
                        cia_set = set()
                        for g in zugeordnete:
                            cia_set.update(threat2cia.get(g, []))
                        raw_cia = ",".join(sorted(cia_set))
                        if raw_cia:
                            cia_source = "200-3-threat-map"

                    # CIA-Flags
                    vertraulichkeit = "C" in raw_cia
                    integritaet = "I" in raw_cia
                    verfuegbarkeit = "A" in raw_cia

                    requirements.append({
                        "Anforderungsnummer": rid,
                        "Anforderung": full,
                        "Anforderungskategorie": katg,
                        "Rollen": rollen,
                        "zugeordnete_gefahren": zugeordnete,
                        "zugeordnete_gefahren_titel": titel_list,
                        "text": _text_from_paras(r),
                        "CIA_Roh": raw_cia,
                        "CIA_Quelle": cia_source,
                        "Vertraulichkeit": vertraulichkeit,
                        "Integrität": integritaet,
                        "Verfügbarkeit": verfuegbarkeit
                    })

            kat["bausteine"].append({
                "baustein_id": mod_id,
                "baustein_titel": mod_titel,
                "Beschreibung": beschreibung,
                "Zielsetzung": zielsetzung,
                "Abgrenzung und Modellierung": abgrenzung,
                "threats": threats,
                "requirements": requirements
            })

        if kat["bausteine"]:
            bausteinkategorien.append(kat)

    return bausteinkategorien


# --------------------------------------------------------------------
# Umwandlung in Vektordokumente
# --------------------------------------------------------------------

def modules_to_documents(bausteinkategorien: List[Dict[str, Any]]) -> List[Document]:
    """
    Wandelt die extrahierten Baustein-Daten in eine Liste von LangChain-Documents um.

    Jedes Objekt (Baustein, Gefährdung, Anforderung) wird in Text-Chunks
    von maximal 1000 Zeichen (Overlap 200) aufgeteilt. Metadaten werden
    für die gezielte Filterung in der Vektordatenbank mitgegeben.

    Args:
        bausteinkategorien: Liste der Kategorien mit ihren Bausteinen und Anforderungen.
    Returns:
        Liste von langchain.schema.Document-Objekten, bereit für das Embedding.
    """
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for kat in bausteinkategorien:
        base_kat_meta = {
            "bausteinkategorie_id": kat["bausteinkategorie_id"],
            "bausteinkategorie_titel": kat["bausteinkategorie_titel"],
        }
        for b in kat["bausteine"]:
            base_meta = {
                **base_kat_meta,
                "baustein_id": b["baustein_id"],
                "baustein_titel": b["baustein_titel"],
            }

            # Baustein-Haupttext
            full_text = "\n\n".join([
                f"Beschreibung:\n{b['Beschreibung']}",
                f"Zielsetzung:\n{b['Zielsetzung']}",
                f"Abgrenzung und Modellierung:\n{b['Abgrenzung und Modellierung']}"
            ])
            for i, chunk in enumerate(splitter.split_text(full_text)):
                meta = {**base_meta, "Art": "Baustein", "chunk_index": i}
                docs.append(Document(page_content=chunk, metadata=meta))

            # Gefährdungen (aus der Gefährdungslage des Bausteins)
            for thr in b["threats"]:
                thr_text = f"{thr['gefahren_id']}\n\n{thr['text']}"
                for i, chunk in enumerate(splitter.split_text(thr_text)):
                    meta = {
                        **base_meta,
                        "Art": "Gefahrenlage",
                        "GefahrenID": thr["gefahren_id"],
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

            # Anforderungen
            for req in b["requirements"]:
                cia_list = _split_cia_string(req.get("CIA_Roh", ""))
                for i, chunk in enumerate(splitter.split_text(req["text"])):
                    meta = {
                        **base_meta,
                        "Art": "Anforderung",
                        "Anforderungsnummer": req["Anforderungsnummer"],
                        "Anforderung": req["Anforderung"],
                        "Anforderungskategorie": req["Anforderungskategorie"],
                        "Rollen": ", ".join(req["Rollen"]),
                        "zugeordnete_gefahren": ", ".join(req["zugeordnete_gefahren"]),
                        "zugeordnete_gefahren_titel": ", ".join(req["zugeordnete_gefahren_titel"]),
                        "chunk_index": i,
                        # CIA-Infos
                        "CIA": ",".join(cia_list),
                        "CIA_Quelle": req.get("CIA_Quelle"),
                        "Vertraulichkeit": req["Vertraulichkeit"],
                        "Integrität": req["Integrität"],
                        "Verfügbarkeit": req["Verfügbarkeit"],
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

    return docs


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main() -> None:
    """
    CLI-Einstiegspunkt.

    Modi:
      - 'json': exportiert die angereicherte Struktur als JSON-Datei.
      - 'vectordb': erzeugt Embeddings und befüllt eine Chroma-Vektordatenbank.

    CLI-Argumente:
      xml (str): Pfad zur XML-Datei.
      --mode (str): 'json' (Default) oder 'vectordb'.
      --output (str): Zielpfad (bei 'json' eine Datei, bei 'vectordb' ein Verzeichnis).

    Effekte:
      - Liest KRT-Excel und BSI-200-3 CSV.
      - Extrahiert Struktur aus XML und leitet CIA ggf. aus Gefährdungen ab.
      - Schreibt JSON oder befüllt Chroma (persist_directory=--output).
    """
    parser = argparse.ArgumentParser(description="IT-Grundschutz XML → JSON oder VectorDB")
    parser.add_argument("xml", help="Pfad zur XML-Datei")
    parser.add_argument("--mode", choices=["json", "vectordb"], default="json")
    parser.add_argument("--output", help="JSON-Datei (json) oder DB-Verzeichnis (vectordb)")
    args = parser.parse_args()

    xml_path = args.xml
    out = args.output or ("resources/requirements.json" if args.mode == "json" else "db")

    # Check OpenAI Key frühzeitig
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY ist nicht gesetzt. Bitte .env anlegen oder Env-Var setzen.")

    # Excel-Mapping & Threat-Titel laden
    gefahr_map, cia_map = load_excel_mapping(EXCEL_MAP_FILE)
    gefahr_titel = load_threat_titles(xml_path)

    # BSI-200-3 Threat→CIA laden
    threat2cia = load_bsi2023_threat2cia(BSI2023_THREAT2CIA_CSV)

    # Struktur extrahieren (inkl. CIA-Ableitung)
    bausteinkategorien = extract_structure(xml_path, gefahr_map, gefahr_titel, cia_map, threat2cia)
    bc = sum(len(k["bausteine"]) for k in bausteinkategorien)
    rc = sum(len(b["requirements"]) for k in bausteinkategorien for b in k["bausteine"])

    if args.mode == "json":
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(bausteinkategorien, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON: {bc} Bausteine, {rc} Anforderungen → {out}")
    else:
        # OpenAI-Embeddings (keine GPU/torch-Konfiguration nötig)
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        docs = modules_to_documents(bausteinkategorien)
        Chroma.from_documents(docs, embeddings, persist_directory=out)
        print(f"✅ Chroma: {len(docs)} Dokumente → {out}")


if __name__ == "__main__":
    main()
