import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import torch

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")
MODULE_RE = re.compile(r"^([A-Z]+(?:\.\d+)+)\s+")
CHAPTER_RE = re.compile(r"^[A-Z]{2,5}\b")
NS = {"db": "http://docbook.org/ns/docbook"}

EXCEL_MAP_FILE = "resources/krt2023_Excel.xlsx"


def _text_from_paras(el: ET.Element) -> str:
    paras = ["".join(p.itertext()).strip() for p in el.findall(".//db:para", NS)]
    return "\n\n".join(filter(None, paras))


def _find_subsection(parent: ET.Element, title: str) -> Optional[ET.Element]:
    for sec in parent.findall("db:section", NS):
        t = sec.find("db:title", NS)
        if t is not None and t.text and t.text.strip() == title:
            return sec
    return None


def load_excel_mapping(excel_path: str) -> Dict[str, List[str]]:
    """
    Liest alle Sheets der Excel-Datei ein und erstellt ein Dict:
      { Anforderungs-ID: [Gefahren-ID, …], … }
    """
    mapping: Dict[str, List[str]] = {}
    sheets = pd.read_excel(excel_path, sheet_name=None)
    for df in sheets.values():
        req_ids = df.iloc[:, 0].fillna("").astype(str)
        threat_cols = df.columns[3:]
        flag_matrix = df.iloc[:, 3:].fillna("")
        for idx, rid in enumerate(req_ids):
            if not rid.strip():
                continue
            vals = flag_matrix.iloc[idx]
            zugeordnete = [col for col, mark in zip(threat_cols, vals) if str(mark).strip()]
            mapping[rid] = zugeordnete
    return mapping


def load_threat_titles(xml_path: str) -> Dict[str, str]:
    """
    Extrahiert aus dem Kapitel "Elementare Gefährdungen" alle IDs ("G 0.x") und Titel.
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
                    titles[gid] = titel
            break
    return titles


def extract_structure(
        xml_path: str,
        gefahr_map: Dict[str, List[str]],
        gefahr_titel: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Lädt das XML, splittert Text, extrahiert Bausteine und Anforderungen
    und ergänzt für jede Anforderung:
      - zugeordnete_gefahren (Liste von G-IDs)
      - zugeordnete_gefahren_titel (Liste von Titeln)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bausteinkategorien: List[Dict[str, Any]] = []

    for chap in root.findall("db:chapter", NS):
        ct = chap.find("db:title", NS)
        if ct is None or not ct.text:
            continue
        title = ct.text.strip()
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

            beschreibung = _text_from_paras(_find_subsection(mod, "Beschreibung") or mod)
            zielsetzung = _text_from_paras(_find_subsection(mod, "Zielsetzung") or mod)
            abgrenzung = _text_from_paras(_find_subsection(mod, "Abgrenzung und Modellierung") or mod)

            threats = []
            th_sec = _find_subsection(mod, "Gefährdungslage")
            if th_sec:
                for tsec in th_sec.findall("db:section", NS):
                    tt = tsec.find("db:title", NS)
                    if tt is None or not tt.text:
                        continue
                    threats.append({
                        "gefahren_id": tt.text.strip(),
                        "text": _text_from_paras(tsec)
                    })

            requirements = []
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
                    lvl_m = re.search(r"\((B|S|H)\)", full)
                    katg = lvl_m.group(1) if lvl_m else "?"
                    roles_m = re.search(r"\[(.+?)\]", full)
                    rollen = [x.strip() for x in roles_m.group(1).split(",")] if roles_m else []

                    zugeordnete = gefahr_map.get(rid, [])
                    titel_list = [gefahr_titel.get(g, g) for g in zugeordnete]

                    requirements.append({
                        "Anforderungsnummer": rid,
                        "Anforderung": full,
                        "Anforderungskategorie": katg,
                        "Rollen": rollen,
                        "zugeordnete_gefahren": zugeordnete,
                        "zugeordnete_gefahren_titel": titel_list,
                        "text": _text_from_paras(r)
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


def modules_to_documents(bausteinkategorien: List[Dict[str, Any]]) -> List[Document]:
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

            full_text = "\n\n".join([
                f"Beschreibung:\n{b['Beschreibung']}",
                f"Zielsetzung:\n{b['Zielsetzung']}",
                f"Abgrenzung und Modellierung:\n{b['Abgrenzung und Modellierung']}"
            ])
            for i, chunk in enumerate(splitter.split_text(full_text)):
                meta = {**base_meta, "Art": "Baustein", "chunk_index": i}
                docs.append(Document(page_content=chunk, metadata=meta))

            # Gefahren-Chunks
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

            # Anforderungs-Chunks
            for req in b["requirements"]:
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
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

    return docs


def main():
    parser = argparse.ArgumentParser("IT-Grundschutz XML → JSON oder VectorDB")
    parser.add_argument("xml", help="Pfad zur XML-Datei")
    parser.add_argument("--mode", choices=["json", "vectordb"], default="json")
    parser.add_argument("--output", help="JSON-Datei (json) oder DB-Verzeichnis (vectordb)")
    args = parser.parse_args()

    xml_path = args.xml
    out = args.output or ("resources/requirements.json" if args.mode == "json" else "db")

    gefahr_map = load_excel_mapping(EXCEL_MAP_FILE)
    gefahr_titel = load_threat_titles(xml_path)

    bausteinkategorien = extract_structure(xml_path, gefahr_map, gefahr_titel)
    bc = sum(len(k["bausteine"]) for k in bausteinkategorien)
    rc = sum(len(b["requirements"]) for k in bausteinkategorien for b in k["bausteine"])

    if args.mode == "json":
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(bausteinkategorien, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON: {bc} Bausteine, {rc} Anforderungen → {out}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        docs = modules_to_documents(bausteinkategorien)
        Chroma.from_documents(docs, embeddings, persist_directory=out)
        print(f"✅ Chroma: {len(docs)} Dokumente → {out}")


if __name__ == "__main__":
    main()
