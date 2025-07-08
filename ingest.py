import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import torch

from dotenv import load_dotenv

from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")  # Anforderungen
MODULE_RE = re.compile(r"^([A-Z]+(?:\.\d+)+)\s+")  # Bausteine
CHAPTER_RE = re.compile(r"^[A-Z]{2,5}\b")  # Bausteinkategorie
NS = {"db": "http://docbook.org/ns/docbook"}


def _text_from_paras(el: ET.Element) -> str:
    paras = ["".join(p.itertext()).strip() for p in el.findall(".//db:para", NS)]
    return "\n\n".join(filter(None, paras))


def _find_subsection(parent: ET.Element, title: str) -> Optional[ET.Element]:
    for sec in parent.findall("db:section", NS):
        t = sec.find("db:title", NS)
        if t is not None and t.text and t.text.strip() == title:
            return sec
    return None


def extract_structure(xml_path: str) -> List[Dict[str, Any]]:
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
            "bausteinkategorie_title": title,
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
            mod_title = raw[len(mod_id):].strip()

            beschreibung = _text_from_paras(_find_subsection(mod, "Beschreibung") or mod)
            zielsetzung = _text_from_paras(_find_subsection(mod, "Zielsetzung") or mod)
            abgrenzung = _text_from_paras(_find_subsection(mod, "Abgrenzung und Modellierung") or mod)

            threats = []
            th_sec = _find_subsection(mod, "Gefährdungslage")
            if th_sec:
                for t in th_sec.findall("db:section", NS):
                    t_title = t.find("db:title", NS)
                    if t_title is None or not t_title.text:
                        continue
                    threats.append({
                        "Gefährdungslage": t_title.text.strip(),
                        "text": _text_from_paras(t)
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
                    cat = lvl_m.group(1) if lvl_m else "?"
                    roles_m = re.search(r"\[(.+?)\]", full)
                    rollen = [x.strip() for x in roles_m.group(1).split(",")] if roles_m else []
                    requirements.append({
                        "Anforderungsnummer": rid,
                        "Anforderung": full,
                        "Anforderungskategorie": cat,
                        "Rollen": rollen,
                        "text": _text_from_paras(r)
                    })

            kat["bausteine"].append({
                "baustein_id": mod_id,
                "baustein_title": mod_title,
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
            "bausteinkategorie_title": kat["bausteinkategorie_title"],
        }
        for b in kat.get("bausteine", []):
            base_meta = {
                **base_kat_meta,
                "baustein_id": b["baustein_id"],
                "baustein_title": b["baustein_title"],
            }

            full_text = "\n\n".join([
                f"Beschreibung:\n{b['Beschreibung']}",
                f"Zielsetzung:\n{b['Zielsetzung']}",
                f"Abgrenzung und Modellierung:\n{b['Abgrenzung und Modellierung']}"
            ])
            for i, chunk in enumerate(splitter.split_text(full_text)):
                meta = {
                    **base_meta,
                    "Art": "Baustein",
                    "chunk_index": i
                }
                docs.append(Document(page_content=chunk, metadata=meta))

            for thr in b.get("threats", []):
                thr_text = f"{thr['Gefährdungslage']}\n\n{thr['text']}"
                for i, chunk in enumerate(splitter.split_text(thr_text)):
                    meta = {
                        **base_meta,
                        "Art": "Gefahrenlage",
                        "Gefährdungslage": thr['Gefährdungslage'],
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

            for req in b.get("requirements", []):
                for i, chunk in enumerate(splitter.split_text(req["text"])):
                    meta = {
                        **base_meta,
                        "Art": "Anforderung",
                        "Anforderungsnummer": req["Anforderungsnummer"],
                        "Anforderung": req["Anforderung"],
                        "Anforderungskategorie": req["Anforderungskategorie"],
                        "Rollen": ", ".join(req.get("Rollen", [])),
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

    return docs


def main():
    parser = argparse.ArgumentParser(
        description="IT-Grundschutz XML → JSON oder VectorDB"
    )
    parser.add_argument("xml", help="Pfad zur XML-Datei")
    parser.add_argument(
        "--mode", choices=["json", "vectordb"], default="json"
    )
    parser.add_argument(
        "--output",
        help="JSON-Ausgabedatei (json) oder DB-Verzeichnis (vectordb)"
    )
    args = parser.parse_args()

    if not args.output:
        args.output = (
            "resources/requirements.json" if args.mode == "json" else "db"
        )

    bausteinkategorien = extract_structure(args.xml)
    bc = sum(len(k["bausteine"]) for k in bausteinkategorien)
    rc = sum(len(b["requirements"]) for k in bausteinkategorien for b in k["bausteine"])

    if args.mode == 'json':
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(bausteinkategorien, f, indent=2, ensure_ascii=False)
        print(
            f"✅ JSON: {bc} Bausteine, {rc} Anforderungen in {len(bausteinkategorien)} Bausteinkategorien → {args.output}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        docs = modules_to_documents(bausteinkategorien)
        vectordb = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=args.output
        )
        print(f"✅ Chroma: {len(docs)} Dokumente → {args.output}")


if __name__ == '__main__':
    main()
