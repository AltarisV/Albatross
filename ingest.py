import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Optional (nur benötigt für --mode vectordb)
from langchain.schema import Document  # type: ignore

# Embeddings: bevorzugt aus langchain_openai
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings  # fallback alte Version

from langchain_community.vectorstores import Chroma  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

###############################################################################
# Regex & XML-Namespace
###############################################################################
REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")  # Anforderungen
MODULE_RE = re.compile(r"^([A-Z]+(?:\.\d+)+)\s+")  # Bausteine
CHAPTER_RE = re.compile(r"^[A-Z]{2,5}\b")  # Kapitel-Kürzel
NS = {"db": "http://docbook.org/ns/docbook"}  # DocBook-NS


###############################################################################
# Hilfsfunktionen
###############################################################################

def _text_from_paras(el: ET.Element) -> str:
    paras = ["".join(p.itertext()).strip() for p in el.findall(".//db:para", NS)]
    return "\n\n".join(filter(None, paras))


def _find_subsection(parent: ET.Element, title: str) -> Optional[ET.Element]:
    for sec in parent.findall("db:section", NS):
        t = sec.find("db:title", NS)
        if t is not None and t.text and t.text.strip() == title:
            return sec
    return None


###############################################################################
# Struktur extrahieren
###############################################################################

def extract_structure(xml_path: str) -> List[Dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chapters: List[Dict[str, Any]] = []

    for chap in root.findall("db:chapter", NS):
        ct = chap.find("db:title", NS)
        if ct is None or not ct.text:
            continue
        title = ct.text.strip()
        if not CHAPTER_RE.match(title):
            continue
        chap_id = title.split()[0]
        chapter = {"chapter_id": chap_id, "chapter_title": title, "modules": []}

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

            description = _text_from_paras(_find_subsection(mod, "Beschreibung") or mod)
            goal = _text_from_paras(_find_subsection(mod, "Zielsetzung") or mod)
            scope = _text_from_paras(_find_subsection(mod, "Abgrenzung und Modellierung") or mod)

            # Gefährdungen sammeln
            threats = []
            th_sec = _find_subsection(mod, "Gefährdungslage")
            if th_sec:
                for t in th_sec.findall("db:section", NS):
                    t_title = t.find("db:title", NS)
                    if t_title is None or not t_title.text:
                        continue
                    threats.append({"title": t_title.text.strip(), "text": _text_from_paras(t)})

            # Anforderungen sammeln
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
                    level = lvl_m.group(1) if lvl_m else "?"
                    roles_m = re.search(r"\[(.+?)\]", full)
                    roles = [x.strip() for x in roles_m.group(1).split(",")] if roles_m else []
                    requirements.append({
                        "requirement_id": rid,
                        "title": full,
                        "level": level,
                        "roles": roles,
                        "text": _text_from_paras(r)
                    })

            chapter["modules"].append({
                "module_id": mod_id,
                "module_title": mod_title,
                "description": description,
                "goal": goal,
                "scope": scope,
                "threats": threats,
                "requirements": requirements
            })

        if chapter["modules"]:
            chapters.append(chapter)
    return chapters


###############################################################################
# Documents für Vektordb mit Chunking
###############################################################################

def modules_to_documents(chapters: List[Dict[str, Any]]) -> List[Document]:
    """
    Flatten modules, threats and requirements into LangChain Documents,
    chunking long texts and embedding rich metadata including full titles.
    """
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for chap in chapters:
        # Base metadata for this chapter
        base_chap_meta = {
            "chapter_id":    chap["chapter_id"],
            "chapter_title": chap["chapter_title"],
        }

        for mod in chap["modules"]:
            # Extend base metadata with module info
            base_meta = {
                **base_chap_meta,
                "module_id":    mod["module_id"],
                "module_title": mod["module_title"],
            }

            # --- Module narrative chunks ---
            full_mod_text = "\n\n".join([
                f"Beschreibung:\n{mod['description']}",
                f"Zielsetzung:\n{mod['goal']}",
                f"Abgrenzung:\n{mod['scope']}"
            ])
            for i, chunk in enumerate(splitter.split_text(full_mod_text)):
                meta = {
                    **base_meta,
                    "type":        "module",
                    "chunk_index": i,
                }
                docs.append(Document(page_content=chunk, metadata=meta))

            # --- Threat chunks ---
            for thr in mod.get("threats", []):
                thr_text = f"{thr['title']}\n\n{thr['text']}"
                for i, chunk in enumerate(splitter.split_text(thr_text)):
                    meta = {
                        **base_meta,
                        "type":         "threat",
                        "threat_title": thr["title"],
                        "chunk_index":  i,
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

            # --- Requirement chunks ---
            for req in mod.get("requirements", []):
                full_req_text = f"{req['title']}\n\n{req['text']}"
                for i, chunk in enumerate(splitter.split_text(full_req_text)):
                    meta = {
                        **base_meta,
                        "type":               "requirement",
                        "requirement_id":     req["requirement_id"],
                        "requirement_title":  req["title"],           # vollständiger Titel
                        "level":              req["level"],
                        "roles":              ", ".join(req["roles"]),
                        "chunk_index":        i,
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))

    return docs


###############################################################################
# CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="IT-Grundschutz XML → JSON oder VectorDB")
    parser.add_argument("xml", help="Pfad zur XML-Datei")
    parser.add_argument("--mode", choices=["json", "vectordb"], default="json")
    parser.add_argument(
        "--output",
        help="JSON-Ausgabedatei (json) oder DB-Verzeichnis (vectordb)"
    )
    args = parser.parse_args()

    # Standard-Pfade setzen
    if not args.output:
        args.output = (
            "resources/requirements.json" if args.mode == "json" else "db"
        )

    chapters = extract_structure(args.xml)
    bc = sum(len(c['modules']) for c in chapters)
    rc = sum(len(m['requirements']) for c in chapters for m in c['modules'])

    if args.mode == 'json':
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON: {bc} Bausteine, {rc} Anforderungen in {len(chapters)} Kapiteln → {args.output}")
    else:
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('Missing OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(openai_api_key=key)
        docs = modules_to_documents(chapters)
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=args.output)
        print(f"✅ Chroma: {len(docs)} Dokumente → {args.output}")


if __name__ == '__main__':
    main()
