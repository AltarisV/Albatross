import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Optional (nur benötigt für --mode vectordb)
try:
    from langchain.schema import Document  # type: ignore
    from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
    from langchain_community.vectorstores import Chroma  # type: ignore
except ImportError:  # pragma: no cover
    Document = Any  # type: ignore

load_dotenv()

###############################################################################
# Regex & XML‑Namespace
###############################################################################
REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")  # APP.1.2.A3 …
MODULE_RE = re.compile(r"^([A-Z]+(?:\.\d+)+)\s+")  # SYS.2.4 …
CHAPTER_RE = re.compile(r"^[A-Z]{2,5}\b")  # SYS, APP …
NS = {"db": "http://docbook.org/ns/docbook"}  # DocBook 5 NS


###############################################################################
# Hilfsfunktionen
###############################################################################

def _text_from_paras(el: ET.Element) -> str:
    """Alle <para>-Nachfahren zu einem Fließtext zusammenfassen."""
    paras = ["".join(p.itertext()).strip() for p in el.findall(".//db:para", NS)]
    return "\n\n".join(filter(None, paras))


def _find_subsection(parent: ET.Element, title_query: str) -> Optional[ET.Element]:
    """Direkte Kind‑<section> mit passendem <title> finden."""
    for sec in parent.findall("db:section", NS):
        title_el = sec.find("db:title", NS)
        if title_el is not None and title_el.text and title_el.text.strip() == title_query:
            return sec
    return None


###############################################################################
# Kern‑Parsing
###############################################################################

def extract_structure(xml_path: str) -> List[Dict[str, Any]]:
    """XML des IT‑Grundschutz‑Kompendiums hierarchisch auslesen."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    chapters: List[Dict[str, Any]] = []

    for chap in root.findall("db:chapter", NS):
        chap_title_el = chap.find("db:title", NS)
        if chap_title_el is None or not chap_title_el.text:
            continue
        chap_title = chap_title_el.text.strip()
        if not CHAPTER_RE.match(chap_title):
            continue  # Meta‑Kapitel überspringen (Vorwort, Glossar …)

        chap_id = chap_title.split()[0]  # z. B. "SYS"
        chapter_dict: Dict[str, Any] = {
            "chapter_id": chap_id,
            "chapter_title": chap_title,
            "modules": []
        }

        # Innerste Sections = Bausteine
        for mod in chap.findall("db:section", NS):
            mod_title_el = mod.find("db:title", NS)
            if mod_title_el is None or not mod_title_el.text:
                continue
            mod_title_raw = mod_title_el.text.strip()
            m = MODULE_RE.match(mod_title_raw)
            if not m:
                continue  # kein Baustein

            module_id = m.group(1)  # "SYS.2.4"
            module_title = mod_title_raw[len(module_id):].strip()

            # Narrative Teile
            description = _text_from_paras(_find_subsection(mod, "Beschreibung") or mod)
            goal = _text_from_paras(_find_subsection(mod, "Zielsetzung") or mod)
            scope = _text_from_paras(_find_subsection(mod, "Abgrenzung und Modellierung") or mod)

            # Gefährdungen
            threats: List[Dict[str, str]] = []
            threats_sec = _find_subsection(mod, "Gefährdungslage")
            if threats_sec is not None:
                for t in threats_sec.findall("db:section", NS):
                    tt = t.find("db:title", NS)
                    if tt is None or not tt.text:
                        continue
                    threats.append({
                        "title": tt.text.strip(),
                        "text": _text_from_paras(t)
                    })

            # Anforderungen – jetzt rekursiv & namespace‑sicher
            requirements: List[Dict[str, Any]] = []
            req_root = _find_subsection(mod, "Anforderungen")
            if req_root is not None:
                for req in req_root.findall(".//db:section", NS):
                    title_el = req.find("db:title", NS)
                    if title_el is None or not title_el.text:
                        continue
                    title_full = title_el.text.strip()
                    if not REQ_RE.match(title_full):
                        continue  # Überschriften wie "Basis‑Anforderungen" o. Ä.

                    req_id = title_full.split()[0]
                    level_match = re.search(r"\((B|S|H)\)", title_full)
                    level = level_match.group(1) if level_match else "?"
                    roles_match = re.search(r"\[(.+?)\]", title_full)
                    roles = [r.strip() for r in roles_match.group(1).split(',')] if roles_match else []

                    requirements.append({
                        "requirement_id": req_id,
                        "title": title_full,
                        "level": level,
                        "roles": roles,
                        "text": _text_from_paras(req)
                    })

            chapter_dict["modules"].append({
                "module_id": module_id,
                "module_title": module_title,
                "description": description,
                "goal": goal,
                "scope": scope,
                "threats": threats,
                "requirements": requirements
            })

        if chapter_dict["modules"]:
            chapters.append(chapter_dict)

    return chapters


###############################################################################
# Vektor‑DB (optional)
###############################################################################

def modules_to_documents(chapters: List[Dict[str, Any]]) -> List[Document]:  # type: ignore
    """
    Wandelt die hierarchische Kapitel-Liste in flache LangChain-Documents um.

    • pro Baustein  ->  1 Document  (type="module")
    • pro Gefährdung -> 1 Document  (type="threat")
    • pro Anforderung -> 1 Document  (type="requirement")
    """
    docs: List[Document] = []

    for chap in chapters:
        for mod in chap["modules"]:
            meta_base = {
                "chapter_id": chap["chapter_id"],
                "module_id": mod["module_id"],
            }

            # ──────────────────────────── Modul-Narrativ ───────────────────────────
            narrativa = "\n\n".join([
                f"Beschreibung:\n{mod['description']}",
                f"Zielsetzung:\n{mod['goal']}",
                f"Abgrenzung:\n{mod['scope']}",
            ])
            docs.append(
                Document(
                    page_content=f"{mod['module_id']} {mod['module_title']}\n\n{narrativa}",
                    metadata={**meta_base, "type": "module"},
                )
            )

            # ───────────────────────────── Gefährdungen ────────────────────────────
            for thr in mod["threats"]:
                docs.append(
                    Document(
                        page_content=f"{mod['module_id']} THREAT: {thr['title']}\n\n{thr['text']}",
                        metadata={**meta_base,
                                  "type": "threat",
                                  "threat_title": thr["title"]},
                    )
                )

            # ───────────────────────────── Anforderungen ───────────────────────────
            for req in mod["requirements"]:
                docs.append(
                    Document(
                        page_content=f"{req['title']}\n\n{req['text']}",
                        metadata={**meta_base,
                                  "type": "requirement",
                                  "requirement_id": req["requirement_id"],
                                  "level": req.get("level"),
                                  "roles": req.get("roles", [])},
                    )
                )

    return docs


###############################################################################
# CLI
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description="IT‑Grundschutz XML in hierarchisches JSON oder Chroma laden")
    ap.add_argument("xml", help="Pfad zur grundschutz_2023.xml")
    ap.add_argument("--mode", choices=["json", "vectordb"], default="json")
    ap.add_argument("--output", default="out.json", help="JSON‑Datei oder Directory für Chroma")
    args = ap.parse_args()

    chapters = extract_structure(args.xml)
    baustein_count = sum(len(c["modules"]) for c in chapters)
    req_count = sum(len(m["requirements"]) for c in chapters for m in c["modules"])

    if args.mode == "json":
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(chapters, fh, ensure_ascii=False, indent=2)
        print(f"✅ {baustein_count} Bausteine, {req_count} Anforderungen in {len(chapters)} Kapiteln → {args.output}")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY fehlt für vectordb‑Modus")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        docs = modules_to_documents(chapters)
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=args.output)
        vectordb.persist()
        print(f"✅ {len(docs)} Dokumente (Module + Anforderungen) nach Chroma → {args.output}")


if __name__ == "__main__":
    main()
