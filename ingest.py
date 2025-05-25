import os
import argparse
import json
import re
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Regex for requirement IDs e.g. APP.1.1.A2, IND.1.A23
REQ_RE = re.compile(r"^[A-Z]+(?:\.\d+)+\.A\d+\b")

# XML namespace for DocBook
NS = {'db': 'http://docbook.org/ns/docbook'}


def load_requirements_from_xml(xml_path: str) -> List[Document]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    docs: List[Document] = []

    # Iterate all <section> elements including nested
    for sec in root.findall('.//db:section', NS):
        title_el = sec.find('db:title', NS)
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()
        # only process requirement sections
        if not REQ_RE.match(title):
            continue

        # gather all <para> text under this section
        paras = []
        for para in sec.findall('db:para', NS):
            # accumulate child text and nested elements
            text = ''.join(para.itertext()).strip()
            if text:
                paras.append(text)

        # flatten to single text
        full_text = title + ("\n\n" + "\n\n".join(paras) if paras else "")

        # metadata: requirement_id and xml:id
        xml_id = sec.get('{http://www.w3.org/XML/1998/namespace}id') or sec.get('xml:id')
        meta = {
            'requirement_id': title.split()[0],
            'xml_id': xml_id
        }
        docs.append(Document(page_content=full_text, metadata=meta))

    return docs


def ingest_vectordb(docs: List[Document], persist_directory: str):
    api_key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Ingested {len(docs)} requirements into '{persist_directory}' using OpenAI Embeddings")


def ingest_json(docs: List[Document], output_path: str):
    out = []
    for doc in docs:
        out.append({
            'requirement_id': doc.metadata.get('requirement_id'),
            'xml_id': doc.metadata.get('xml_id'),
            'text': doc.page_content
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(docs)} requirements to JSON file '{output_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest requirements from IT-Grundschutz Kompendium XML into Chroma or JSON."
    )
    parser.add_argument(
        'xml_path', help='Path to the IT-Grundschutz Kompendium XML file.'
    )
    parser.add_argument(
        '--mode', choices=['vectordb', 'json'], default='vectordb',
        help="Choose 'vectordb' to store embeddings or 'json' to dump requirements to JSON."
    )
    parser.add_argument(
        '--output', default='db',
        help="For vectordb mode: directory to persist; for json mode: output JSON file path."
    )
    args = parser.parse_args()

    docs = load_requirements_from_xml(args.xml_path)
    print(f"📄 Extracted {len(docs)} requirements from '{args.xml_path}'.")

    if args.mode == 'vectordb':
        ingest_vectordb(docs, args.output)
    else:
        ingest_json(docs, args.output)


if __name__ == '__main__':
    main()
