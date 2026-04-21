import argparse
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from bed_names import extract_beds

ROOT = Path(__file__).parent
FIELD_LOG = ROOT / "docs" / "field_log.md"
XLSX = ROOT / "docs" / "farm_records.xlsx"
CHROMA_DIR = ROOT / "chroma_db"
SQLITE_DB = ROOT / "farm.db"

COLLECTION_NAME = "field_log"
EMBED_MODEL = "nomic-embed-long"
MAX_CHUNK_CHARS = 5_500

_ENTRY_RE = re.compile(r"(?=^\*\*[A-Za-z]+ \d{1,2}/\d{1,2}/\d{2}\*\*)", re.MULTILINE)


def _sub_chunk(chunk: str, heading: str, n: int) -> list[str]:
    """Split a month chunk into n roughly equal parts, repeating the ## heading in each."""
    body = chunk[len(heading):].lstrip("\n")
    entries = [e for e in _ENTRY_RE.split(body) if e.strip()]
    base, extra = divmod(len(entries), n)
    groups, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        groups.append(entries[start:start + size])
        start += size
    return [heading + "\n\n" + "".join(g).strip() for g in groups if g]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest farm docs into Chroma + SQLite.")
    p.add_argument("--force", action="store_true", help="Re-ingest even if outputs already exist.")
    return p.parse_args()


def _chroma_has_data() -> bool:
    try:
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        )
        return vs._collection.count() > 0
    except Exception:
        return False


def ingest_field_log(force: bool) -> int | None:
    if not force and _chroma_has_data():
        print("Chroma: collection already populated — skipping (--force to re-ingest)")
        return None

    text = FIELD_LOG.read_text(encoding="utf-8")

    # Split at each ## Month Year heading; lookahead preserves heading at chunk start
    raw_chunks = re.split(r"(?=^## [A-Z][a-z]+ \d{4})", text, flags=re.MULTILINE)
    chunks = [c.strip() for c in raw_chunks if re.match(r"^## [A-Z][a-z]+ \d{4}", c.strip())]

    docs, ids = [], []
    for chunk in chunks:
        m = re.match(r"^## ([A-Z][a-z]+) (\d{4})", chunk)
        if not m:
            continue
        month_str, year_str = m.group(1), m.group(2)
        if len(chunk) > MAX_CHUNK_CHARS:
            n = math.ceil(len(chunk) / MAX_CHUNK_CHARS)
            print(f"  Splitting {month_str} {year_str} ({len(chunk):,} chars) into {n} parts")
            sub_chunks = _sub_chunk(chunk, m.group(0), n)
            total_parts = len(sub_chunks)
            for part, sub in enumerate(sub_chunks, 1):
                beds = extract_beds(sub)
                ids.append(f"{year_str}-{month_str}-p{part}")
                docs.append(Document(
                    page_content=sub,
                    metadata={
                        "year": int(year_str),
                        "month": month_str,
                        "part": part,
                        "total_parts": total_parts,
                        "mentioned_beds": ",".join(beds),
                    },
                ))
        else:
            beds = extract_beds(chunk)
            ids.append(f"{year_str}-{month_str}")
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "year": int(year_str),
                    "month": month_str,
                    "part": 1,
                    "total_parts": 1,
                    "mentioned_beds": ",".join(beds),
                },
            ))

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
    )
    if force:
        vs.delete_collection()
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        )

    print(f"Embedding {len(docs)} chunks...")
    vs.add_documents(docs, ids=ids)
    print("Done.")

    return len(docs)


def ingest_xlsx(force: bool) -> list[tuple[str, int]] | None:
    if not force and SQLITE_DB.exists():
        print("SQLite: farm.db already exists — skipping (--force to re-ingest)")
        return None

    table_summary = []
    xls = pd.ExcelFile(XLSX)
    with sqlite3.connect(SQLITE_DB) as conn:
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            table_name = sheet.lower()
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            table_summary.append((table_name, len(df)))
    return table_summary


def main() -> None:
    args = parse_args()

    print("=== field_log.md → Chroma ===")
    n_chunks = ingest_field_log(args.force)

    print("\n=== farm_records.xlsx → SQLite ===")
    tables = ingest_xlsx(args.force)

    print("\n=== Summary ===")
    if n_chunks is not None:
        print(f"  Chroma : {n_chunks} month chunks embedded")
    if tables is not None:
        print(f"  SQLite : {len(tables)} tables created")
        for name, rows in tables:
            print(f"    {name}: {rows:,} rows")


if __name__ == "__main__":
    main()
