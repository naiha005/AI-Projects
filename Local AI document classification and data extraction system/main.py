#!/usr/bin/env python3
"""
Local AI Document Intelligence System
======================================
CLI entry point.

Usage:
    python main.py --docs ./documents                    # Process + extract (first time)
    python main.py --docs ./documents --search "query"   # Search only (uses cache)
    python main.py --docs ./documents --interactive      # Interactive search shell
    python main.py --docs ./documents --reprocess        # Force reprocess all docs
"""

import os
import sys
import json
import argparse
import logging
import hashlib
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processor import extract_text_from_pdf, classify_document, process_documents
from src.retrieval import RetrievalEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

CACHE_FILE = ".doc_cache.pkl"


# ─── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _folder_fingerprint(folder: Path) -> str:
    """Hash of all PDF filenames + sizes to detect any changes in the folder."""
    parts = []
    for p in sorted(folder.glob("*.pdf")):
        parts.append(f"{p.name}:{p.stat().st_size}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_cache(docs_folder: Path):
    cache_path = docs_folder / CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # Invalidate if folder contents changed
        if cache.get("fingerprint") != _folder_fingerprint(docs_folder):
            return None
        return cache
    except Exception:
        return None


def save_cache(docs_folder: Path, results: dict, doc_texts: dict):
    cache_path = docs_folder / CACHE_FILE
    cache = {
        "fingerprint": _folder_fingerprint(docs_folder),
        "results": results,
        "doc_texts": doc_texts,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


# ─── PIPELINE ──────────────────────────────────────────────────────────────────

def build_enriched_texts(results: dict, doc_texts: dict) -> dict:
    """
    Enrich each document's search text with its extracted fields so that
    queries like '4 years experience' or 'high usage' find the right docs.
    """
    enriched = {}
    for fname, fields in results.items():
        base = doc_texts.get(fname, "")
        cls = fields.get("class", "")

        extras = [cls]  # always include class name

        if cls == "Invoice":
            extras += [
                fields.get("company") or "",
                f"invoice number {fields.get('invoice_number') or ''}",
                f"date {fields.get('date') or ''}",
                f"total amount {fields.get('total_amount') or ''} dollars",
            ]
        elif cls == "Resume":
            extras += [
                fields.get("name") or "",
                f"{fields.get('experience_years') or ''} years experience",
                f"email {fields.get('email') or ''}",
            ]
        elif cls == "Utility Bill":
            extras += [
                f"usage {fields.get('usage_kwh') or ''} kwh kilowatt electricity",
                f"amount due {fields.get('amount_due') or ''} dollars",
                f"account {fields.get('account_number') or ''}",
                f"billing date {fields.get('date') or ''}",
            ]

        enriched[fname] = base + "\n" + " ".join(str(e) for e in extras if e)

    return enriched


def load_and_index(docs_folder: str, force_reprocess: bool = False):
    folder = Path(docs_folder)
    cache = None if force_reprocess else load_cache(folder)

    if cache:
        print("\n  ✓ Using cached results (folder unchanged). Use --reprocess to force re-extraction.\n")
        results  = cache["results"]
        doc_texts = cache["doc_texts"]
    else:
        results = process_documents(docs_folder)
        doc_texts = {}
        for fname in results:
            text = extract_text_from_pdf(str(folder / fname))
            if text:
                doc_texts[fname] = text
        save_cache(folder, results, doc_texts)

    enriched_texts = build_enriched_texts(results, doc_texts)

    print("  Setting up retrieval engine...")
    engine = RetrievalEngine()
    engine.build_index(enriched_texts, results)

    return results, engine


# ─── OUTPUT HELPERS ────────────────────────────────────────────────────────────

def run_search(engine: RetrievalEngine, query: str, top_k: int = 5):
    hits = engine.search(query, top_k=top_k)

    print(f"\n  Query : \"{query}\"")
    print(f"  {'─'*55}")

    if not hits:
        print("  No results found.\n")
        return

    for rank, (fname, score, doc_class) in enumerate(hits, 1):
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  {rank}. [{doc_class:15s}] {fname}")
        print(f"     Score: {score:.4f}  {bar}")
    print()


def interactive_shell(engine: RetrievalEngine):
    print("\n" + "="*60)
    print("  Interactive Search  (type 'quit' to exit)")
    print("="*60)
    while True:
        try:
            query = input("\n  Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            run_search(engine, query)


def print_summary(results: dict):
    from collections import Counter
    counts = Counter(v["class"] for v in results.values())
    print("="*60)
    print("  CLASSIFICATION SUMMARY")
    print("="*60)
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:<20} {cnt} document(s)")
    print(f"\n  Total: {len(results)} documents")
    print("="*60 + "\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local AI Document Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--docs",        default="./documents", help="Folder with PDF documents")
    parser.add_argument("--output",      default="./output/output.json", help="Output JSON path")
    parser.add_argument("--search",      type=str, help="Semantic search query")
    parser.add_argument("--interactive", action="store_true", help="Interactive search shell")
    parser.add_argument("--top-k",       type=int, default=5, help="Number of search results")
    parser.add_argument("--reprocess",   action="store_true", help="Force re-extract all documents")
    args = parser.parse_args()

    if not Path(args.docs).exists():
        print(f"ERROR: Documents folder not found: {args.docs}")
        sys.exit(1)

    results, engine = load_and_index(args.docs, force_reprocess=args.reprocess)

    # Save output.json
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ output.json saved → {out_path}\n")

    print_summary(results)

    if args.search:
        run_search(engine, args.search, top_k=args.top_k)

    if args.interactive:
        interactive_shell(engine)


if __name__ == "__main__":
    main()
