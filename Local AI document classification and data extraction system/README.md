# Local AI Document Intelligence System

A fully offline, open-source document processing pipeline that:

- **Ingests** PDFs from a folder
- **Classifies** each document (Invoice / Resume / Utility Bill / Other / Unclassifiable)
- **Extracts** structured fields per document type
- **Semantic search** over all documents (no internet required)

---

## Installation

**Python 3.8+ required.**

```bash
pip install pypdf sentence-transformers faiss-cpu scikit-learn numpy
```

**Optional (PDF extraction fallbacks):**
```bash
pip install pdfminer.six pdfplumber
```

> **Offline mode:** If HuggingFace models are not cached, the system automatically falls back to TF-IDF + BM25 keyword search — no internet required at all.

---

## How to Run

### Process all documents and save output.json
```bash
python main.py --docs ./documents
```

### Process + run a search query
```bash
python main.py --docs ./documents --search "find documents mentioning payments due"
```

### Interactive search shell
```bash
python main.py --docs ./documents --interactive
```

### All options
```bash
python main.py --help

Options:
  --docs         Path to folder containing PDF files (default: ./documents)
  --output       Output JSON path (default: ./output/output.json)
  --search       Run a single semantic search query
  --interactive  Launch interactive search shell
  --top-k        Number of search results to return (default: 5)
```

---

## Output Format

`output/output.json` contains structured data for every document:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "1001",
    "date": "2025-06-16",
    "company": "Pioneer Ltd",
    "total_amount": 2073.0
  },
  "resume_3.pdf": {
    "class": "Resume",
    "name": "Ali Khan",
    "email": "ali.khan@example.com",
    "phone": "+1-555-980-6266",
    "experience_years": 4
  },
  "utilitybill_1.pdf": {
    "class": "Utility Bill",
    "account_number": "ACC-49575",
    "date": "2025-05-24",
    "usage_kwh": 406.0,
    "amount_due": 193.0
  },
  "other_1.pdf": {
    "class": "Other"
  }
}
```

---

## Project Structure

```
doc_ai/
├── main.py               ← CLI entry point
├── src/
│   ├── processor.py      ← PDF extraction, classification, field extraction
│   └── retrieval.py      ← Embedding index + semantic search
├── documents/            ← Put your PDF files here
├── output/
│   └── output.json       ← Generated results
└── README.md
```

---

## Libraries & Methods

| Component | Library | Method |
|---|---|---|
| PDF Text Extraction | `pypdf`, `pdfminer.six`, `pdfplumber` | Multi-fallback chain |
| Classification | Pure Python | Keyword scoring (regex patterns per category) |
| Field Extraction | Pure Python | Regex pattern matching per document type |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Dense vector encoding |
| Vector Index | `faiss-cpu` | Inner-product (cosine) IndexFlatIP |
| Offline Fallback | `scikit-learn` TfidfVectorizer + BM25 | Keyword overlap scoring |

### Classification Logic

Each document is scored against keyword pattern sets for each category (Invoice, Resume, Utility Bill). The category with the highest match count wins. Ties or zero-match documents are labelled `Other`.

### Extraction Logic

Regex patterns tuned per document type extract fields like invoice numbers, dates, emails, phone numbers, kWh usage, etc. Multiple fallback patterns are tried for each field.

### Retrieval Logic

- **With internet (first run):** Downloads `all-MiniLM-L6-v2` (22MB) once, cached locally. Uses FAISS cosine similarity.
- **Offline / cached:** Uses cached model automatically.
- **No model available:** Falls back to TF-IDF vectorization + BM25 keyword overlap — fully local, no downloads.

---

## Adding More Documents

Just drop additional `.pdf` files into the `documents/` folder and re-run. The system auto-discovers all PDFs.

---

## Technical Rules Compliance

- ✅ No paid or hosted AI APIs (OpenAI, Claude, Gemini, etc.)
- ✅ All processing runs locally
- ✅ Open-source libraries only
- ✅ Works fully offline (fallback to TF-IDF/BM25)
- ✅ CLI interface

