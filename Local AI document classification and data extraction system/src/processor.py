"""
Document Processor: Ingests PDFs, classifies them, and extracts structured data.
Uses rule-based classification + keyword matching (no paid APIs).
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

# PDF extraction with fallback chain
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using multiple fallback methods."""
    text = ""

    # Method 1: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            logging.debug(f"pypdf succeeded for {pdf_path}")
            return text
    except Exception as e:
        logging.warning(f"pypdf failed for {pdf_path}: {e}")

    # Method 2: pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(pdf_path).strip()
        if text:
            logging.debug(f"pdfminer succeeded for {pdf_path}")
            return text
    except Exception as e:
        logging.warning(f"pdfminer failed for {pdf_path}: {e}")

    # Method 3: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
        if text:
            logging.debug(f"pdfplumber succeeded for {pdf_path}")
            return text
    except Exception as e:
        logging.warning(f"pdfplumber failed for {pdf_path}: {e}")

    logging.error(f"All PDF extraction methods failed for {pdf_path}")
    return ""


# ─── CLASSIFICATION ────────────────────────────────────────────────────────────

INVOICE_KEYWORDS = [
    r'\binvoice\b', r'\binv[\-#]?\d+', r'total amount', r'amount due',
    r'\bbill to\b', r'\bpayment due\b', r'\bsubtotal\b', r'\btax\b',
    r'thank you for your business'
]
RESUME_KEYWORDS = [
    r'\bresume\b', r'\bcurriculum vitae\b', r'\bcv\b', r'\bexperience\b',
    r'\beducation\b', r'\bskills\b', r'\bsummary\b', r'\bwork history\b',
    r'\blinkedin\b', r'\breferences\b', r'\bprofessional\b'
]
UTILITY_KEYWORDS = [
    r'\butility\b', r'\belectric\b', r'\bgas\b', r'\bwater\b',
    r'\bkwh\b', r'\bkilowatt\b', r'\baccount number\b', r'\bbilling date\b',
    r'\busage\b', r'\bprovider\b', r'\bmeter\b', r'\btherms\b'
]

def score_keywords(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))

def classify_document(text: str) -> str:
    """Classify document into Invoice/Resume/Utility Bill/Other/Unclassifiable."""
    if not text or len(text.strip()) < 20:
        return "Unclassifiable"

    scores = {
        "Invoice":      score_keywords(text, INVOICE_KEYWORDS),
        "Resume":       score_keywords(text, RESUME_KEYWORDS),
        "Utility Bill": score_keywords(text, UTILITY_KEYWORDS),
    }
    best_class, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score == 0:
        return "Other"
    # Require a minimum confidence gap to avoid ambiguity
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1] and best_score < 3:
        return "Other"
    return best_class


# ─── EXTRACTION ────────────────────────────────────────────────────────────────

def _find(pattern: str, text: str, group: int = 1, flags=re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None

def _find_float(pattern: str, text: str) -> Optional[float]:
    val = _find(pattern, text)
    if val:
        val = re.sub(r'[,$]', '', val)
        try:
            return float(val)
        except ValueError:
            return None
    return None

def _find_int(pattern: str, text: str) -> Optional[int]:
    val = _find(pattern, text)
    if val:
        try:
            return int(re.sub(r'[^\d]', '', val))
        except ValueError:
            return None
    return None


def extract_invoice(text: str) -> dict:
    return {
        "invoice_number": (
            _find(r'invoice\s*[#:]?\s*(\w+[-/]?\w+)', text) or
            _find(r'inv[-#]?\s*(\w+)', text) or
            _find(r'#\s*(\d+)', text)
        ),
        "date": (
            _find(r'date[:\s]+(\d{4}[-/]\d{2}[-/]\d{2})', text) or
            _find(r'date[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text) or
            _find(r'(\d{4}-\d{2}-\d{2})', text)
        ),
        "company": (
            _find(r'company[:\s]+(.+)', text) or
            _find(r'from[:\s]+(.+)', text) or
            _find(r'bill(?:ed)?\s+(?:to|from)[:\s]+(.+)', text)
        ),
        "total_amount": (
            _find_float(r'total\s*amount[:\s]*\$?([\d,]+\.?\d*)', text) or
            _find_float(r'total[:\s]*\$?([\d,]+\.?\d*)', text) or
            _find_float(r'\$\s*([\d,]+\.?\d+)', text)
        ),
    }


def extract_resume(text: str) -> dict:
    # Name: first non-empty line that looks like a name (2–4 words, no digits)
    name = None
    for line in text.split('\n'):
        line = line.strip()
        if line and re.match(r'^[A-Za-z][a-zA-Z\s\-\.]{3,50}$', line) and len(line.split()) <= 5:
            name = line
            break

    return {
        "name": name or _find(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text, flags=re.MULTILINE),
        "email": _find(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text, group=0),
        "phone": _find(
            r'(\+?[\d\s\-().]{7,20})',
            re.search(r'(?:phone|tel|mobile|cell)[:\s]*([\+\d\s\-().]{7,25})', text, re.IGNORECASE) and
            re.search(r'(?:phone|tel|mobile|cell)[:\s]*([\+\d\s\-().]{7,25})', text, re.IGNORECASE).group(0) or text
        ) or _find(r'(?:phone|tel|mobile|cell)[:\s]*([\+\d\s\-().]{7,25})', text),
        "experience_years": (
            _find_int(r'experience[:\s]+(\d+)\s*years?', text) or
            _find_int(r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', text) or
            _find_int(r'(\d+)\s*years?', text)
        ),
    }


def extract_utility_bill(text: str) -> dict:
    return {
        "account_number": (
            _find(r'account\s*(?:number|#|no\.?)[:\s]*([\w\-]+)', text) or
            _find(r'acc[-\s]*([\w\d]+)', text)
        ),
        "date": (
            _find(r'billing\s*date[:\s]+(\d{4}[-/]\d{2}[-/]\d{2})', text) or
            _find(r'date[:\s]+(\d{4}[-/]\d{2}[-/]\d{2})', text) or
            _find(r'(\d{4}-\d{2}-\d{2})', text)
        ),
        "usage_kwh": (
            _find_float(r'usage[:\s]*([\d,]+\.?\d*)\s*kwh', text) or
            _find_float(r'([\d,]+\.?\d*)\s*kwh', text)
        ),
        "amount_due": (
            _find_float(r'amount\s*due[:\s]*\$?([\d,]+\.?\d*)', text) or
            _find_float(r'total\s*due[:\s]*\$?([\d,]+\.?\d*)', text) or
            _find_float(r'\$\s*([\d,]+\.?\d+)', text)
        ),
    }


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def process_documents(folder: str) -> dict:
    """Process all PDFs in a folder. Returns structured results dict."""
    results = {}
    folder = Path(folder)
    pdfs = sorted(folder.glob("*.pdf"))

    if not pdfs:
        logging.warning(f"No PDF files found in {folder}")
        return results

    print(f"\n{'='*60}")
    print(f"  Processing {len(pdfs)} documents from: {folder}")
    print(f"{'='*60}\n")

    for pdf_path in pdfs:
        fname = pdf_path.name
        print(f"  [{fname}]")

        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            print(f"    ⚠  Could not extract text")
            results[fname] = {"class": "Unclassifiable", "error": "text extraction failed"}
            continue

        doc_class = classify_document(text)
        print(f"    Class: {doc_class}")

        entry = {"class": doc_class}

        if doc_class == "Invoice":
            entry.update(extract_invoice(text))
        elif doc_class == "Resume":
            entry.update(extract_resume(text))
        elif doc_class == "Utility Bill":
            entry.update(extract_utility_bill(text))

        results[fname] = entry

    return results
