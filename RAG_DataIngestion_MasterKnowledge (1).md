# RAG Data Ingestion — Master Knowledge Documentation
### Complete Deep-Dive: Concepts, Libraries, Techniques, Trade-offs & Research

> **Based on**: Your `Rag-architecture/` repository + research-backed expansion covering ML, Deep Learning, NLP & AI principles underlying every design decision.
> **Goal**: By the end of this document, you can make every ingestion decision with full understanding of *why*, not just *what*.

---

## Table of Contents

1. [The Foundation — Why Data Ingestion is the Hardest RAG Stage](#1-the-foundation)
2. [The Document Dataclass — Universal Pipeline Contract](#2-the-document-dataclass)
3. [PDF Parsing — The 3-Tier Cascade](#3-pdf-parsing)
4. [Text & Encoding — The chardet Layer](#4-text--encoding)
5. [HTML Parsing — BeautifulSoup vs the Alternatives](#5-html-parsing)
6. [Table Extraction — Camelot vs pdfplumber](#6-table-extraction)
7. [Image Handling — Tesseract OCR vs Vision API](#7-image-handling)
8. [Email Parsing — MIME & Multipart](#8-email-parsing)
9. [PPTX Parsing — Slides & Speaker Notes](#9-pptx-parsing)
10. [Unstructured.io — The Universal Fallback](#10-unstructuredio)
11. [Quality Pipeline — The 5 Steps Deep-Dived](#11-quality-pipeline)
    - 11.1 Noise Filtering
    - 11.2 Metadata Validation
    - 11.3 Deduplication (Hash → SimHash → Semantic)
    - 11.4 PII Detection with Presidio
    - 11.5 Freshness Tracking
12. [Orchestration Patterns — Sync vs Async](#12-orchestration-patterns)
13. [Critical Bugs in Your Codebase — Root Cause Analysis](#13-critical-bugs)
14. [Production Architecture Upgrade Path](#14-production-architecture)
15. [The Bigger RAG Picture — How Ingestion Quality Affects Downstream](#15-the-bigger-picture)

---

## 1. The Foundation

### What is RAG Data Ingestion in ML Terms?

RAG (Retrieval-Augmented Generation) connects LLMs to private knowledge. The ingestion stage is a **text normalization and feature extraction pipeline** — its job is to transform heterogeneous, noisy, format-specific binary data into **clean, consistent, machine-readable Unicode strings** with rich metadata.

Every subsequent stage — chunking, embedding, retrieval — is a pure NLP/ML operation that assumes clean text input. If ingestion is wrong, no amount of ML sophistication downstream can recover.

```
Raw Bytes (PDF/HTML/DOCX/EML)
       ↓  Format Parsing Layer
       ↓  Character Encoding Normalization
       ↓  Structural Extraction (text, tables, images)
       ↓  Quality Pipeline (noise, dedup, PII, freshness)
       ↓
Clean Document Objects → [Chunking] → [Embedding] → [Vector DB]
```

### Why Ingestion is Uniquely Hard

Most ML pipelines have clean, structured inputs (CSV files, image arrays). Ingestion deals with the **pre-structured chaos** of real-world documents:

| Problem Class | Root Cause | ML Consequence |
|---|---|---|
| Format diversity | 40+ file formats exist, each a different binary spec | No single parser handles all |
| Encoding entropy | 100+ character encodings in the wild | Garbled Unicode → garbage embeddings |
| Layout complexity | PDFs encode visual layout, not logical text | Wrong reading order → incoherent chunks |
| Semantic noise | Headers, footers, cookies, ads pollute content | Lowers retrieval precision |
| Near-duplicate content | Same guide in PDF + HTML + email attachment | Wastes vector DB space, hurts diversity |
| PII contamination | SSNs, emails in support tickets | Legal/compliance failure |
| Knowledge staleness | 2019 VPN guide still answers "current" questions | Wrong answers with high confidence |

---

## 2. The Document Dataclass

### Design Pattern: Bounded Context / Universal Contract

```python
@dataclass
class Document:
    content: str          # Cleaned text (UTF-8 normalized)
    metadata: dict        # Source, parser, timestamps, category
    tables: list[str]     # Tables as Markdown strings
    images: list[str]     # OCR/Vision descriptions
    doc_id: str           # Content hash (dedup key)
```

This is the **Adapter Pattern** from software engineering applied to ML pipelines. Every parser is an adapter that translates a different format into this single contract. The quality pipeline and all downstream stages only need to understand this one structure — they are completely decoupled from file formats.

**Why tables and images are separate lists, not merged into content:**

In chunking (the next stage), tables and images are semantically dense units. Merging them into the text body at ingestion time causes chunkers to split them mid-table (losing the header row context). By keeping them separate until the chunking stage decides how to handle them, you preserve their structural integrity.

**Why doc_id is content-based, not path-based:**

Same document at different paths (synced copy, email attachment) will get the same `doc_id`. This is **content-addressable storage** — the same principle used by Git. A path-based ID would fail to detect cross-location duplicates.

**The Bug: MD5 vs SHA-256**

Your codebase uses MD5 for `doc_id` in `loader.py` and SHA-256 in `deduplicator.py`. This is a critical inconsistency.

- **MD5** is a 128-bit hash. It has known collision vulnerabilities. A malicious or corrupted document could produce the same MD5 as a legitimate document, causing the legitimate document to be treated as a duplicate and dropped.
- **SHA-256** is a 256-bit hash (from the SHA-2 family). No practical collisions are known.

**Fix**: Standardize on SHA-256 everywhere:
```python
import hashlib
doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
```

---

## 3. PDF Parsing

### Why PDFs Are Special — The Deep Reason

A PDF is not a text document. It is a **page description language** (based on PostScript) — a set of drawing instructions that tell a renderer *where to place each glyph on a canvas*. There is no concept of "a sentence" or "a paragraph" in the file format itself.

The PDF specification (ISO 32000) defines objects like:
- **Text objects**: sequences of glyph codes + transformation matrices
- **Content streams**: instructions like "move to (x, y)", "set font F at size 12", "show text 'Hello'"
- **Character encoding tables**: mapping glyph codes → Unicode codepoints (often absent or incorrect in old PDFs)

A PDF parser's job is to reverse-engineer the *visual rendering* back into *logical text structure*. This is inherently lossy and ambiguous — especially for multi-column layouts, footnotes, and tables.

### The 3-Tier Cascade — Why Each Tier Exists

**Tier 1: PyMuPDF (fitz)**

PyMuPDF is Python bindings for **MuPDF**, a C library written by Artifex Software. MuPDF is one of the highest-quality PDF rendering engines in the world (it powers Adobe Reader on embedded devices).

Why PyMuPDF is Tier 1:
- It directly reads the PDF's character map (CMap) and ToUnicode tables to extract text
- It is the **fastest** option: benchmarks from the py-pdf benchmark suite show PyMuPDF at 0.1s average vs pdfminer at 0.8s+ for comparable documents
- A 2024 research study (Adhikari & Agarwal, arxiv.org/abs/2410.09871) comparing 10 PDF parsers across 6 document categories found PyMuPDF and pypdfium2 **generally outperformed all others** for text extraction quality
- For the majority of well-formed digital PDFs (born-digital, not scanned), PyMuPDF extracts near-perfect text

Why it fails on some PDFs:
- Scanned documents: no text layer exists at all — the page is one big image
- PDFs with corrupt or missing ToUnicode tables: glyph codes cannot be mapped to Unicode
- Complex 2-column academic papers: reading order detection can fail (reads left column, then right column interleaved)

**Tier 2: pdfplumber**

pdfplumber is built on top of `pdfminer.six`. It adds spatial analysis — it can reason about the *positions* of characters on the page, not just their code values.

Why pdfplumber is Tier 2:
- It offers coordinate-based text extraction, which helps with multi-column layouts
- It has dedicated table extraction using line geometry (gaps between columns)
- It is slower than PyMuPDF but produces better layout-aware output for complex documents
- The 50-character threshold in your code is a heuristic: if PyMuPDF extracted less than 50 characters from a page, it likely failed and pdfplumber should try

**Important**: pdfplumber opens the PDF a second time in your code. This is a double I/O operation. Fix: pass the already-open file object or cache the binary content.

**Tier 3: Tesseract OCR**

Tesseract is an open-source OCR engine, originally developed by HP and now maintained by Google. It is the last resort for pages where no text could be extracted by either of the above tiers — which typically means the page is an image (a scan).

How Tesseract works (the ML/DL part):
- The page is rasterized (rendered as pixels) at 300 DPI — the standard medical-imaging resolution for OCR
- Tesseract 4+ uses an **LSTM (Long Short-Term Memory) neural network** — a type of RNN — for sequence modeling. The LSTM reads a horizontal strip of the image and predicts the most likely character sequence
- Before LSTM (Tesseract 3), it used classical pattern matching which was far less accurate
- The 300 DPI setting in your code is critical: at 150 DPI, OCR accuracy degrades significantly because characters have too few pixels to resolve fine features

**Why DPI matters for OCR:**
At 300 DPI, a typical 12pt font character is ~50 pixels tall. The LSTM has enough pixel information to distinguish 'l', 'I', and '1'. At 150 DPI, that same character is only ~25 pixels tall — confusable characters merge and error rates spike.

**Library Comparison Summary:**

| Library | Underlying Tech | Speed | Best Document Type | Weakness |
|---|---|---|---|---|
| PyMuPDF (fitz) | MuPDF C library | Very fast (~0.1s/page) | Digital PDFs | Scanned, corrupt encodings |
| pdfplumber | pdfminer.six + spatial | Medium (~0.5s/page) | Multi-column, complex layout | Still fails on images |
| Tesseract | LSTM neural network | Slow (~2-5s/page) | Any image/scanned page | Low accuracy on small fonts, handwriting |
| pypdfium2 | PDFium (Chrome's engine) | Fastest (0.003s/page) | Simple digital PDFs | Less Python integration, no table support |
| Nougat (Meta) | Vision Transformer + text | Very slow (GPU needed) | Scientific/academic papers | Requires GPU, heavy model |

**Why not use Nougat (or similar AI-based parsers) as Tier 1?**

Nougat (from Meta) uses a Vision Transformer to convert PDF pages to Markdown. It achieves excellent results on scientific papers with equations. However:
- It requires GPU inference to be practical
- It is 10-100× slower than PyMuPDF on CPU
- For most enterprise documents (policies, manuals, emails), PyMuPDF is better or equal
- Reserve Nougat/AI-based parsers for specialized scientific/patent document corpora

---

## 4. Text & Encoding

### The Encoding Problem — Deep Explanation

Computers store text as numbers. The question is: which number represents which character? An **encoding** is the mapping. The most common encodings:

- **ASCII**: 7-bit, 128 characters (English only). Every system agrees on this.
- **Latin-1 (ISO-8859-1)**: 8-bit, 256 characters. Extended ASCII for Western European languages.
- **Windows-1252 (CP1252)**: Microsoft's superset of Latin-1. Used by Word documents saved on Windows machines.
- **UTF-8**: Variable-width (1-4 bytes). Encodes all Unicode characters. The modern standard.

**Why this breaks in practice:**

A file saved in Windows-1252 has no header declaring its encoding. If you try to decode it as UTF-8, byte sequences like `\x93` (Windows left quote) will fail — because `\x93` is not a valid UTF-8 start byte.

In enterprise environments: HR documents, old policy files, email exports — many are still Windows-1252 or Latin-1. Without detection, you get `UnicodeDecodeError` exceptions or worse, silently wrong characters (the "Mojibake" problem).

### chardet — How It Works

`chardet` is a port of Mozilla's Universal Charset Detector. It uses **statistical analysis of byte distributions** to infer the encoding:

1. **Probers**: It runs multiple encoding-specific probers in parallel (UTF-8 prober, Latin-1 prober, Windows-1252 prober, UTF-16 prober, etc.)
2. Each prober maintains a **confidence score** based on how well the byte sequence matches that encoding's statistical distribution
3. **Return value**: `{'encoding': 'UTF-8', 'confidence': 0.99, 'language': ''}`

This is a probabilistic, not deterministic, detection. The confidence score matters:
- `confidence > 0.95`: High confidence, use the detected encoding
- `confidence 0.7-0.95`: Possible, but validate
- `confidence < 0.7`: Uncertain — try multiple encodings with error handling

**chardet vs charset-normalizer:**

Your code uses `chardet`. The alternative `charset-normalizer` is newer and often recommended for RAG:

| | chardet | charset-normalizer |
|---|---|---|
| Algorithm | Mozilla-ported statistical probers | Mess ratio analysis + coherence scoring |
| Speed | Faster for small files | Slower but more accurate for mixed content |
| Accuracy | Good | Better on edge cases (mixed encodings) |
| Maintenance | Less actively maintained | Actively maintained; used by `requests` library |

For RAG pipelines processing thousands of diverse enterprise documents, `charset-normalizer` is the safer choice. It is already used by the `requests` library and is well-battle-tested.

**Correct usage pattern:**
```python
from charset_normalizer import from_bytes

result = from_bytes(raw_bytes).best()
if result:
    text = str(result)  # Decoded text
    encoding_used = result.encoding
```

---

## 5. HTML Parsing

### The Challenge: HTML is Not Paragraph Text

HTML mixes content with structure. A raw HTML string like:
```html
<html>
  <nav>Home | About | Contact</nav>
  <div class="cookie-banner">We use cookies...</div>
  <article>
    <h1>VPN Setup Guide</h1>
    <p>Step 1: Download the client...</p>
  </article>
  <footer>© 2024 Company. All rights reserved.</footer>
</html>
```

If you strip all tags naively, you get:
`Home | About | Contact We use cookies... VPN Setup Guide Step 1: Download the client... © 2024 Company. All rights reserved.`

The navigation, cookie banner, and footer are noise — they have no value for RAG retrieval and actively hurt it by adding irrelevant tokens to embedded chunks.

### BeautifulSoup — How It Works

BeautifulSoup is a parser-agnostic HTML/XML tree builder. It builds a DOM (Document Object Model) tree in Python, then lets you navigate and search it with CSS selectors, `.find()`, `.find_all()`, tag traversal, etc.

**Your code uses `html.parser` (Python's built-in) but lxml is already a dependency — this is a bug.**

BeautifulSoup supports multiple backends:
- `html.parser`: Pure Python, slower, lenient on malformed HTML
- `lxml`: C-based, faster (2-5×), handles malformed HTML better
- `html5lib`: Most lenient, slowest, parses exactly as browsers do

Since `lxml` is already installed, use it:
```python
soup = BeautifulSoup(html_content, "lxml")  # Not "html.parser"
```

**What your HTMLParser does (correctly):**
1. Removes `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>` tags (structural/noise)
2. Extracts text from the remaining nodes using `.get_text(separator=" ", strip=True)`
3. Cleans whitespace

**What trafilatura does (and why it's better for web content):**

`trafilatura` is a web text extraction library designed specifically for news articles and web pages. It uses multiple heuristics:
- **Block scoring**: Scores each text block by content density (characters per HTML tag) — content blocks have high density, navigation has low density
- **Fallback**: If scoring fails, falls back to `jusText` or `readability-lxml`
- **Boilerplate removal**: Identifies and removes repeated patterns across pages

For internal IT documents (your current use case), BeautifulSoup with manual tag removal is sufficient. For web-scraped content (knowledge bases from the internet), trafilatura is significantly better.

---

## 6. Table Extraction

### Why Tables Are Fundamentally Different From Text

In a PDF, a table exists as one of two visual representations:

**Lattice tables**: Have visible grid lines (borders). The table structure is encoded as drawing commands for horizontal/vertical lines. Parsers can detect the lines and reconstruct the cell grid.

**Stream tables**: Have no visible borders. Columns are aligned by whitespace only. Parsers must infer column boundaries from character x-position distributions. Far harder and error-prone.

### Why Tables Must Be Preserved for RAG

If a policy document has:
```
| Risk Level | Response Time | Escalation |
|------------|--------------|-----------|
| Critical   | 15 minutes   | CISO      |
| High       | 4 hours      | Manager   |
```

And you parse it as plain text:
`Risk Level Response Time Escalation Critical 15 minutes CISO High 4 hours Manager`

When a user asks "What is the response time for a Critical risk?", the LLM gets garbled input and may fail to extract the correct answer. The table structure carries the semantics.

**The solution: convert tables to Markdown.** LLMs are trained on massive amounts of Markdown-formatted data and understand the pipe table syntax natively.

### Camelot vs pdfplumber Tables

**Camelot** is a dedicated table extraction library for PDFs. It has two modes:
- `lattice`: Uses OpenCV to detect grid lines. Excellent for tables with visible borders. Works best for financial documents, formal reports.
- `stream`: Uses pdfminer under the hood with whitespace gap analysis. For borderless tables.

**pdfplumber Tables**:
- Uses spatial analysis of character positions
- Computes column boundaries from x-coordinate gaps
- Simpler API: `page.extract_table()` returns a nested list

**Research finding (arxiv.org/abs/2410.09871):** Camelot achieved the highest table detection score (0.72 Jaccard) for Tender documents. Table Transformer (a deep learning model from Microsoft) outperformed all rule-based tools for Financial, Patent, and Scientific categories. PyMuPDF showed the most consistent recall across all categories for rule-based tools.

**Practical recommendation:**
- Use Camelot for financial/formal documents with clear borders
- Use pdfplumber for general-purpose tables
- For highest accuracy on scientific/financial PDFs: Table Transformer (TATR) — but requires a ML model and is significantly slower

```
Your current approach: Camelot → pdfplumber fallback ✅ Correct
```

**Converting to Markdown — why this format specifically:**

```python
# Markdown table output
"| Column A | Column B |\n|---|---|\n| Value 1 | Value 2 |"
```

LLMs are trained on GitHub READMEs, Wikipedia, and documentation — all of which use Markdown tables. The `|---|` separator pattern is a strong indicator to the LLM that tabular data follows, activating the correct attention patterns for structured data processing.

---

## 7. Image Handling

### OCR vs Vision API — Two Different Approaches to Image Understanding

**Tesseract OCR (Rule-Based + LSTM):**

Tesseract's LSTM pipeline works as follows:
1. **Preprocessing**: Binarization (convert grayscale to black/white), deskewing, noise removal
2. **Line finding**: Detect horizontal text lines using projection profiles
3. **Word recognition**: For each text line, the LSTM reads a fixed-height strip of pixels left-to-right and outputs a probability distribution over characters at each step
4. **CTC decoding**: Connectionist Temporal Classification decodes the LSTM output sequence into characters, handling the many-to-one mapping (multiple input frames → one character)

**What Tesseract can extract**: Pure text from images — scanned documents, screenshots, whiteboards
**What Tesseract cannot understand**: Diagrams, charts, photographs with embedded meaning, architectural diagrams

**OpenAI Vision API (Multimodal LLM):**

GPT-4V and similar vision models use a **Vision Transformer (ViT)** as the image encoder, producing patch embeddings that are then processed by the LLM's transformer layers alongside text tokens.

What a Vision API can do that OCR cannot:
- Describe diagrams: "This is a network architecture diagram showing three servers connected to a load balancer"
- Understand charts: "The bar chart shows Q3 revenue was $2.1M, up 15% from Q2"
- Read handwritten notes
- Understand flowcharts and process diagrams

**When to use which:**

| Scenario | Use OCR | Use Vision API |
|---|---|---|
| Scanned document with text | ✅ | Overkill + expensive |
| Screenshot of a table | ✅ (with structure extraction) | Better for complex tables |
| Architecture diagram | ❌ (produces garbage) | ✅ |
| Chart/graph with data labels | Partial | ✅ |
| Handwritten notes | Poor accuracy | ✅ |
| High volume (1000s of images) | ✅ (free, fast) | Expensive at scale |

**Your code's approach**: Try Tesseract first. If Tesseract yields empty/short text, fall back to OpenAI Vision. This is the correct cost-aware cascade — Tesseract is free and fast, Vision API is paid and slower.

**The Critical Bug: New OpenAI client per image:**

```python
# ❌ Your current code (image_handler.py)
def describe_image(self, image):
    client = OpenAI()  # Created on EVERY call — loads config, creates connection
    response = client.chat.completions.create(...)
```

This creates a new TCP connection, loads API keys, and initializes the HTTP client for every single image. For a PDF with 50 embedded images, this runs 50 times.

```python
# ✅ Fix: Initialize once in __init__
def __init__(self, config):
    self._openai_client = OpenAI(api_key=config.openai_api_key)

def describe_image(self, image):
    response = self._openai_client.chat.completions.create(...)
```

---

## 8. Email Parsing

### MIME Structure — Why Emails Are Complex

An email is a MIME (Multipurpose Internet Mail Extensions) message. MIME was designed to allow non-ASCII content and binary attachments in email, which was originally ASCII-only.

A real email's raw structure:
```
From: sender@company.com
To: recipient@company.com
Subject: IT Policy Update
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="=_boundary_string_="

--=_boundary_string_=
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: quoted-printable

Please review the updated VPN policy attached.

--=_boundary_string_=
Content-Type: text/html; charset="utf-8"

<html>Please review the updated VPN policy attached.</html>

--=_boundary_string_=
Content-Type: application/pdf; name="vpn_policy_v3.pdf"
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="vpn_policy_v3.pdf"

JVBERi0xLjQKJeLjz9...  [base64-encoded PDF bytes]
```

Python's `email` module (used by your `email_parser.py`) handles MIME parsing via `email.message_from_bytes()` which decodes this structure into a tree of `Message` objects.

**Your code's critical gap**: It lists attachments but **doesn't parse them**. A support ticket `.eml` file might have a PDF attachment containing the actual detailed incident report. That content is invisible to your RAG system.

**Fix — recursive attachment parsing:**
```python
def _extract_attachments(self, msg):
    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()
        
        if content_type == "application/pdf":
            pdf_bytes = part.get_payload(decode=True)
            # Route to PDFParser
            yield self.pdf_parser.parse_bytes(pdf_bytes, source=filename)
        
        elif content_type.startswith("image/"):
            image_bytes = part.get_payload(decode=True)
            # Route to ImageHandler for OCR/Vision
            yield self.image_handler.describe_bytes(image_bytes)
```

**Content-Transfer-Encoding:**
MIME attachments can be encoded as:
- `base64`: Binary data encoded as ASCII (common for PDFs, images)
- `quoted-printable`: Text with special chars encoded as `=XX` hex sequences
- `7bit` / `8bit`: Plain text

Python's `email` module handles decoding automatically when you call `part.get_payload(decode=True)`.

---

## 9. PPTX Parsing

### Why Speaker Notes Are Critical for RAG

A PowerPoint presentation in a corporate setting often looks like:
- **Slide text**: "Q3 Revenue: $12.4M (+18% YoY)" — compact, context-free
- **Speaker notes**: "This number excludes the EMEA acquisition which closed November 2nd. The organic growth rate was actually 11%. See Appendix B for the breakdown by product line."

The speaker notes contain the *interpretation* and *context* that make the slide text meaningful. For RAG retrieval, a question like "What was the organic growth rate in Q3?" can only be answered from the notes, not the slide.

**python-pptx library**: Reads the OOXML format (.pptx = ZIP archive of XML files).

```python
from pptx import Presentation

prs = Presentation("deck.pptx")
for slide in prs.slides:
    # Slide text from all shape text frames
    slide_text = " ".join([
        shape.text_frame.text 
        for shape in slide.shapes 
        if shape.has_text_frame
    ])
    
    # Speaker notes — in the notes_slide XML
    notes_text = ""
    if slide.has_notes_slide:
        notes_text = slide.notes_slide.notes_text_frame.text
```

**What to do with tables in slides:**

Slides can contain tables too. python-pptx exposes `shape.table` for table shapes, and you should convert these to Markdown just as with PDF tables.

---

## 10. Unstructured.io

### Why a Universal Fallback Exists

No matter how many specialized parsers you build, there will always be file types you didn't anticipate:
- `.rtf` (Rich Text Format)
- `.odt` (OpenDocument Text, LibreOffice format)
- `.epub` (eBook format)
- `.xlsx` (Excel spreadsheets — treated as documents)
- Proprietary formats

The [Unstructured.io](https://github.com/Unstructured-IO/unstructured) library is built exactly for this purpose. It provides document processing for 30+ file types with a unified API:

```python
from unstructured.partition.auto import partition

elements = partition("document.rtf")
text = "\n".join([str(el) for el in elements])
```

**How Unstructured works internally:**
- It auto-detects the file type using `python-magic` (libmagic)
- Routes to format-specific processors (including calling Tesseract for image-heavy content)
- Returns `Element` objects with semantic types: `Title`, `NarrativeText`, `ListItem`, `Table`, `Image`

**The critical value**: Unstructured returns *semantically typed* elements, not just raw text. A `Title` element comes from a heading; a `Table` element has the table content. This semantic typing can be used in chunking to keep title-body pairs together.

**When to use Unstructured as primary (not just fallback):**
For RAG systems where document format diversity is extremely high and you don't need maximum extraction quality for any specific format, Unstructured as the primary parser simplifies your codebase significantly. The tradeoff: it is slower and less accurate for PDFs than the 3-tier cascade.

---

## 11. Quality Pipeline

### Why the Ordering Matters — A Causal Chain

The 5-step order is not arbitrary. Each step depends on the previous:

```
Noise Filter FIRST → so deduplication compares clean text (not formatting artifacts)
Metadata Validator SECOND → it needs clean content to auto-extract author/date
Deduplicator THIRD → operates on clean, consistent content for accurate comparison
PII Detector FOURTH → scans clean content; runs before indexing to prevent data leaks
Freshness Tracker LAST → read-only assessment; doesn't need any previous output
```

If you ran deduplication before noise filtering, two copies of the same document — one with a "Page 3 of 12" header and one without — would look like different documents. They would both pass deduplication and get indexed. The noise filter removes these artifacts so the deduplicator sees the same canonical content from both copies.

---

### 11.1 Noise Filtering

**The NLP Problem: Boilerplate vs Content**

Boilerplate text (headers, footers, disclaimers, cookie banners) is characterized by:
- **High repetition**: The same footer appears on every document from the same source
- **Low information density**: Generic phrases, not domain-specific knowledge
- **Low position entropy**: Always at the top or bottom, never in the middle of documents

**Regex-based approach** (your current method):

```python
# Patterns for common boilerplate
patterns = [
    r"Page \d+ of \d+",           # Page numbers
    r"Confidential\s*[-–]\s*Internal Only",  # Standard disclaimers
    r"©\s*\d{4}.*All rights reserved",       # Copyright footers
    r"^\s*\d+\s*$",                          # Standalone page numbers
]
```

Regex is fast (O(n) per pattern) and deterministic but brittle — it only catches patterns you anticipated.

**trafilatura** (listed as a dependency in your code but not in requirements.txt — this is Bug #9):

trafilatura uses **content density scoring** — a heuristic from web scraping research. Each block's score = `len(text) / len(html_tags)`. Content has high density; navigation menus have low density (many `<a>` tags, little text).

This is a domain-adapted version of **TextRank** thinking: content blocks that are dense, unique, and centrally located are more likely to be the actual article.

**For enterprise documents** (your IT docs use case), regex patterns for known boilerplate structures are often more precise than trafilatura (which is optimized for web content). The right approach: combine both.

---

### 11.2 Metadata Validation

**Why Auto-Extraction Matters**

Metadata drives several downstream features:
- **Freshness tracking**: Needs `created_date` or `modified_date`
- **Source attribution**: Users need to know where the answer came from
- **Category-aware retrieval**: Different freshness thresholds for "policy" vs "tutorial" docs

When metadata is missing (common with ad-hoc file uploads), your validator auto-extracts it:

**Date extraction**: Uses `dateutil.parser` which handles an enormous variety of formats (`"January 3, 2024"`, `"03/01/24"`, `"2024-01-03T14:22:00Z"`, etc.) by trying multiple parsers in sequence.

**Author extraction**: Reads file system metadata (`os.stat()`), then tries PDF metadata (`fitz.open().metadata`), then pattern matches in content (`"Written by: John Smith"`).

**Category classification**: Your code appears to use keyword matching against a taxonomy. A more robust approach would be a zero-shot classifier using a BERT-like model:
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(content[:500], candidate_labels=["policy", "tutorial", "FAQ", "report"])
category = result["labels"][0]  # Highest-probability label
```

---

### 11.3 Deduplication — The 3-Tier Strategy

Deduplication in RAG is critical: if the same document is indexed 3 times, retrieval returns it 3 times, the LLM gets repetitive context, and response quality degrades.

**Tier 1: SHA-256 Hash (Exact Match)**

```python
import hashlib
content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
if content_hash in seen_hashes:
    return DUPLICATE
seen_hashes.add(content_hash)
```

Catches: byte-for-byte identical documents. Common scenario: same file copied to multiple directories, or same content synced from SharePoint.

Misses: documents that are 99% identical but differ by a version number in the footer.

**Tier 2: SimHash (Near-Duplicate Detection)**

SimHash was developed by Moses Charikar (2002) and made practically famous by Google's paper "Detecting Near-Duplicates for Web Crawling" (Manku et al., 2007 — research.google.com/pubs/archive/33026.pdf), where they used it on 8 billion web pages.

**How SimHash works — the math made clear:**

1. **Feature extraction**: Tokenize the document into weighted features (words weighted by TF-IDF, or just frequency)

2. **Hash each feature**: For each feature `f` with weight `w`, compute a standard hash `h(f)` as a 64-bit integer (binary vector)

3. **Weighted voting**: Maintain a 64-element counter vector `V` (initialized to zero). For each bit position `i`:
   - If bit `i` of `h(f)` is 1: `V[i] += w`
   - If bit `i` of `h(f)` is 0: `V[i] -= w`

4. **Binarize**: The SimHash fingerprint `F[i] = 1 if V[i] > 0 else 0`

5. **Compare**: Hamming distance between two fingerprints = number of bit positions that differ

**The key property**: Similar documents (same words, similar frequency distributions) will have similar `V` vectors and therefore similar fingerprints → small Hamming distance.

**Google's finding**: For an 8B page web crawl, using 64-bit SimHash with Hamming distance threshold k=3 achieved practical near-duplicate detection with high precision.

**Your code's Hamming threshold**: `distance ≤ 3` (out of 64 bits) — consistent with Google's recommendation for documents.

**Why not just use Jaccard similarity (MinHash)?**

MinHash (based on Jaccard similarity) is another locality-sensitive hashing approach. SimHash is chosen over MinHash because:
- SimHash uses **weighted** features (TF-IDF), capturing term importance
- SimHash is a single hash operation; MinHash requires multiple independent hash functions
- SimHash's cosine similarity approximation is more appropriate for long documents than Jaccard (set-based)

**Tier 3: Semantic Deduplication**

This catches conceptually duplicate documents that use different words — e.g., "VPN Setup Guide for Windows" and "How to Configure VPN on Windows Machines" are near-identical in content but would have very different SHA-256 hashes and moderately different SimHash fingerprints.

Semantic dedup embeds the document content as a dense vector and computes cosine similarity:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(doc.content[:512])  # Your code uses first 512 chars
cosine_sim = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))

if cosine_sim >= 0.92:
    return SEMANTIC_DUPLICATE
```

**Critical Bug #5 in your code**: Semantic dedup uses only the **first 512 characters**.

This is a serious problem for paraphrased duplicates where the beginning is different (e.g., one document starts with an executive summary, another starts directly with content). The entire unique semantic signal is in the first 512 chars.

**Fix**: Use mean pooling over multiple windows, or use a document-level embedding:
```python
# Better: embed entire document (or representative chunks)
chunks = [content[i:i+512] for i in range(0, min(len(content), 3000), 512)]
embeddings = model.encode(chunks)
doc_embedding = np.mean(embeddings, axis=0)  # Mean pooling
```

**Choosing the cosine similarity threshold (0.92)**:
- `0.95+`: Only catches near-identical documents (different formatting, minor additions)
- `0.92`: Catches strongly paraphrased content while avoiding false positives on topically related but distinct documents
- `0.85`: Too aggressive — will flag legitimately different documents about the same topic (e.g., two different guides about VPN) as duplicates

---

### 11.4 PII Detection — Microsoft Presidio

### Why PII Detection is Mandatory in RAG

Without PII scrubbing, documents like:
```
Employee complaint filed by John Smith (SSN: 123-45-6789, 
manager: Sarah Jones, email: sarah.jones@company.com). 
Incident occurred at the San Francisco office.
```

...get indexed and returned verbatim in RAG answers. Any user who asks a broadly related question receives PII they shouldn't have access to.

**Why Presidio over pure regex:**

Pure regex can detect structured PII (phone numbers, SSNs, credit cards) but fails on:
- **Person names**: "John Smith" is not detectable by regex — you need Named Entity Recognition (NER)
- **Context-dependent PII**: "the system administrator" might refer to a specific identifiable person when combined with other context
- **Organization-specific patterns**: Internal employee IDs, project codes that identify individuals

**How Presidio Works:**

Presidio uses a **multi-layer detection architecture** (presidio.microsoft.io):

1. **Regex recognizers**: Fast pattern matching for structured PII — email addresses (`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`), US phone numbers, SSNs, credit cards, IPs

2. **Checksum validators**: Validates detected candidates against mathematical checksums. A US credit card number has a Luhn checksum. Presidio validates it before flagging — this eliminates false positives on random 16-digit numbers.

3. **spaCy NER model**: For entities that require linguistic understanding — PERSON names, LOCATION, ORGANIZATION. spaCy's NER uses a **transition-based parser** with CNN feature extractors, achieving real-time inference.

4. **Context-aware enhancer**: If the word "phone" appears near a number pattern, confidence score increases. This uses lemma matching in a 5-token window around the candidate entity.

**The Critical Bug #2 in your code:**

```python
# ❌ Your current code
def detect(self, text):
    analyzer = AnalyzerEngine()  # Creates new instance on EVERY call!
    # AnalyzerEngine.__init__() loads spaCy model from disk
    # For en_core_web_lg: ~750MB model loaded from disk each time
    # This takes 2-5 seconds per call
    results = analyzer.analyze(text=text, language="en")
```

For a batch of 1000 documents, this reloads the spaCy model 1000 times = 2000-5000 seconds of pure overhead, vs ~10 seconds if initialized once.

```python
# ✅ Fix
class PIIDetector:
    def __init__(self):
        self._analyzer = AnalyzerEngine()  # Loaded ONCE
    
    def detect(self, text):
        return self._analyzer.analyze(text=text, language="en")
```

**Presidio's anonymization options:**

After detection, Presidio's `AnonymizerEngine` can apply:
- `replace`: `John Smith` → `<PERSON>`
- `mask`: `123-45-6789` → `***-**-6789` (partial masking, preserves format)
- `hash`: `john.smith@company.com` → `sha256_hash_of_email` (reversible with key, useful for analytics)
- `redact`: Complete removal

For RAG, `replace` with entity type tags is the standard approach — `<PERSON>`, `<PHONE_NUMBER>`, `<EMAIL_ADDRESS>` — as it preserves the semantic structure ("The complaint was filed by `<PERSON>`") while preventing PII exposure.

**Why not just use regex for everything?**

The fundamental limit of regex for PII: regex is a **finite automaton** — it recognizes regular languages (fixed patterns). Names are drawn from an open vocabulary and cannot be expressed as a regular expression. The only way to detect names reliably is with a trained NER model that understands language structure.

---

### 11.5 Freshness Tracking

**Why Stale Documents Hurt RAG**

A document from 2019 explaining "How to set up 2FA with SMS" might rank highest in retrieval for "2FA setup" because it has exactly the right keywords. But your company now uses an authenticator app. The LLM will confidently give the wrong answer.

Freshness tracking doesn't delete documents but flags them — your retrieval/reranking stage can use the freshness score to downrank old results.

**How to compute freshness:**

```python
from datetime import datetime, timedelta

def compute_freshness_score(doc_date: datetime, 
                             category: str,
                             thresholds: dict) -> float:
    age_days = (datetime.now() - doc_date).days
    max_age = thresholds.get(category, 365)  # Default: 1 year
    
    if age_days <= 0:
        return 1.0  # Fresh
    elif age_days >= max_age:
        return 0.0  # Stale
    else:
        return 1.0 - (age_days / max_age)  # Linear decay
```

**Category-specific thresholds** (configurable in your system):

| Category | Suggested Max Age | Reason |
|---|---|---|
| Security policy | 90 days | Threats evolve fast |
| Software tutorial | 180 days | Software versions change |
| HR policy | 365 days | Annual review cycles |
| Architecture document | 730 days | Infrastructure is slower-changing |
| Compliance regulation | 365 days | Annual updates typical |

**Version detection**: Your `freshness.py` also looks for version numbers in document titles (`v2.1`, `Version 3`, `2024 Edition`). When multiple versions of the same document exist, the older versions should be marked stale even if their absolute age is recent.

---

## 12. Orchestration Patterns

### The Synchronous Bottleneck — Why It's Critical

Your `load_directory()` is synchronous:
```python
for file_path in directory.rglob("*"):
    doc = self._load_file(file_path)  # Blocks entire process
    if doc:
        documents.append(doc)
```

For 10,000 documents:
- PyMuPDF (Tier 1): ~0.5s per document → **~83 minutes total**
- If Tesseract OCR needed (Tier 3): ~3s per page × 10 pages → **~8 hours total**

**Solution: Concurrent Processing**

Document parsing is **CPU-bound** for PyMuPDF/OCR and **I/O-bound** for loading from disk. Python's concurrency options:

| Method | Best For | GIL Impact |
|---|---|---|
| `threading.ThreadPoolExecutor` | I/O-bound (disk reads, API calls) | GIL limits CPU parallelism |
| `multiprocessing.ProcessPoolExecutor` | CPU-bound (PyMuPDF, OCR) | Full parallelism, separate GIL per process |
| `asyncio` | Many concurrent I/O operations | Single-threaded, event-loop based |

```python
import concurrent.futures

def load_directory_parallel(self, path: Path, max_workers: int = 4):
    files = list(path.rglob("*"))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files as futures
        future_to_file = {
            executor.submit(self._load_file_safe, f): f 
            for f in files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            doc = future.result()
            if doc:
                yield doc  # Streaming yield — don't accumulate all in memory
```

**Note**: ProcessPoolExecutor requires that `_load_file_safe` is picklable (serializable for inter-process communication). Avoid passing `self` directly — pass the file path and reconstruct necessary context inside the worker.

**Retry Logic for API Calls (tenacity library):**

Vision API and OCR service calls can fail transiently (network timeout, rate limit). `tenacity` handles exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_vision_api(self, image_base64: str) -> str:
    response = self._openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}]
    )
    return response.choices[0].message.content
```

This retries with delays of 2s, 4s, 8s — giving transient failures time to resolve.

---

## 13. Critical Bugs — Root Cause Analysis

| # | File | Bug | Root Cause | Correct Fix |
|---|---|---|---|---|
| 1 | `loader.py` | MD5 for `doc_id`, SHA-256 in deduplicator | Two engineers wrote different files without alignment | Standardize on SHA-256 everywhere. Never use MD5 for new code. |
| 2 | `pii_detector.py` | New `AnalyzerEngine()` per call | spaCy model initialization misunderstood as stateless | Initialize once in `__init__`, reuse the single instance |
| 3 | `image_handler.py` | New `OpenAI()` client per image | Same misconception — client init seen as trivial | Initialize once in `__init__` |
| 4 | `loader.py` | Fully synchronous `load_directory()` | No async design from the start | Use `ProcessPoolExecutor` for CPU-bound parsing |
| 5 | `deduplicator.py` | First 512 chars only for semantic dedup | Simplification that misses paraphrased duplicates | Mean-pool embeddings across multiple 512-char windows |
| 6 | `loader.py` | Double dedup (MD5 in loader + SHA-256 in pipeline) | Quality pipeline added later without removing original dedup | Remove MD5 dedup from `load_directory()`, let quality pipeline own all dedup |
| 7 | `config.py` | Plain class, not Pydantic BaseSettings | Pydantic not considered or not known | Use `pydantic_settings.BaseSettings` for env var loading + validation |
| 8 | `table_extractor.py` | Silent `except Exception` | Defensive coding mistake | `except ImportError: logger.warning(...)` — separate import errors from runtime errors |
| 9 | `noise_filter.py` | trafilatura/readability listed but not in requirements.txt | Requirements not synced with code | Add to requirements.txt or remove the code references |
| 10 | All parsers | No file size limits | Not considered during design | Add `if file_path.stat().st_size > config.max_file_bytes: raise FileTooLargeError` |
| 11 | `pdf_parser.py` | PDF opened twice (PyMuPDF + pdfplumber) | Tier 1 and Tier 2 implemented independently | Cache file bytes: `raw_bytes = file_path.read_bytes()`, pass to both |
| 12 | `email_parser.py` | Attachments listed but not parsed | Recursive parsing not designed | Implement recursive MIME walker that routes attachments back to the parser registry |
| 13 | `test_ingestion.py` | Print-based script, not pytest | Tests added as quick validation, not proper testing | Convert to pytest with parametrized fixtures, snapshot assertions |

---

## 14. Production Architecture Upgrade Path

### The Parser Registry Pattern

Replace the if/elif chain with a registry:

```python
from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        ...
    
    @classmethod
    def can_parse(cls, file_path: Path) -> bool:
        return file_path.suffix in cls.supported_extensions

class ParserRegistry:
    _parsers: dict[str, type[BaseParser]] = {}
    
    @classmethod
    def register(cls, parser_class: type[BaseParser]):
        for ext in parser_class.supported_extensions:
            cls._parsers[ext] = parser_class
        return parser_class
    
    @classmethod
    def get_parser(cls, file_path: Path) -> BaseParser:
        ext = file_path.suffix.lower()
        parser_class = cls._parsers.get(ext, UnstructuredParser)
        return parser_class()

# Registration (in each parser file):
@ParserRegistry.register
class PDFParser(BaseParser):
    supported_extensions = [".pdf"]
    ...
```

Adding a new format (`.xlsx`, `.csv`) now requires:
1. Create `xlsx_parser.py` inheriting `BaseParser`
2. Add `supported_extensions = [".xlsx", ".csv"]`
3. Decorate with `@ParserRegistry.register`
4. No changes to `loader.py`

### Pydantic Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Automatically reads from .env file and environment variables
    openai_api_key: str
    ocr_dpi: int = 300
    max_file_size_mb: float = 100.0
    semantic_dedup_threshold: float = 0.92
    simhash_distance_threshold: int = 3
    
    model_config = {"env_file": ".env"}

settings = Settings()  # Validates types, raises clear error if required vars missing
```

### Incremental Ingestion (Critical for Production)

Re-processing all 10,000 documents every night is wasteful. Incremental ingestion only processes new or changed files:

```python
import sqlite3
from pathlib import Path
from hashlib import sha256

class IngestionCache:
    def __init__(self, db_path: str = "ingestion_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT PRIMARY KEY,
                mtime REAL,
                content_hash TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def needs_processing(self, file_path: Path) -> bool:
        mtime = file_path.stat().st_mtime
        row = self.conn.execute(
            "SELECT mtime FROM processed_files WHERE file_path=?", 
            (str(file_path),)
        ).fetchone()
        return row is None or row[0] != mtime
    
    def mark_processed(self, file_path: Path, content_hash: str):
        self.conn.execute("""
            INSERT OR REPLACE INTO processed_files (file_path, mtime, content_hash)
            VALUES (?, ?, ?)
        """, (str(file_path), file_path.stat().st_mtime, content_hash))
        self.conn.commit()
```

---

## 15. The Bigger Picture — How Ingestion Quality Affects Downstream

### The Garbage In, Garbage Out Chain

**Bad ingestion → Bad chunks → Bad embeddings → Bad retrieval → Bad answers**

Each stage amplifies upstream errors:

**Effect on chunking (Stage 2):**
If noise (page headers, footers) is not removed, a chunker may create chunks entirely composed of boilerplate. These chunks get embedded, indexed, and retrieved — returning "Page 4 of 16 | Confidential" as relevant context to the LLM.

**Effect on embeddings (Stage 3):**
Embedding models like `text-embedding-ada-002` or `all-MiniLM-L6-v2` convert text to dense vectors. Noisy, incoherent text (from poor PDF parsing) produces vectors in semantically meaningless regions of the embedding space. Retrieving these vectors for a user query returns irrelevant chunks.

The embedding model treats every token equally — it cannot distinguish "this is boilerplate" from "this is content". The cleaner your text, the more meaningful your embedding vectors.

**Effect on retrieval (Stage 4):**
Duplicate documents create "retrieval bias". If "VPN Setup Guide" is indexed 5 times, all 5 slots of the top-k retrieved results may be variations of the same document, crowding out other relevant documents. The LLM receives redundant context and lacks diversity.

**Effect on generation (Stage 5):**
LLMs with stale documents confidently give outdated information. PII in the context window leaks into generated answers. Tables parsed as unstructured text produce garbled numerical reasoning.

**Key insight**: Investing in ingestion quality has compounding returns. Every improvement at Stage 1 improves the quality of all 6 subsequent stages.

### What "Good" Ingestion Looks Like — Metrics

To measure your ingestion quality:

| Metric | How to Measure | Target |
|---|---|---|
| Text extraction accuracy | Compare extracted text against human transcription on 50 test PDFs (BLEU/F1) | BLEU-4 > 0.85 |
| Table preservation rate | Count tables correctly converted to Markdown vs total tables | > 90% |
| PII detection recall | Run on annotated test set with known PII | Recall > 95% |
| Duplicate detection rate | Inject known duplicates, measure removal rate | > 98% exact, > 90% near-duplicate |
| Ingestion throughput | Documents processed per minute | Define per your SLA |
| Freshness flag accuracy | Documents correctly flagged as stale vs human review | > 90% |

---

## Quick Reference: Library Decision Guide

| Task | Use This | Not This | Why |
|---|---|---|---|
| PDF text (normal) | PyMuPDF | pypdf/pdfminer | 10× faster, equal quality (benchmarked) |
| PDF text (complex layout) | pdfplumber | PyMuPDF alone | Spatial analysis handles columns |
| PDF text (scanned) | Tesseract via pytesseract | Nothing | Only option for image-based PDFs |
| PDF tables (with borders) | Camelot (lattice) | PyMuPDF table | Purpose-built for lattice tables |
| PDF tables (borderless) | pdfplumber | Camelot | Camelot stream mode is less accurate |
| Image understanding (diagram) | OpenAI Vision API | Tesseract | OCR produces garbage on diagrams |
| HTML content extraction | BeautifulSoup + lxml | html.parser | 2-5× faster, lxml already a dependency |
| Web-scraped HTML (articles) | trafilatura | BeautifulSoup | Boilerplate-aware, article-optimized |
| Encoding detection | charset-normalizer | chardet | More accurate, actively maintained |
| Near-duplicate detection | SimHash (64-bit, k≤3) | Jaccard/MinHash | Weighted features, single-pass, O(n) |
| Semantic duplicate detection | SentenceTransformers + cosine | LLM-based similarity | Fast, free, offline |
| PII detection | Microsoft Presidio | Pure regex | Catches names via NER; regex can't |
| Config management | Pydantic BaseSettings | Plain class | Type validation, .env loading, clear errors |
| Retry logic | tenacity | Manual try/except loops | Exponential backoff, configurable |
| Parallel processing | ProcessPoolExecutor | ThreadPoolExecutor | CPU-bound tasks need separate GIL |

---

## Summary: The 5 Most Important Things to Remember

1. **Ingestion quality has compounding effects.** Every noise token you don't remove, every duplicate you don't catch, every table you parse as flat text — all of these degrade every downstream stage. It is always worth investing more in ingestion.

2. **The 3-tier PDF cascade is evidence-based.** PyMuPDF → pdfplumber → Tesseract maps directly to the three PDF failure modes: corrupt encoding (OCR solves), complex layout (pdfplumber solves), normal text (PyMuPDF handles 90%). This is backed by the 2024 arxiv benchmark study.

3. **SimHash is O(n), not O(n²).** Comparing every pair of documents for similarity is quadratic. SimHash makes near-duplicate detection linear by mapping each document to a 64-bit fingerprint and using Hamming distance in that compact space. This is how Google processes billions of web pages.

4. **Presidio is not "just regex with extra steps."** The NER component is why it exists. Person names, organization names, and context-dependent PII are fundamentally undetectable by regex — they require a trained language model.

5. **Never re-initialize heavy models inside loops.** The spaCy model (Presidio), sentence transformer (semantic dedup), and OpenAI client (Vision API) should all be initialized once and reused. These initializations load 100MB+ of data — doing it per-document destroys performance.

---

## References & Sources

- **PDF Parser Benchmark Study (2024)**: Adhikari & Agarwal, "A Comparative Study of PDF Parsing Tools Across Diverse Document Categories" — [arxiv.org/abs/2410.09871](https://arxiv.org/abs/2410.09871)
- **SimHash Algorithm**: Manku et al., "Detecting Near-Duplicates for Web Crawling", Google Research — [research.google.com/pubs/archive/33026.pdf](https://research.google.com/pubs/archive/33026.pdf)
- **Microsoft Presidio**: [microsoft.github.io/presidio](https://microsoft.github.io/presidio/analyzer/)
- **PyMuPDF Performance Benchmarks**: [pymupdf.readthedocs.io/en/latest/app4.html](https://pymupdf.readthedocs.io/en/latest/app4.html)
- **py-pdf Benchmark Suite**: [github.com/py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)
- **Trafilatura Deduplication (SimHash)**: [trafilatura.readthedocs.io/en/latest/deduplication.html](https://trafilatura.readthedocs.io/en/latest/deduplication.html)
- **Charikar SimHash (Original Paper)**: M. Charikar, "Similarity Estimation Techniques from Rounding Algorithms", STOC 2002

---
*Document Version: 1.0 | Generated: February 2026 | Based on your Rag-architecture codebase + research*
