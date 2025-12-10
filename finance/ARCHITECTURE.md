# System Architecture

**Financial Document Retrieval System Design**

This document provides detailed architecture diagrams and component descriptions for the financial document retrieval system.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINANCIAL QA SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                    ┌─────────────┐       │
│   │   10-K PDF  │                                    │   Question  │       │
│   │   Documents │                                    │    Input    │       │
│   └──────┬──────┘                                    └──────┬──────┘       │
│          │                                                  │              │
│          ▼                                                  ▼              │
│   ┌──────────────────────────────┐            ┌─────────────────────────┐  │
│   │     INGESTION PIPELINE       │            │     QUERY PIPELINE      │  │
│   │                              │            │                         │  │
│   │  PDF Parser ──▶ TableFormer  │            │  Question ──▶ Statement │  │
│   │       │            │         │            │  Analysis      Detection│  │
│   │       ▼            ▼         │            │       │            │    │  │
│   │  Text      ──▶  Chunking     │            │       ▼            ▼    │  │
│   │  Extraction      Service     │            │  Entity     ──▶ Hybrid  │  │
│   │                    │         │            │  Extraction     Search  │  │
│   │                    ▼         │            │                    │    │  │
│   │              Embedding       │            │                    ▼    │  │
│   │              Service         │            │               Reranking │  │
│   └────────────────┬─────────────┘            └──────────┬──────────────┘  │
│                    │                                     │                 │
│                    ▼                                     ▼                 │
│          ┌─────────────────┐                   ┌─────────────────┐        │
│          │     QDRANT      │◀─── Search ──────▶│   CoT REASONING │        │
│          │   Vector DB     │                   │   (Claude LLM)  │        │
│          │                 │                   │                 │        │
│          │  - Embeddings   │                   │  - Step-by-step │        │
│          │  - Metadata     │                   │  - Source citing│        │
│          │  - Filters      │                   │  - Validation   │        │
│          └─────────────────┘                   └────────┬────────┘        │
│                                                         │                 │
│                                                         ▼                 │
│                                                ┌─────────────────┐        │
│                                                │     ANSWER      │        │
│                                                │   + Evidence    │        │
│                                                └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline

### Overview

The ingestion pipeline transforms raw 10-K PDF documents into searchable, structured chunks with metadata.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────┐                                                              │
│  │  10-K    │                                                              │
│  │  PDF     │                                                              │
│  └────┬─────┘                                                              │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      PDF PARSER (PyMuPDF)                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │    Text     │  │   Images    │  │    Page     │                  │  │
│  │  │  Extraction │  │  Extraction │  │  Metadata   │                  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │  │
│  └─────────┼────────────────┼────────────────┼──────────────────────────┘  │
│            │                │                │                             │
│            ▼                ▼                │                             │
│  ┌─────────────────────────────────────┐    │                             │
│  │        TABLE EXTRACTOR              │    │                             │
│  │                                     │    │                             │
│  │  ┌───────────────────────────────┐ │    │                             │
│  │  │     TableFormer Detection     │ │    │                             │
│  │  │  (microsoft/table-transformer)│ │    │                             │
│  │  │                               │ │    │                             │
│  │  │  Input: Page Image (150 DPI)  │ │    │                             │
│  │  │  Output: Table Bounding Boxes │ │    │                             │
│  │  │  Threshold: 0.7 confidence    │ │    │                             │
│  │  └───────────────┬───────────────┘ │    │                             │
│  │                  │                 │    │                             │
│  │                  ▼                 │    │                             │
│  │  ┌───────────────────────────────┐ │    │                             │
│  │  │  TableFormer Structure Recog  │ │    │                             │
│  │  │                               │ │    │                             │
│  │  │  Input: Cropped Table Image   │ │    │                             │
│  │  │  Output: Cell Bounding Boxes  │ │    │                             │
│  │  │  Threshold: 0.6 confidence    │ │    │                             │
│  │  └───────────────┬───────────────┘ │    │                             │
│  │                  │                 │    │                             │
│  │                  ▼                 │    │                             │
│  │  ┌───────────────────────────────┐ │    │                             │
│  │  │       Cell Text Extraction    │ │    │                             │
│  │  │                               │ │    │                             │
│  │  │  - Map boxes to PDF coords    │ │    │                             │
│  │  │  - Extract text per cell      │ │    │                             │
│  │  │  - Preserve row/col structure │ │    │                             │
│  │  └───────────────────────────────┘ │    │                             │
│  └──────────────────┬──────────────────┘    │                             │
│                     │                       │                             │
│                     ▼                       │                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    CHUNKING SERVICE                                  │  │
│  │                                                                      │  │
│  │   TEXT CHUNKS                        TABLE CHUNKS                    │  │
│  │  ┌────────────────┐                 ┌────────────────┐              │  │
│  │  │ - 512 tokens   │                 │ - Row-by-row   │              │  │
│  │  │ - 50 overlap   │                 │ - Full context │              │  │
│  │  │ - Preserve     │                 │ - Headers      │              │  │
│  │  │   paragraphs   │                 │ - Caption      │              │  │
│  │  └────────────────┘                 │ - Scale info   │              │  │
│  │                                     └────────────────┘              │  │
│  │                                                                      │  │
│  │   METADATA ATTACHED TO EACH CHUNK:                                   │  │
│  │   ┌────────────────────────────────────────────────────────────┐    │  │
│  │   │ {                                                          │    │  │
│  │   │   "company_name": "APPLE",                                 │    │  │
│  │   │   "fiscal_year": 2019,                                     │    │  │
│  │   │   "document_type": "10-K",                                 │    │  │
│  │   │   "page_number": 45,                                       │    │  │
│  │   │   "chunk_type": "table_row",                               │    │  │
│  │   │   "statement_type": "income_statement",                    │    │  │
│  │   │   "table_caption": "Consolidated Statements of Income",    │    │  │
│  │   │   "scale": "millions"                                      │    │  │
│  │   │ }                                                          │    │  │
│  │   └────────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                     │                                                      │
│                     ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    EMBEDDING SERVICE                                 │  │
│  │                                                                      │  │
│  │  Model: sentence-transformers/all-MiniLM-L6-v2                      │  │
│  │  Dimension: 384                                                      │  │
│  │                                                                      │  │
│  │  Input: Chunk text                                                   │  │
│  │  Output: 384-dim embedding vector                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                     │                                                      │
│                     ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    QDRANT VECTOR DATABASE                            │  │
│  │                                                                      │  │
│  │  Collection: financial_documents                                     │  │
│  │                                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐    │  │
│  │  │  Point Structure:                                           │    │  │
│  │  │  {                                                          │    │  │
│  │  │    "id": "chunk_uuid",                                      │    │  │
│  │  │    "vector": [0.123, -0.456, ...],  // 384 dims             │    │  │
│  │  │    "payload": {                                             │    │  │
│  │  │      "text": "Net sales | FY2021: $33,067M | ...",          │    │  │
│  │  │      "company_name": "APPLE",        // Filterable          │    │  │
│  │  │      "fiscal_year": 2021,            // Filterable          │    │  │
│  │  │      "statement_type": "income",     // Filterable          │    │  │
│  │  │      "page_number": 45,                                     │    │  │
│  │  │      "table_caption": "..."                                 │    │  │
│  │  │    }                                                        │    │  │
│  │  │  }                                                          │    │  │
│  │  └─────────────────────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Query Pipeline

### Overview

The query pipeline analyzes questions, retrieves relevant chunks, and generates answers with chain-of-thought reasoning.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  INPUT: "What is the FY2020 operating cash flow ratio for Nike?"   │   │
│  └───────────────────────────────┬────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    QUESTION ANALYSIS                                 │  │
│  │                                                                      │  │
│  │  ┌───────────────────────┐  ┌───────────────────────┐              │  │
│  │  │   ENTITY EXTRACTION   │  │  STATEMENT DETECTION  │              │  │
│  │  │                       │  │                       │              │  │
│  │  │  Company: NIKE        │  │  "operating cash      │              │  │
│  │  │  (matched "Nike")     │  │   flow ratio"         │              │  │
│  │  │                       │  │        │              │              │  │
│  │  │  Fiscal Year: 2020    │  │        ▼              │              │  │
│  │  │  (matched "FY2020")   │  │  REQUIRED:           │              │  │
│  │  │                       │  │  - Cash Flow Stmt    │              │  │
│  │  └───────────────────────┘  │  - Balance Sheet     │              │  │
│  │                             └───────────────────────┘              │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     HYBRID RETRIEVAL                                 │  │
│  │                                                                      │  │
│  │   STEP 1: Semantic Search                                            │  │
│  │   ┌────────────────────────────────────────────────────────────┐    │  │
│  │   │  Query: "operating cash flow ratio"                        │    │  │
│  │   │  Filters: company=NIKE, fiscal_year=2020                   │    │  │
│  │   │  Top-K: 15                                                 │    │  │
│  │   │  Output: 15 semantically similar chunks                    │    │  │
│  │   └────────────────────────────────────────────────────────────┘    │  │
│  │                                                                      │  │
│  │   STEP 2: Statement-Specific Retrieval                               │  │
│  │   ┌────────────────────────────────────────────────────────────┐    │  │
│  │   │  FOR EACH needed statement type:                           │    │  │
│  │   │                                                            │    │  │
│  │   │  Query: "2020 cash flow statement"                         │    │  │
│  │   │  Filters: company=NIKE, statement_type=cash_flow           │    │  │
│  │   │  Top-K: 10                                                 │    │  │
│  │   │                                                            │    │  │
│  │   │  Query: "2020 balance sheet"                               │    │  │
│  │   │  Filters: company=NIKE, statement_type=balance_sheet       │    │  │
│  │   │  Top-K: 10                                                 │    │  │
│  │   └────────────────────────────────────────────────────────────┘    │  │
│  │                                                                      │  │
│  │   STEP 3: Merge & Deduplicate                                        │  │
│  │   ┌────────────────────────────────────────────────────────────┐    │  │
│  │   │  - Combine all retrieved chunks                            │    │  │
│  │   │  - Remove duplicates (by chunk_id)                         │    │  │
│  │   │  - Sort by relevance score                                 │    │  │
│  │   │  - Keep top 50 chunks                                      │    │  │
│  │   └────────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                 CHAIN-OF-THOUGHT REASONING (Claude)                  │  │
│  │                                                                      │  │
│  │  Model: claude-sonnet-4-20250514                                    │  │
│  │  Temperature: 0 (deterministic)                                      │  │
│  │  Max Tokens: 1024                                                    │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────┐     │  │
│  │  │  PROMPT STRUCTURE:                                         │     │  │
│  │  │                                                            │     │  │
│  │  │  You are an expert financial analyst...                    │     │  │
│  │  │                                                            │     │  │
│  │  │  Financial Data:                                           │     │  │
│  │  │  [50 retrieved chunks with source references]              │     │  │
│  │  │                                                            │     │  │
│  │  │  Question: What is the FY2020 operating cash flow ratio... │     │  │
│  │  │                                                            │     │  │
│  │  │  Answer using this structure:                              │     │  │
│  │  │  ## Step 1: IDENTIFY the relevant data points              │     │  │
│  │  │  ## Step 2: EXTRACT the exact numbers needed               │     │  │
│  │  │  ## Step 3: PERFORM calculations                           │     │  │
│  │  │  ## Step 4: VERIFY your answer                             │     │  │
│  │  │  ## Step 5: FORMAT final answer                            │     │  │
│  │  │                                                            │     │  │
│  │  │  ANSWER:                                                   │     │  │
│  │  └────────────────────────────────────────────────────────────┘     │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────┐     │  │
│  │  │  LLM OUTPUT EXAMPLE:                                       │     │  │
│  │  │                                                            │     │  │
│  │  │  ## Step 1: IDENTIFY                                       │     │  │
│  │  │  I need: Cash from Operations, Current Liabilities         │     │  │
│  │  │                                                            │     │  │
│  │  │  ## Step 2: EXTRACT                                        │     │  │
│  │  │  From Source 3: Cash from ops = $8,738M                    │     │  │
│  │  │  From Source 7: Current liabilities = $6,567M              │     │  │
│  │  │                                                            │     │  │
│  │  │  ## Step 3: PERFORM                                        │     │  │
│  │  │  Ratio = $8,738M / $6,567M = 1.33                          │     │  │
│  │  │                                                            │     │  │
│  │  │  ## Step 4: VERIFY                                         │     │  │
│  │  │  - Both values from FY2020 ✓                               │     │  │
│  │  │  - Ratio > 1.0 indicates healthy liquidity ✓               │     │  │
│  │  │                                                            │     │  │
│  │  │  ## Step 5: FORMAT                                         │     │  │
│  │  │  ANSWER: 1.33                                              │     │  │
│  │  └────────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    ANSWER PARSING & VALIDATION                       │  │
│  │                                                                      │  │
│  │  1. Extract answer from "ANSWER:" marker                             │  │
│  │  2. Clean formatting ($, commas)                                     │  │
│  │  3. Validate against common sense                                    │  │
│  │     - Ratios: 0-100 range                                            │  │
│  │     - Percentages: -100 to 500                                       │  │
│  │     - Skip year values (2015-2025)                                   │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  OUTPUT:                                                             │  │
│  │  {                                                                   │  │
│  │    "answer": "1.33",                                                 │  │
│  │    "reasoning": "## Step 1: ...",                                    │  │
│  │    "sources": [                                                      │  │
│  │      {"chunk_id": "...", "text": "Cash from ops: $8,738M", ...}     │  │
│  │    ],                                                                │  │
│  │    "confidence": 0.85                                                │  │
│  │  }                                                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### TableFormer Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                      TABLEFORMER PIPELINE                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INPUT: PDF Page                                                       │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  1. PAGE TO IMAGE CONVERSION                                  │     │
│  │                                                               │     │
│  │     page_pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))  │     │
│  │     image = Image.frombytes(...)                              │     │
│  │                                                               │     │
│  │     Resolution: 150 DPI (balance speed/quality)               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  2. TABLE DETECTION                                           │     │
│  │                                                               │     │
│  │     Model: microsoft/table-transformer-detection              │     │
│  │                                                               │     │
│  │     ┌──────────┐      ┌──────────────┐      ┌──────────┐    │     │
│  │     │  Image   │ ──▶  │   DETR-based │ ──▶  │  Boxes   │    │     │
│  │     │  (full   │      │   Detection  │      │  + Scores│    │     │
│  │     │   page)  │      │              │      │          │    │     │
│  │     └──────────┘      └──────────────┘      └──────────┘    │     │
│  │                                                               │     │
│  │     Threshold: 0.7 confidence                                 │     │
│  │     Output: List of (x1, y1, x2, y2) table regions            │     │
│  └──────────────────────────────────────────────────────────────┘     │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  3. TABLE STRUCTURE RECOGNITION                               │     │
│  │                                                               │     │
│  │     Model: microsoft/table-transformer-structure-recognition  │     │
│  │                                                               │     │
│  │     FOR EACH detected table:                                  │     │
│  │                                                               │     │
│  │     ┌──────────┐      ┌──────────────┐      ┌──────────┐    │     │
│  │     │  Cropped │ ──▶  │   Structure  │ ──▶  │  Cell    │    │     │
│  │     │  Table   │      │   Recognition│      │  Boxes   │    │     │
│  │     │  Image   │      │              │      │  + Labels│    │     │
│  │     └──────────┘      └──────────────┘      └──────────┘    │     │
│  │                                                               │     │
│  │     Cell types: table, table-column, table-row,               │     │
│  │                 table-column-header, table-projected-row-header│     │
│  │                 table-spanning-cell                           │     │
│  └──────────────────────────────────────────────────────────────┘     │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  4. CELL TEXT EXTRACTION                                      │     │
│  │                                                               │     │
│  │     FOR EACH cell bounding box:                               │     │
│  │                                                               │     │
│  │     1. Scale box from image coords to PDF coords              │     │
│  │        pdf_x = img_x * (page_width / img_width)               │     │
│  │                                                               │     │
│  │     2. Create clip rectangle                                  │     │
│  │        rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)      │     │
│  │                                                               │     │
│  │     3. Extract text from rectangle                            │     │
│  │        text = page.get_text("text", clip=rect)                │     │
│  │                                                               │     │
│  │     4. Clean and normalize text                               │     │
│  │        - Strip whitespace                                     │     │
│  │        - Handle special characters                            │     │
│  │        - Preserve numeric formatting                          │     │
│  └──────────────────────────────────────────────────────────────┘     │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  5. TABLE DATA STRUCTURE                                      │     │
│  │                                                               │     │
│  │     TableData:                                                │     │
│  │       caption: str              # "Consolidated Statements..."│     │
│  │       headers: List[List[str]]  # Multi-level headers         │     │
│  │       rows: List[List[Cell]]    # Data rows                   │     │
│  │       page_number: int                                        │     │
│  │       bounding_box: Tuple[4]                                  │     │
│  │       scale_indicator: str      # "in millions"               │     │
│  │                                                               │     │
│  │     Cell:                                                     │     │
│  │       value: str                                              │     │
│  │       row_span: int                                           │     │
│  │       col_span: int                                           │     │
│  │       is_header: bool                                         │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│  OUTPUT: List[TableData] - All tables from page with structure         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Statement Type Detection

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STATEMENT TYPE DETECTION                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INPUT: "What is the operating cash flow ratio?"                       │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  METRIC → STATEMENT MAPPING                                   │     │
│  │                                                               │     │
│  │  ┌─────────────────────────────────────────────────────────┐ │     │
│  │  │ INCOME STATEMENT METRICS                                │ │     │
│  │  │                                                         │ │     │
│  │  │ revenue, sales, gross profit, operating income,         │ │     │
│  │  │ ebitda, net income, earnings, eps, cogs,                │ │     │
│  │  │ gross margin, operating margin, sg&a, r&d               │ │     │
│  │  └─────────────────────────────────────────────────────────┘ │     │
│  │                                                               │     │
│  │  ┌─────────────────────────────────────────────────────────┐ │     │
│  │  │ BALANCE SHEET METRICS                                   │ │     │
│  │  │                                                         │ │     │
│  │  │ total assets, current assets, total liabilities,        │ │     │
│  │  │ current liabilities, equity, working capital,           │ │     │
│  │  │ accounts receivable, inventory, debt, ppe               │ │     │
│  │  └─────────────────────────────────────────────────────────┘ │     │
│  │                                                               │     │
│  │  ┌─────────────────────────────────────────────────────────┐ │     │
│  │  │ CASH FLOW STATEMENT METRICS                             │ │     │
│  │  │                                                         │ │     │
│  │  │ cash from operations, operating cash flow,              │ │     │
│  │  │ free cash flow, capex, dividends paid                   │ │     │
│  │  └─────────────────────────────────────────────────────────┘ │     │
│  │                                                               │     │
│  │  ┌─────────────────────────────────────────────────────────┐ │     │
│  │  │ RATIO METRICS (require multiple statements)             │ │     │
│  │  │                                                         │ │     │
│  │  │ operating cash flow ratio → [cash flow, balance sheet]  │ │     │
│  │  │ return on assets         → [income, balance sheet]      │ │     │
│  │  │ return on equity         → [income, balance sheet]      │ │     │
│  │  │ asset turnover           → [income, balance sheet]      │ │     │
│  │  │ current ratio            → [balance sheet]              │ │     │
│  │  │ debt to equity           → [balance sheet]              │ │     │
│  │  └─────────────────────────────────────────────────────────┘ │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│  DETECTION FLOW:                                                       │
│                                                                        │
│  1. Check ratio metrics first (most specific)                          │
│     "operating cash flow ratio" matches → ["cash flow", "balance"]     │
│                                                                        │
│  2. If no ratio match, check individual metrics                        │
│     "revenue growth" → ["income statement"]                            │
│                                                                        │
│  3. If nothing matches, return all statement types                     │
│                                                                        │
│  OUTPUT: ["cash flow statement", "balance sheet"]                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END DATA FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INGESTION (One-time per document)                                     │
│   ─────────────────────────────────                                     │
│                                                                         │
│   10-K PDF ──▶ PyMuPDF ──▶ TableFormer ──▶ Chunking ──▶ Embeddings     │
│       │          │            │              │              │          │
│       │          │            │              │              ▼          │
│       │          │            │              │         ┌─────────┐     │
│       │          │            │              │         │ Qdrant  │     │
│       │          │            │              │         │         │     │
│       │          │            │              │         │ 621,673 │     │
│       │          │            │              │         │ chunks  │     │
│       │          │            │              │         └────┬────┘     │
│       │          │            │              │              │          │
│   ────┴──────────┴────────────┴──────────────┴──────────────┘          │
│                                                                         │
│   QUERY (Per question)                                                  │
│   ────────────────────                                                  │
│                                                                         │
│   Question ──▶ Entity ──▶ Statement ──▶ Hybrid ──▶ Claude ──▶ Answer   │
│       │        Extract    Detection     Search      CoT         │      │
│       │          │            │           │         │           │      │
│       ▼          ▼            ▼           ▼         ▼           ▼      │
│   "What is   "NIKE"      [cash flow,   50 chunks  Step-by-   "1.33"   │
│    FY2020    "2020"       balance]     retrieved   step                │
│    OCF                                            reasoning            │
│    ratio                                                               │
│    for                                                                 │
│    Nike?"                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **PDF Processing** | PyMuPDF (fitz) | Text and image extraction |
| **Table Detection** | TableFormer (Microsoft) | DETR-based table localization |
| **Table Structure** | TableFormer | Cell-level bounding boxes |
| **Vector DB** | Qdrant | Embeddings + metadata filtering |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim dense vectors |
| **LLM** | Claude Sonnet 4 | Chain-of-thought reasoning |
| **API** | FastAPI | HTTP endpoints |
| **Async** | AsyncAnthropic | Non-blocking LLM calls |

---

## Configuration Reference

```python
# config.py

class Config:
    # TableFormer
    TABLE_DETECTION_MODEL = "microsoft/table-transformer-detection"
    TABLE_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"
    TABLE_DETECTION_THRESHOLD = 0.7
    CELL_DETECTION_THRESHOLD = 0.6
    TABLE_IMAGE_DPI = 150

    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Chunking
    MAX_CHUNK_SIZE = 512  # tokens
    CHUNK_OVERLAP = 50    # tokens

    # Retrieval
    SEMANTIC_TOP_K = 15
    STATEMENT_TOP_K = 10
    FINAL_CHUNKS = 50

    # LLM
    LLM_MODEL = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS = 1024
    LLM_TEMPERATURE = 0

    # Qdrant
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "financial_documents"
```

---

Built by [Mixpeek](https://mixpeek.com)
