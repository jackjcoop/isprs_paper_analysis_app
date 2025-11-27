# ISPRS Paper Formatting Analyzer

Automated compliance checker for ISPRS full paper submissions. Validates formatting requirements including page layout, fonts, citations, and document structure.

## Features

- **Hybrid Analysis**: Combines Google Cloud Document AI with PyMuPDF for robust extraction
- **56+ Validation Checks**: Page size, margins, fonts, headings, citations, references, and more
- **Citation Validation**: Author-Year format detection, citation-reference matching, figure/table citations
- **PDF Report Generation**: Annotated PDF with highlighted issues and comment annotations
- **Anonymization Check**: Detects named entities for review version compliance
- **Format Support**: PDF, DOCX, and LaTeX files (with automatic conversion)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Google Cloud

```bash
# Option A: Environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Option B: Pass as argument (see Usage below)
```

### 3. Analyze a Paper

```bash
python main.py paper.pdf
```

## Installation

### Prerequisites

- Python 3.8+
- Google Cloud Account with Document AI API enabled
- Optional: LibreOffice (Word conversion) or pdflatex (LaTeX conversion)

### Google Cloud Setup

1. Create a Google Cloud project
2. Enable the Document AI API
3. Create a Document AI processor (Layout Parser)
4. Create a service account and download credentials JSON

## Usage

```bash
# Basic analysis
python main.py document.pdf

# Custom output path
python main.py document.pdf --output results.json

# Specify credentials
python main.py document.pdf --credentials /path/to/credentials.json

# Check anonymization (for review submissions)
python main.py document.pdf --anon

# Analyze Word/LaTeX (auto-converts to PDF)
python main.py paper.docx
python main.py paper.tex
```

### Command Options

```
Arguments:
  document              Path to document (PDF, DOCX, or TEX)

Options:
  --output, -o         Path for JSON output file
  --credentials, -c    Path to Google Cloud credentials JSON
  --anon               Check for anonymization (review version)
  --no-report          Skip PDF report generation
```

## Output

The analyzer produces:

1. **Console Summary**: Pass/fail status with warnings and errors
2. **JSON Report**: Detailed extraction results and validation checks
3. **Annotated PDF**: Original document with highlighted issues and comments

## Validation Checks

### Document Structure
- Page size (A4), margins, page count (6-8)
- Required sections: Title, Abstract, Authors, Keywords
- Single-column header, two-column body layout

### Formatting
- Font type (Times New Roman), sizes (12pt title, 9pt body)
- Heading alignment, numbering sequence
- Reference alphabetical order, justification

### Citations
- Author-Year format validation
- Citation-reference matching
- Figure/Table citation coverage and proximity (±1 page)

### Review Version
- Anonymization check (named entity detection)

## Project Structure

```
app_3/
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── reference/                   # ISPRS checklist
└── support/
    ├── converter.py             # Document format conversion
    ├── document_ai_client.py    # Google Cloud Document AI
    ├── pymupdf_extractor.py     # Font/style extraction
    ├── citation_validator.py    # Citation validation
    ├── anonymization_checker.py # Named entity recognition
    ├── validator.py             # Compliance validation
    ├── report_generator.py      # PDF report generation
    └── output_generator.py      # JSON output formatting
```

## Troubleshooting

**Google Cloud credentials not found**
- Set `GOOGLE_APPLICATION_CREDENTIALS` or use `--credentials` flag

**LibreOffice/pdflatex not found**
- Install the tool or convert to PDF manually before analysis

**Label not verified warning**
- Document may use different label text or positioning

## Requirements

- google-cloud-documentai
- google-auth
- PyMuPDF
- spaCy (with en_core_web_sm model)

## License

This tool is part of the Paper Analysis project.
