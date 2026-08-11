'''
Extracts text from the raw guideline PDFs and produces citation-ready chunks.

Usage:
    python src/ingest.py
'''

import json
import re 
import sys
from pathlib import Path
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent))
from sources import SOURCES

RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed' 

HEADING_PATTERN = re.compile(r'^[A-Z][A-Za-z0-9 ,\-:/&]{3,80}$')

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def extract_pages(pdf_path: Path):
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            yield i, text

def guess_heading(page_text: str):
    for line in page_text.splitlines()[:8]:
        line = line.strip()
        if HEADING_PATTERN.match(line):
            return line 
    return None

def chunk_text(text: str, size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start+size*0.5:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end-overlap > start else end
    return [c for c in chunks if c]

def process_source(src: dict):
    pdf_path = RAW_DIR/src['filename']
    if not pdf_path.exists():
        print(f'[missing] {pdf_path} - run download_sources.py first')
        return []

    records = []
    current_heading = None
    for page_num, page_text in extract_pages(pdf_path):
        if not page_text.strip():
            continue
        heading = guess_heading(page_text)
        if heading:
            current_heading = heading

        for chunk in chunk_text(page_text):
            records.append(
                {
                    'text': chunk,
                    'metadata': {
                        'disease': src['disease'],
                        'source_title': src['title'],
                        'publisher': src['publisher'],
                        'year': src['year'],
                        'source_url': src['source_page'],
                        'page': page_num,
                        'section': current_heading,
                    },
                }
            )
    return records

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    for src in SOURCES:
        print(f'[ingest] {src['disease']} ({src['filename']})')
        records = process_source(src)
        print(f' --> {len(records)} chunks')
        all_records.extend(records)

        out_path = PROCESSED_DIR / f'{src['disease']}.json'
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    combined_path = PROCESSED_DIR / 'all_chunks.json'
    combined_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f'\n[done] {len(all_records)} total chunks --> {combined_path}')


if __name__ == '__main__':
    main()