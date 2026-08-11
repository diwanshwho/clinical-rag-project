'''
Downloads the raw guideline PDFs listed in sources.py into data/raw/.

Usage:
    python src/download_sources.py
'''

import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
from sources import SOURCES

RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw'
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
    )
}

def download_all():
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    for src in SOURCES:
        out_path = RAW_DIR/src['filename']
        if out_path.exists():
            print(f'[skip]{src['filename']} already exists')
            continue
        print(f'[fetch]{src['disease']} <-- {src['pdf_url']}')
        resp = requests.get(src['pdf_url'], headers=HEADERS, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f'[saved] {out_path} ({len(resp.content)/1024:.0f} KB)')

if __name__ == "__main__":
    download_all()