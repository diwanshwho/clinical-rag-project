'''
Embeds all processed chunks and stores them in a persistent Chroma vector store.

Usage:
    python src/build_vectorstore.py
'''

import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed' 
CHROMA_DIR = Path(__file__).parent.parent / 'data' / 'chroma_db' 
COLLECTION_NAME = 'clinical_guidelines'
BATCH_SIZE = 500

# all-MiniLM-L6-v2: fast, good baseline for retrieval. Swap for a larger model later.
EMBED_MODEL = 'all-MiniLM-L6-v2'

def load_chunks():
    with open(PROCESSED_DIR / 'all_chunks.json', encoding='utf-8') as f:
        return json.load(f)
    

def build():
    chunks = load_chunks()
    print(f'[load] {len(chunks)} chunks from {PROCESSED_DIR/'all_chunks.json'}')

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    #start fresh each time this run , to avoid duplicate/stale entries
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f'[success] Deleted collection successfully')
    except Exception:
        print(f'[warning] Collection deletion failed -- {Exception}')
        pass
    
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i+BATCH_SIZE]
        collection.add(
            ids=[f'chunk_{i+j}' for j in range(len(batch))], 
            documents=[c['text'] for c in batch], 
            metadatas=[c['metadata'] for c in batch]
        )
        print(f'[embedded] {min(i+BATCH_SIZE, len(chunks))}/{len(chunks)}')
    
    print(f'\n[done] Collection "{COLLECTION_NAME}" has {collection.count()} vectors')
    print(f'[saved] {CHROMA_DIR}')

    # sanity check
    res = collection.query(query_texts=['first line drug for hypertension'], n_results=2)
    print('\n[sanity check] top match:')
    print(' ', res['documents'][0][0][:150], '...')
    print(' ', res['metadatas'][0][0])


if __name__=='__main__':
    build()
