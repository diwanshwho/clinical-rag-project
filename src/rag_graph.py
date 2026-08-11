'''
LangGraph RAG pipeline with guardrails for the Clinical Guidelines Assistant.

Flow:
    classify --> [emergency | off_topic | guideline_question]
    guideline_question --> retrieve --> [low_confidence | sufficient]
    sufficient --> generate --> validate_citation --> [retry | finalize]
'''

import os
from typing import TypedDict, List, Dict, Optional
import chromadb
import httpx
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END 
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

##---- config ----##
CHROMA_DIR = './data/chroma_db'
COLLECTION_NAME = 'clinical_guidelines'
EMBED_MODEL = 'all-MiniLM-L6-v2'
CONFIDENCE_THRESHOLD = 0.55     # should adjust this after eval testing
MAX_RETRIES = 2
TOP_K = 4

GATEWAY_BASE_URL = os.environ.get('GATEWAY_BASE_URL')
GATEWAY_API_KEY = os.environ.get('GATEWAY_API_KEY')
GATEWAY_MODEL = os.environ.get('GATEWAY_MODEL')

httpx_client = httpx.Client(verify=False)
llm_client = OpenAI(base_url=GATEWAY_BASE_URL, api_key=GATEWAY_API_KEY, http_client=httpx_client)
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_collection(COLLECTION_NAME, embedding_function=_embedder)

EMERGENCY_KEYWORDS = ["chest pain", "can't breathe", "cannot breathe", "unconscious", "severe bleeding", "suicidal", "overdose", "blue lips", "not breathing"]
KNOWN_DISEASES = ["hypertension", "diabetes", "asthma", "blood pressure", "insulin", "inhaler"]

class RAGState(TypedDict):
    query: str
    classification: str
    retrieved: List[Dict]
    confidence: float
    answer: str
    retries: int

## Defining Nodes
def classify_query(state: RAGState) -> Dict:
    query = state['query'].lower()
    if any(kw in query for kw in EMERGENCY_KEYWORDS):
        return {'classification': 'emergency'}
    if any(kw in query for kw in KNOWN_DISEASES):
        return {'classification': 'guideline_question'}

    #ambiguous case
    try:
        response = llm_client.chat.completions.create(
            model=GATEWAY_MODEL,
            messages=[{
                'role': 'user',
                'content': (
                    'Classify this question into exactly one label: '
                    '"guideline_question" (about diabetes, hypertension, or asthma clinical management),'
                    '"emergency" (describes an acute medical emergency), '
                    'or "off_topic" (anything else). '
                    f'Reply with only the label. \n\nQuestion: {state['query']}'
                ),
            }],
            max_tokens=10,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        if label not in ('guideline_question', 'emergency', 'off_topic'):
            label = 'off_topic'
    except Exception as e:
        print(f'[classify_query] LLM call failed, defaulting to off_topic: {e}')
        label = 'off_topic'

    return {'classification': label}
    

def route_after_classify(state: RAGState) -> str:
    return state['classification']


def retrieve(state: RAGState) -> dict:
    results = _collection.query(query_texts=[state['query']], n_results=TOP_K)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    #converting distance to a rough 0-1 confidence score
    confidence = max(0.0, 1 - distances[0]) if distances else 0.0
    retrieved = [{'text': d, 'metadata': m} for d, m in zip(docs, metas)]
    return {'retrieved': retrieved, 'confidence': confidence}


def route_after_retrieve(state: RAGState) -> str:
    return 'sufficient' if state['confidence'] >= CONFIDENCE_THRESHOLD else 'low_confidence'


def generate_answer(state: RAGState) -> dict:
    context_blocks = []
    for i, r in enumerate(state['retrieved']):
        m = r['metadata']
        context_blocks.append(f'[Source {i+1}: {m['source_title']}, p.{m['page']}]\n{r['text']}')
    context = '\n\n'.join(context_blocks)

    system_prompt = (
        'You are a clinical guidelines assistant. Answer ONLY using the provided context.'
        'Every claim must cite its source using the format [Source N].'
        'If the context does not fully answer the question, say so explicitly.'
        'Never provide individualized dosing or diagnostic advice - describe guideline-level recommendations only.'
    )
    user_prompt = f'Context:\n{context}\n\nQuestion: {state['query']}'

    response = llm_client.chat.completions.create(
        model=GATEWAY_MODEL,
        messages=[
            {'role': 'system',
            'content': system_prompt},
            {'role': 'user',
            'content': user_prompt},
        ],
        max_tokens=500,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return {'answer': answer, 'retries': state.get('retries', 0) + 1}


def route_after_validation(state:RAGState) -> str:
    has_citation = '[Source' in state['answer']
    if has_citation:
        return 'finalize'
    if state['retries'] >= MAX_RETRIES:
        return 'finalize'  #give up gracefully
    return 'retry'


def emergency_response(state: RAGState) -> dict:
    return {
        'answer': (
            'This sound like it may be a medical emergency.'
            'Please contact emergency services or go to the nearest emergency room immediately.'
            'This assistant is not equipped to help with acute emergencies.'
        )
    }


def off_topic_response(state: RAGState) -> dict:
    return {
        'answer': (
            "I'm scoped to clinical guideline questions on diabetes, hypertension, and asthma management. This question falls outside that scope."
        )
    }


def insufficient_coverage(state: RAGState) -> dict:
    return {
        'answer': (
            "I don't have sufficient guideline coverage to answer this confidently."
            'Please consult a clinician or rephrase with more specific clinical terms.'
        )
    }


def finalize(state: RAGState) -> dict:
    disclaimer = (
        '\n\n---\nThis is guideline-level information, not individualized medical advice. Consult a qualified clinician for patient specific decisions.'
    )
    if not state['answer'].strip().endswith(disclaimer.strip()):
        return {'answer': state['answer']+disclaimer}
    return {}


## Graph
def build_graph():
    g = StateGraph(RAGState)
    g.add_node('classify', classify_query)
    g.add_node('retrieve', retrieve)
    g.add_node('generate', generate_answer)
    g.add_node('emergency_response', emergency_response)
    g.add_node('off_topic_response', off_topic_response)
    g.add_node('insufficient_coverage', insufficient_coverage)
    g.add_node('finalize', finalize)

    g.set_entry_point('classify')
    g.add_conditional_edges('classify', route_after_classify, {
        'emergency': 'emergency_response',
        'off_topic': 'off_topic_response',
        'guideline_question': 'retrieve'
    })
    g.add_conditional_edges('retrieve', route_after_retrieve, {
        'sufficient': 'generate',
        'low_confidence': 'insufficient_coverage'
    })
    g.add_conditional_edges('generate', route_after_validation, {
        'retry': 'generate',
        'finalize': 'finalize'
    })
    g.add_edge('emergency_response', END)
    g.add_edge('off_topic_response', END)
    g.add_edge('insufficient_coverage', END)
    g.add_edge('finalize', END)

    return g.compile()


if __name__=='__main__':
    app = build_graph()
    test_queries = [
        'What is the first-line drug treatment for hypertension?',
        'I have severe chest pain, what should I do ?',
        'What is a good recipe for pasta ?',
    ]
    for q in test_queries:
        result = app.invoke({'query': q, 'retries': 0})
        print(f'\nQuery: {q}')
        print(f'\nClassification: {result['classification']}')
        print(f'\nAnswer: {result['answer']}')