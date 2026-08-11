'''
Automated tests for the RAG pipeline: classification routing, retrieval relevance, citation enforcement, and guardrail behavior.

Usage:
    pytest tests/test_rag_pipeline.py -v
'''

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent/'src'))

from rag_graph import build_graph, classify_query, retrieve, CONFIDENCE_THRESHOLD
from eval_questions import EVAL_QUESTIONS

sys.path.insert(0, str(Path(__file__).parent))

@pytest.fixture(scope='module')
def app():
    return build_graph()


class TestClassification:
    '''Fast, deterministic tests using keyword-routed cases (no LLM call).'''

    def test_emergency_keywords_detected(self):
        result = classify_query({'query': 'severe chest pain right now'})
        assert result['classification']=='emergency'
    
    
    def test_guideline_keywords_detected(self):
        result = classify_query({'query': 'diabetes insulin dosing guideline'})
        assert result['classification']=='guideline_question'

    
class TestRetrieval:
    '''Confirms retrieval pulls chunks from the correct disease corpus.'''

    @pytest.mark.parametrize('item', [q for q in EVAL_QUESTIONS if q.get('expected_disease')])
    def test_retrieval_matches_expected_disease(self, item):
        result = retrieve({'query': item['query']})
        top_disease = result['retrieved'][0]['metadata']['disease']
        assert top_disease==item['expected_disease'], (
            f"Query '{item['query']}' retrieved '{top_disease}', "
            f"Expected '{item['expected_disease']}'"
        )

    
    def test_confidence_scores_are_reasonable_range(self):
        result = retrieve({'query': 'first line treatment for hypertension'})
        assert 0.0 <= result['confidence'] <= 1.0

    
class TestGuardrails:
    ''' End-to-end graph tests for each routing path.'''

    def test_emergency_short_circuits_no_retrieval(self, app):
        result = app.invoke({'query': 'I am having severe chest pain', 'retries': 0})
        assert result['classification']=='emergency'
        assert 'emergency' in result['answer'].lower()
        assert 'retrieved' not in result


    def test_off_topic_deflected(self, app):
        result = app.invoke({'query': 'What blood pressure level required starting treatment', 'retries': 0})
        assert '[Source' in result['answer']


    def tests_guideline_answer_has_disclaimer(self, app):
        result = app.invoke({'query': 'What is first-line treatment for type 2 diabetes?', 'retries': 0})
        assert 'not individualized medical advice' in result['answer'].lower()

    
    @pytest.mark.parametrize('item', [q for q in EVAL_QUESTIONS if q.get('expect_low_confidence')])
    def test_out_of_corpus_topics_flagged_low_confidence(self, item):
        result = retrieve({'query': item['query']})
        assert result['confidence'] < CONFIDENCE_THRESHOLD, (
            f"Expected low confidence for out-of-corpus query '{item['query']}', "
            f"got {result['confidence']:.3f}"
        )


def print_confidence_report():
    '''Run standalone to see the actual confidence score distribution and calibrate CONFIDENCE_THRESHOLD in rag_graph.py accordingly.'''

    print(F"\n{'Query':<70} {'Confidence':<10} {'Class'}")
    print('-' * 95)

    for item in EVAL_QUESTIONS:
        result = Aruntcs({'query': item['query']})
        expected = item.get('expected_disease', item['expected_class'])
        print(f"{item['query'][:68]:<70} {result['confidence']:<10.3f} {expected}")

        
if __name__=='__main__':
    print_confidence_report()