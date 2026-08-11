'''
Streamlit UI for the Clinical Guidelines RAG Assistant.

Usage:
    streamlit run app.py
'''

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent/'src'))
from rag_graph import build_graph

st.set_page_config(page_title='Clinical Guidelines Assistant', page_icon='🩺', layout='wide')

CLASSIFICATION_LABELS = {
    'guideline_question': '📋 Guideline Question',
    'emergency': '⚠️ Emergency Detected',
    'off_topic': '❓ Out of Scope'
}


@st.cache_resource
def get_app():
    return build_graph()


def render_sources(retrieved):
    if not retrieved:
        return
    
    with st.expander(f' Sources used ({len(retrieved)})'):
        for i, r in enumerate(retrieved, 1):
            m = r['metadata']
            st.markdown(
                f'**Source {i}:** {m['source_title']}'
                f"(p.{m['page']}) -- [{m['publisher']}, {m['year']}] ({m['source_url']})"
            )
            st.caption(r['text'][:300] + '...')

    
def main():
    st.title('Clinical Guidelines Q&A Assistant')
    st.caption(
        'Answers grounded in WHO Hypertension (2021), WHO Diabetes (2018), '
        'and NHLBI Asthma (2007) guidelines. Not a subsitute for clinical judgement.'
    )

    app = get_app()

    if 'history' not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message('user'):
            st.write(turn['query'])
        
        with st.chat_message('assistant'):
            label = CLASSIFICATION_LABELS.get(turn['classification'], '')
            if label:
                st.caption(label)
            st.write(turn['answer'])
            render_sources(turn.get('retrieved'))

    query = st.chat_input('Ask about diabetes, hypertension, or asthma management...')

    if query:
        with st.chat_message('uesr'):
            st.write(query)
        
        with st.chat_message('assistant'):
            with st.spinner('Checking guidelines...'):
                result = app.invoke({'query': query, 'retries': 0})
            
            label = CLASSIFICATION_LABELS.get(result['classification'], '')
            if label:
                st.caption(label)

            confidence = result.get('confidence')
            if confidence is not None:
                st.progress(min(confidence, 1.0), text=f'Retrieval confidence: {confidence:.2f}')

            st.write(result['answer'])
            render_sources(result.get('retrieved'))
        
        st.session_state.history.append(result | {'query': query})

    with st.sidebar:
        st.header('About')
        st.markdown(
            'This assistant answers **guideline-level** clinical questions'
            'on 3 conditions only. It will:\n'
            '- Refuse individualized dosing/diagnosis advice\n'
            '- Flag emergencies and redirect to care\n'
            '- Say when it lacks sufficient guideline coverage\n'
            '- Always cite the source page it used'
        )
        if st.button('Clear conversation'):
            st.session_state.history = []
            st.rerun()


if __name__=='__main__':
    main()