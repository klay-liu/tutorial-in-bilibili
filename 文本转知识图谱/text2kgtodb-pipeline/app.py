import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import utils
from kb import KB
import pandas as pd

st.set_page_config(page_title="E2E-App-for-KG", 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

st.header("Build Knowledge Graph from text: End-to-end information extraction pipeline with Rebel and Neo4j")

# sidebar
with st.sidebar:
    st.markdown("Referred articles: \n")
    st.markdown("[Building a Knowledge Base from Texts: a Full Practical Example](https://medium.com/nlplanet/building-a-knowledge-base-from-texts-a-full-practical-example-8dbbffb912fa)")
    st.markdown("[Building Knowledge Graphs with Rebel: Step By Step Guide for Extracting Entities & Enriching Info](https://www.youtube.com/watch?v=Xc4t38UPc2k)")
    st.markdown("[Building Knowledge Graphs: REBEL, LlamaIndex, and REBEL + LlamaIndex](https://medium.com/@sauravjoshi23/building-knowledge-graphs-rebel-llamaindex-and-rebel-llamaindex-8769cf800115)")
    st.header("What is a Knowledge Base")
    st.markdown("A [**Knowledge Base (KB)**](https://en.wikipedia.org/wiki/Knowledge_base) is information stored in structured data, ready to be used for analysis or inference. Usually a KB is stored as a graph (i.e. a [**Knowledge Graph**](https://www.ibm.com/cloud/learn/knowledge-graph)), where nodes are **entities** and edges are **relations** between entities.")
    st.markdown("_For example, from the text \"Fabio lives in Italy\" we can extract the relation triplet <Fabio, lives in, Italy>, where \"Fabio\" and \"Italy\" are entities._")
    st.header("How to build a Knowledge Graph")
    st.markdown("To build a Knowledge Graph from text, we typically need to perform two steps:\n- Extract entities, a.k.a. **Named Entity Recognition (NER)**, i.e. the nodes.\n- Extract relations between the entities, a.k.a. **Relation Classification (RC)**, i.e. the edges.\nRecently, end-to-end approaches have been proposed to tackle both tasks simultaneously. This task is usually referred to as **Relation Extraction (RE)**. In this demo, an end-to-end model called [**REBEL**](https://github.com/Babelscape/rebel/blob/main/docs/EMNLP_2021_REBEL__Camera_Ready_.pdf) is used, trained by [Babelscape](https://babelscape.com/).")
    st.header("How REBEL works")
    st.markdown("REBEL is a **text2text** model obtained by fine-tuning [**BART**](https://huggingface.co/docs/transformers/model_doc/bart) for translating a raw input sentence containing entities and implicit relations into a set of triplets that explicitly refer to those relations. You can find [REBEL in the Hugging Face Hub](https://huggingface.co/Babelscape/rebel-large).")
    

# Loading the model
st_model_load = st.text('Loading Rebel model. This should take 1-2 minutes.')

@st.cache_resource(show_spinner=False)
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

placeholder_str = """How to build a Knowledge Graph
To build a knowledge graph from text, we typically need to perform two steps:

Extract entities, a.k.a. Named Entity Recognition (NER), which are going to be the nodes of the knowledge graph.
Extract relations between the entities, a.k.a. Relation Classification (RC), which are going to be the edges of the knowledge graph.
These multiple-step pipelines often propagate errors or are limited to a small number of relation types. Recently, end-to-end approaches have been proposed to tackle both tasks simultaneously. This task is usually referred to as Relation Extraction (RE). In this article, we’ll use an end-to-end model called REBEL, from the paper Relation Extraction By End-to-end Language generation.

How REBEL works
REBEL is a text2text model trained by BabelScape by fine-tuning BART for translating a raw input sentence containing entities and implicit relations into a set of triplets that explicitly refer to those relations. It has been trained on more than 200 different relation types.

The authors created a custom dataset for REBEL pre-training, using entities and relations found in Wikipedia abstracts and Wikidata, and filtering them using a RoBERTa Natural Language Inference model (similar to this model). Have a look at the paper to know more about the creation process of the dataset. The authors also published their dataset on the Hugging Face Hub.

The model performs quite well on an array of Relation Extraction and Relation Classification benchmarks.

You can find REBEL in the Hugging Face Hub.
"""

if 'text' not in st.session_state:
    st.session_state.text = ""
text = st.session_state.text
text = st.text_area('Text:', value=text, height=300, disabled=False, max_chars=30000, placeholder=placeholder_str)

def ingest_to_neo4j(filename='relations.csv'):
    with st.spinner('Ingesting to Neo4j...'): 
        import time
        time.sleep(10)
        df = pd.read_csv('./data/' + filename)
        df = df.drop(columns={'meta'})
        for col in df.columns:
            df[col] = df[col].apply(utils.sanitize).str.lower()

        conn = utils.Neo4jConnection(uri=st.secrets['uri'], user=st.secrets['username'], pwd=st.secrets['password'])

        # Loop through data and create Cypher query
        for i in range(len(df['head'])):

            query = f'''
                MERGE (head:Head {{name: "{df['head'][i]}"}})

                MERGE (tail:tail {{value: "{df['tail'][i]}"}})

                MERGE (head)-[:{df['type'][i]}]->(tail)
                '''
            result = conn.query(query, db=st.secrets['database'])
        st.success('Data ingested to Neo4j', icon="✅")

tab1, tab2 = st.tabs(["Generation & Visualization", "Data Ingestion"])

with tab1:
    button_text = "Generate & Show KG"
    with st.spinner('Generating KG...'):
        # generate KB button
        if st.button(button_text):

            kb = utils.from_text_to_kb(text, model, tokenizer, verbose=True)
            
            kb.save_csv(f"./data/{st.secrets['triplets_filename']}")

            # save chart
            utils.save_network_html(kb, filename="network.html")
            st.session_state.kb_chart = "./networks/network.html"
            # st.session_state.kb_text = kb.get_textual_representation()
            st.session_state.error_url = None

            # kb chart session state
            if 'kb_chart' not in st.session_state:
                st.session_state.kb_chart = None
            if 'kb_text' not in st.session_state:
                st.session_state.kb_text = None
            if 'error_url' not in st.session_state:
                st.session_state.error_url = None

            # show graph
            if st.session_state.error_url:
                st.markdown(st.session_state.error_url)
            elif st.session_state.kb_chart:
                with st.container():
                    st.subheader("Generated KG")
                    st.markdown("*You can interact with the graph and zoom.*")
                    html_source_code = open(st.session_state.kb_chart, 'r', encoding='utf-8').read()
                    components.html(html_source_code, width=700, height=700)
                    st.markdown(st.session_state.kb_text)
            st.success('KG Generated!', icon="✅")
with tab2:
    st.button('Ingest to Neo4j', on_click=ingest_to_neo4j)


