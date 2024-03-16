import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import MilvusVectorStore
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext

st.set_page_config(page_title="Chat with the LlamaIndex docs, powered by LlamaIndex", 
                   page_icon="🦙", 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

openai.api_key = st.secrets['openai_key']
st.title("Chat with the LlamaIndex docs 🦙")
st.info("Check out the refered blog (https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about LlamaIndex's documentation!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LlamaIndex docs. This should take 1-2 minutes."):
        # load documents
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the LlamaIndex docs and your job is to answer user's questions. Assume that all questions are related to the LlamaIndex docs. Keep your answers being based on facts – do not hallucinate features.")
        embed_model="local:BAAI/bge-small-en-v1.5"
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )     
        documents = SimpleDirectoryReader("./data/docs/").load_data()

        # # Create an index over the documnts with Milvus as vectordb
        # vector_store = MilvusVectorStore(dim=384, overwrite=False)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # index = VectorStoreIndex.from_documents(
        #     documents, storage_context=storage_context, service_context=service_context
        # )
        ## for demo, you can use the local stroage.
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index

index = load_data()

stream = st.checkbox("Streaming Response: If True, the streaming mode for chat-engine will be on.", value=True)  # Add a checkbox for streaming

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Enter a prompt here"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    """Clears the chat history and resets the state."""
    st.session_state.messages = [
         {"role": "assistant", "content": "Ask me a question about LlamaIndex's documentation!"}
         ]
    del st.session_state["chat_engine"]
# Create the sidebar
st.sidebar.header("Options")

# Add the "Clear History" button to the sidebar
if st.sidebar.button("Clear History", key="clear"):
    clear_chat_history()


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":        
    with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if stream:
                    response = st.session_state["chat_engine"].stream_chat(prompt)
                    response_str = ""
                    response_container = st.empty()
                    for token in response.response_gen:
                        response_str += token
                        response_container.write(response_str)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history
                else:
                    response = st.session_state.chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history
