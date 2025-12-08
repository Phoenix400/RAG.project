# ui_app.py
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- IMPORTANT: Update this path ---
MODEL_PATH = "./Phi-3-mini-4k-instruct-q4.gguf" 

# --- A. CUSTOM CSS FOR A "REALLY GOOD" INTERFACE ---
def load_css():
    st.markdown("""
    <style>
        /* Main page background */
        .stApp {
            background-color: #0E1117; /* Streamlit's dark background */
        }
        [data-testid="stChatMessage"] {
            border-radius: 10px;
            padding: 1em 1.25em;
            margin-bottom: 10px;
        }
        [data-testid="stChatMessage"] p { color: #FFFFFF; }
        div[data-testid="stChatMessage"][class*="user"] {
            background-color: #262730; 
            border: 1px solid #1E5083;
        }
        div[data-testid="stChatMessage"][class*="assistant"] {
            background-color: #1E1F2A; 
            border: 1px solid #4A4A4A;
        }
        .stExpander {
            background-color: #1E1F2A;
            border-radius: 10px;
            border: 1px solid #4A4A4A;
        }
        .stExpander summary { font-size: 0.9rem; color: #A0A0A0; }
        .stExpander summary:hover { color: #00A1FF; }
        .stTitle { font-family: 'Arial', sans-serif; color: #FAFAFA; font-weight: 600; }
        .stChatMessage[data-testid="stChatMessage"][class*="user"] .stAvatar::after {
            content: 'U'; font-size: 1.5rem; display: flex; justify-content: center;
            align-items: center; background-color: #1E5083; border-radius: 50%;
            width: 40px; height: 40px;
        }
        .stChatMessage[data-testid="stChatMessage"][class*="assistant"] .stAvatar::after {
            content: 'A'; font-size: 1.5rem; display: flex; justify-content: center;
            align-items: center; background-color: #4A4A4A; border-radius: 50%;
            width: 40px; height: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- B. FUNCTION TO LOAD THE RAG CHAIN ---
@st.cache_resource
def load_rag_chain():
    print("Loading LLM...")
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        n_ctx=2048,
        verbose=False
    )
    
    print("Loading embeddings...")
    embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, n_batch=32, n_ctx=2048, verbose=False)
    
    print("Loading vector store...")
    vector_store = Chroma(
        persist_directory="./db", 
        embedding_function=embeddings
    )
    
    print("Creating RAG chain...")
    
    template = """
Instruction: You are an expert Q&A assistant.
Your task is to answer the user's question based ONLY on the context provided.
If the context does not contain the answer, you must say: "I'm sorry, that information is not in my database."
Do not add any other information or commentary.

Context:
{context}

Question:
{input}

Answer:
"""
    
    prompt = PromptTemplate.from_template(template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # --- FIXED RETRIEVER: Use simple top-K retrieval (k=3) ---
    # This ensures the top 3 best matching chunks are always included.
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) 
    # --- End of change ---

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG bot is ready.")
    return retrieval_chain

# --- C. STREAMLIT APP UI ---
st.set_page_config(page_title="RAG Study Bot", layout="wide")
load_css() # <-- LOAD OUR CUSTOM STYLES

st.title("🎓 RAG Study Bot: Chat with Your Data")

# --- CLEAR CHAT BUTTON FIX ---
def clear_chat_history():
    st.session_state.messages = []

st.button('🧹 Clear Chat History', on_click=clear_chat_history)
# ----------------------------


# Load the chain
with st.spinner("Loading AI model... This may take a moment."):
    chain = load_rag_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is Kruskal's algorithm?"):
    # Append the user message to history *before* processing
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # We include the *entire* history in the input to maintain context (the default Streamlit behavior)
        
        # NOTE: For better chat, you should construct the full prompt here including history.
        # However, for simplicity and adherence to the original code structure, we only pass the new prompt.
        
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            
            with st.expander("Show Sources"):
                sources = []
                for doc in response["context"]:
                    source_data = doc.__dict__ 
                    metadata = source_data.get("metadata", {})
                    source_name = metadata.get("source", "Unknown Source")
                    page_content = source_data.get("page_content", "No content found.")
                    
                    sources.append({
                        "source": source_name,
                        "content": page_content
                    })
                st.json(sources) 

    st.session_state.messages.append({"role": "assistant", "content": answer})