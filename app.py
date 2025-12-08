# app.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Model path
MODEL_PATH = "./Phi-3-mini-4k-instruct-q4.gguf" 

# --- 1. Initialize the LLM ---
print("Loading LLM...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=2048,
    top_p=1,
    n_ctx=2048, # Context window
    verbose=False
)

# --- 2. Load the Embedding Model ---
print("Loading embeddings...")
embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, n_batch=512, n_ctx=2048)

# --- 3. Load the existing Vector Database ---
print("Loading vector store...")
vector_store = Chroma(
    persist_directory="./db", 
    embedding_function=embeddings
)

# --- 4. Create the LangChain pipeline ---
print("Creating RAG chain...")

# This is the prompt template. It's the "magic" of RAG.
template = """
<|system|>
You are a helpful assistant. Use the following context to answer the user's question.
If you don't know the answer, just say you don't know. Don't make stuff up.
Context: {context}
<|user|>
Question: {input}
<|assistant|>
"""

prompt = PromptTemplate.from_template(template)

# This chain combines the context and question into a prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# This chain does the retrieval *and* the generation
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 means get top 3 results
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("RAG bot is ready. Ask a question (type 'exit' to quit).")

# --- 5. Run the query loop ---
while True:
    query = input("\n> ")
    if query.lower() == 'exit':
        break
    if query.strip() == "":
        continue

    # Invoke the chain
    response = retrieval_chain.invoke({"input": query})
    
    print("\nAnswer:")
    print(response["answer"])