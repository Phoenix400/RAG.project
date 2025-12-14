# RAG-Based Algorithm Question Answering System

## Overview
This project is a Retrieval-Augmented Generation (RAG) system designed to answer
Data Structures and Algorithms (DSA) theory questions using a local Large Language Model (LLM).
Instead of relying on external APIs, the system performs semantic search over curated
algorithm notes and generates context-aware answers in real time.

The goal of this project is to demonstrate applied backend + ML system design,
including document ingestion, vector retrieval, and controlled LLM inference.

---

## Key Features
- End-to-end RAG pipeline using a local LLM (llama-cpp-python)**
- Vector-based semantic search with ChromaDB
- Structured ingestion of algorithm theory and question datasets
- Context-aware answer generation to reduce hallucinations
- Interactive **Streamlit UI** for querying the system

---

## Tech Stack
- Language: Python
- Vector Database: ChromaDB
- LLM Inference: llama-cpp-python (local model)
- Frontend: Streamlit
- Embeddings: Sentence/transformer-based embeddings

---

## System Architecture
1. Algorithm notes and question datasets are preprocessed and chunked
2. Text chunks are converted into embeddings and stored in ChromaDB
3. User submits a query via Streamlit UI
4. Relevant chunks are retrieved using semantic similarity search
5. Retrieved context is injected into the LLM prompt
6. Local LLM generates a concise, grounded answer

---

## Dataset Coverage
- Graph Algorithms: BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall
- Minimum Spanning Tree: Kruskal, Disjoint Set Union
- Sorting: Counting Sort, Radix Sort, Bucket Sort
- Dynamic Programming vs Greedy
- NP, NP-Complete, NP-Hard, P vs NP
- Backtracking (N-Queens)
- String Algorithms: KMP, Rabin-Karp, Huffman Coding

---

## Project Structure
- algorithms.txt      → Core algorithm theory dataset
- Questions.txt       → Question-focused dataset
- ingest.py           → Text preprocessing, embedding, and vector storage
- app.py              → Backend RAG pipeline
- ui_app.py           → Streamlit-based user interface
- requirements.txt    → Dependencies
- commands.txt        → Execution instructions

---

## Installation & Usage

```bash
pip install -r requirements.txt
python ingest.py
python app.py
streamlit run ui_app.py
