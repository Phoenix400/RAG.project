# RAG.project

RAG Based Algorithm Question Answering System


* Overview:
This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions related to Algorithms and Data Structures. It uses a small local LLM (via llama-cpp-python) combined with vector retrieval using ChromaDB. The data source includes your algorithms notes covering MST, Graph Algorithms, Sorting Techniques, NP-Complete concepts, Dynamic Programming, Greedy Algorithms, String Matching Algorithms, and more.


* Project Structure:
1. algorithms.txt        -  Main algorithm theory dataset
2. Questions.txt         -  Additional questions dataset
3. ingest.py             -  Converts text data into embeddings and stores them in ChromaDB
4. app.py                -  Backend RAG pipeline that queries the model and vector DB
5. ui_app.py             -  Streamlit-based user interface to interact with the system
6. requirements.txt      -  Python dependencies
7. commands.txt          -  Commands to run the project [use the given commands in an order]


* Installation:
pip3 install -r requirements.txt


* Data Ingestion:
python3 ingest.py
                  (data → vectors → ChromaDB)


* Running Backend:
python3 app.py


* Running UI:
python3 -m streamlit run ui_app.py


* How It Works:
1. User enters a question. [form the given question text file]
2. System retrieves relevant chunks from algorithms.txt.
3. Chunks are passed as context into the small LLM.
4. LLM generates an accurate answer.


* Dataset Topics Include:
- Minimum Spanning Tree, Kruskal, Disjoint Set
- Bellman-Ford, Dijkstra, Floyd-Warshall
- BFS and DFS
- Counting Sort, Radix Sort, Bucket Sort
- NP, NP-Complete, NP-Hard, P vs NP
- Dynamic Programming vs Greedy Algorithms
- Backtracking and N-Queens
- Rabin-Karp, KMP, Huffman Coding

* Commands Summary:
1. cd ~/Documents/RAG.project
2. pip3 install -r requirements.txt
3. python3 ingest.py
4. python3 app.py
5. python3 -m streamlit run ui_app.py
