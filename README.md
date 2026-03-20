
Mini RAG Chatbot
Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot.
The chatbot answers user queries using information from provided documents instead of relying on general knowledge.
It ensures that responses are based only on the given data, improving accuracy and reliability.

How the System Works

The process can be broken down into three main stages: preparing the documents, finding relevant information, and generating the final answer.

Step 1: Load Documents
The system reads PDF files from the data folder.

Step 2: Chunking
The text is split into smaller parts (chunks).This makes it easier to search and process.

Step 3: Embedding
Each chunk is converted into a vector using the embedding model.

Step 4: Storage
All vectors are stored in a FAISS index.

Step 5: Retrieval
When a user asks a question:
The question is also converted into a vector.
The system finds the most similar chunks.
These chunks are used as context. It does not use any outside information. 

Project Structure

Build-a-Mini-RAG/
── app.py ── rag_pipeline.py ── requirements.txt ── data/

Models Used

Embedding Model:
all-MiniLM-L6-v2 (Sentence Transformers)
We use all-MiniLM-L6-v2.
It helps convert text into numbers so the system can understand meaning.
It is fast and works well for finding similar text

Language Model:
gpt-4o-mini (OpenAI)
We use gpt-4o-mini.
It is used to generate the final answer.
It is efficient and gives good results when given proper context

Grounded Answer Generation
Answers are created only from the retrieved documents
No outside knowledge is used
If the answer is not in the documents → it says “Not found”
This avoids wrong or made-up answers

Transparency
The system shows:
The retrieved document parts (context)
The final answer
So users can check how the answer was generated

Conclusion
This project demonstrates a complete RAG pipeline that combines document retrieval with controlled language model generation to produce accurate and explainable answers.
