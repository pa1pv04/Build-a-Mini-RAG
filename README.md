
Mini RAG Chatbot
Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot.
The chatbot answers user queries using information from provided documents instead of relying on general knowledge.

It ensures that responses are based only on the given data, improving accuracy and reliability.

How the System Works

The process can be broken down into three main stages: preparing the documents, finding relevant information, and generating the final answer.

Stage 1: Document Preparation
Gather Documents: The system first finds all the PDF files located in a specific "data folder".
Read the Text: It reads the text content from each of these PDF documents.
Break into Chunks: The large amount of text is then split into smaller, manageable pieces (chunks) so they can be processed efficiently.
Convert to Numbers (Embeddings): Each text chunk is turned into a special numerical code called a "vector" using a sentence transformer model. Think of this as translating words into a language computers understand.
Organize for Search (Indexing): All these numerical codes (vectors) are stored in a special, fast database called a FAISS index. This makes finding specific information later much quicker. 

Stage 2: Finding the Answer (Retrieval)
User Asks a Question: When you ask a question, your question is also converted into a numerical code (vector).
Search the Database: The system then looks through its FAISS index to find the text chunks whose numerical codes are most similar to your question's code. This is like finding the most relevant paragraphs in a huge library.

Stage 3: Generating the Answer
Provide Context: The few most relevant text chunks found in the previous step are given to a Large Language Model (LLM) as "context".
Generate Answer: The LLM uses only the provided context to create a final, concise, and accurate answer to your question. It does not use any outside information. 

Project Structure

Build-a-Mini-RAG/
── app.py ── rag_pipeline.py ── requirements.txt ── data/

Models Used

Embedding Model:
all-MiniLM-L6-v2 (Sentence Transformers)

Language Model:
gpt-4o-mini (OpenAI)

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
