#### Building a Simple RAG Agent with LangChain, TinyLlama, and Qdrant

Retrieval-Augmented Generation (RAG) is a powerful paradigm that combines information retrieval and generative AI to answer questions grounded in external knowledge. In this post, we’ll explore how to build a simple RAG agent using LangChain, TinyLlama, and Qdrant, focused on answering questions about academic research papers from the arXiv dataset. This guide will walk you through the concepts and key steps without diving into the code itself.


#### Overview of Tools and Concepts

1. LangChain: LangChain provides a framework for building applications powered by language models. It simplifies integrating retrieval mechanisms, custom prompts, and model interactions.

2. TinyLlama: TinyLlama is a lightweight yet capable language model optimized for efficiency. It’s ideal for tasks where computational resources are a constraint.

3. Qdrant: Qdrant is a high-performance vector database used for storing and retrieving embeddings. It’s an excellent choice for implementing similarity search and powering RAG workflows.

4. RAG Architecture: A RAG agent consists of two main components:

      a. Retriever: Fetches relevant context from an external knowledge base (e.g., Qdrant).

      b. Generator: Uses the retrieved context to generate coherent, context-aware responses (e.g., TinyLlama via LangChain).


#### The Workflow

The complete code for the demo is [here](https://github.com/Kunal627/kunal627.github.io/blob/main/code/rag/demo1)

Before running the code, setup docker and bring up the docker containers for ollama, qdrant and UI. (Refer to previous posts in this series)

- Config.py

   You can play with the config.py file. Except, Don't change the values which are tagged don't change. 

- load_vdb.py

   This will load the embeddings in the vector db. 

   The script:

   1. Downloads papers of choice from arxiv database.
   2. Extracts the text from pdfs
   3. Preprocess the text, tokenizes and chunks it into smaller pieces
   4. Creates a collectionon Qdrant
   5. Generate the embeddings for chunks and inserts the embeddings with corresponding text chunk  

- testagent.ipynb
   
   A simple chain created with prompt and model. You can change the query and see how the model responds. 

   The context is retrieved from Qdrant by querrying the collection. The query embedding is passed to query_points function.
   The context from Qdrant is embedded in the prompt before invoking the chain.