# Building RAG Agents with LangChain and LLMs locally

Welcome to a series of blog posts where we explore **LangChain** and **RAG (Retrieval-Augmented Generation)** agents. In this series, you’ll learn how to set up a local LLM (Large Language Model), use LangChain for effective prompt management, and build powerful RAG agents for natural language processing tasks.

## Table of Contents

1. [Introduction to LangChain and RAG](#introduction-to-langchain-and-rag)
2. [Setting Up a Local LLM](#setting-up-a-local-llm)
3. [Simple Prompt and model chain](#simple-prompt-and-model-chain)
4. [Setup Vector DB](#setup-vector-db)
5. [Building a Basic RAG Agent using arxiv papers](#building-a-basic-rag-agent)

---

## Introduction to LangChain and RAG

I will not write a post on this as there are lots of resources available on web. [langchain](https://python.langchain.com/docs/introduction/)

---

## Setting Up a Local LLM

In this post, we'll walk through the process of setting up a local LLM using Docker and using it for LangChain-based applications. We’ll use **Ollama** for this demonstration, but the steps are similar for other LLMs.

### Key Steps:
- Installing Docker and setting up the LLM container.
- Interacting with LLM via API.
- Testing your local LLM for basic text generation tasks.

**Python file**: [llamma_setup](https://github.com/Kunal627/kunal627.github.io/blob/main/code/rag/llamma_setup.ipynb)

---

## Simple Prompt and model chain

Learn how to create a custom LLM wrapper for TinyLlama using LangChain, build a prompt and LLM chain, and run it locally

### Key Steps:
- Custom LLM wrapper implementation
- prompt the tinyllama model and get a response

**Python file**: [tinyllama_docker](https://github.com/Kunal627/kunal627.github.io/blob/main/code/rag/tinyllama_docker.ipynb)


## Setup Vector DB

Learn how to setup Qdrant (vector databse) locally with docker

### Key Steps:
- Update the docker compose file to bring up qdrant container in docker
- test Qdrant setup by running some examples 
- All the examples are taken (from)[https://qdrant.tech/documentation/beginner-tutorials/]

**Python file**: [vecdb_setup](https://github.com/Kunal627/kunal627.github.io/blob/main/code/rag/vecdb_setup.ipynb)


## Building a Basic RAG Agent using arxiv papers

Build a simple RAG agent using Lang Chain to anwswer questions on arxiv papers

### Key Steps:
- Download papers from arxiv 
- Extract text from pdfs, preprocess and chunk the documents
- Generate embeddings for the chunks and insert into Vector db
- Invoke a tinyllama chain with a prompt

**Python file**: [arxiv.ipynb](https://github.com/Kunal627/kunal627.github.io/blob/main/code/rag/demo1/readme.md)