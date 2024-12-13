class Config:
    COLLECTION_NAME = "test_rag_agent"
    HOSTNAME = "localhost"
    QDRANT_PORT = 6333
    SOURCE = "arxiv"
    ARVIX_QUERY = "AI agents in healthcare"    # Query to search for papers on arXiv
    ARVIX_MAX_RESULTS = 1               # Maximum number of papers to fetch from arXiv
    CHUNK_SIZE = 1800     # around 87% of context length
    CHUNK_OVERLAP = 360   # around 20% 