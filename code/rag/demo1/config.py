class Config:
    COLLECTION_NAME = "arvix_papers_ai"
    HOSTNAME = "localhost"
    QDRANT_PORT = 6333
    SOURCE = "arxiv"
    ARVIX_QUERY = "AI agents"    # Query to search for papers on arXiv
    ARVIX_MAX_RESULTS = 10                # Maximum number of papers to fetch from arXiv