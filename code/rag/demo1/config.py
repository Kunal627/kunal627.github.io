class Config:
    COLLECTION_NAME = "test_rag_agent" # Name of the collection in the Qdrant database
    HOSTNAME = "localhost"             # Hostname of the Qdrant server and ollama server, Do not change if running locally
    QDRANT_PORT = 6333                 # Port of the Qdrant server, Do not change
    SOURCE = "arxiv"                   # Source of the data, Don't change
    ARVIX_QUERY = "AI agents in healthcare"    # Query to search for papers on arXiv, can choose any topic
    ARVIX_MAX_RESULTS = 10               # Maximum number of papers to fetch from arXiv
    CHUNK_SIZE = 1800     # Number of words in a chunk, around 87% of context length for tinyllama (2048)
    CHUNK_OVERLAP = 360   # Number of words to overlap between chunks, around 20% of chunk size