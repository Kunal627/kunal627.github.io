import warnings
warnings.filterwarnings("ignore")
import arxiv
import pymupdf
from qdrant_client.models import VectorParams, PointStruct
from config import Config
import requests
import re
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast

# Function to fetch papers from arXiv
def fetch_arxiv_papers(query=Config.ARVIX_QUERY, max_results=Config.ARVIX_MAX_RESULTS):
    """
    Fetch papers from arXiv using the arxiv Python library.

    Args:
    - query: The query string to search for papers on arXiv.
    - max_results: The maximum number of papers to fetch.

    Returns:
    - The Result objects yielded by Client.results include metadata about each paper and helper methods for downloading their content.
    """
    search = arxiv.Search(query=query, max_results=max_results)
    papers = search.results()
    return papers

# Function to parse and extract the title and abstract from the arXiv paper
def parse_arxiv_paper(paper):
    """
    Parse and extract the title, abstract, and PDF URL from an arXiv paper.

    Args:
    - paper: The arXiv paper object.
    
    Returns:
    - title: The title of the paper.
    - abstract: The abstract of the paper.
    - pdf_url: The URL to download the PDF of the paper.
    
    """
    title = paper.title
    abstract = paper.summary
    pdf_url = paper.pdf_url  # PDF URL for extracting content from the PDF
    return title, abstract, pdf_url

# Function to extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_url):
    """
    Extract text from a PDF using PyMuPDF (fitz).
    
    Args:
    - pdf_url: The URL to download the PDF.
    
    Returns:
    - The extracted text from the PDF.
    """

    # Open the PDF using PyMuPDF (fitz)
    print(f"Extracting text from PDF: {pdf_url}")
    r = requests.get(pdf_url)
    data = r.content
    pdf_doc = pymupdf.Document(stream=data)
    text = ""
    for page in pdf_doc:
        text += page.get_text()  # Extract text from each page
    return text

def chunk_text_by_length(text, chunk_size, overlap):
    """
    Chunk text into manageable sizes for TinyLlama or similar models.
    
    Args:
    - text: The input text to be chunked.
    - model_name: Name or path to the tokenizer (e.g., TinyLlama tokenizer).
    - chunk_size: Maximum number of tokens per chunk.
    - overlap: Number of overlapping tokens between chunks.
    
    Returns:
    - List of chunks (strings).
    """
    # Load the tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        # Create chunks with overlap
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks

def get_embeddings(model, chunks, llm_client):
    """
    Get embeddings for the chunks of text using the specified model.
    
    Args:
    - model: The model name or path.
    - chunks: List of text chunks.
    - llm_client: The LlamaClient object.
    
    Returns:
    - List of embeddings.
    - List of text chunks.
    """

    embed = []
    text = []
    for ch in chunks:
        embed.append(llm_client.embeddings(model=model, prompt=ch)['embedding'])
        text.append(ch)

    return embed, text


def create_qdrant_index(vdb_client, collection_name=Config.COLLECTION_NAME):
    """
    Create a collection in Qdrant for indexing the embeddings.
    
    Args:
    - vdb_client: The Qdrant client object.
    - collection_name: The name of the collection to be created.
    
    Returns:
    - None
    """

    collections = vdb_client.get_collections()
    existing_coll = [collection.name for collection in collections.collections]
    print(f"Existing collections {existing_coll}")
    if collection_name not in existing_coll:
        print("creating collection::", Config.COLLECTION_NAME)
        vdb_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2048, distance="Cosine")  # tinyllama embed dimension
        )


# Function to index the chunks and their embeddings into Qdrant
def index_chunks_in_qdrant(chunks, embeddings, vdb_client, collection_name=Config.COLLECTION_NAME):
    """
    Index the chunks and their embeddings into Qdrant.
    
    Args:
    - chunks: List of text chunks.
    - embeddings: List of embeddings corresponding to the chunks.
    - vdb_client: The Qdrant client object.
    - collection_name: The name of the collection in Qdrant.
    
    Returns:
    - None
    """
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk}  # The chunk of text as metadata
        )
        points.append(point)
    
    # Upsert the points in Qdrant
    vdb_client.upsert(collection_name=collection_name, points=points)


def retrieve_context(query_embed, vdb_client):
    """
    Retrieve the context from Qdrant using the query embedding.
    
    Args:
    - query_embed: The query embedding.
    - vdb_client: The Qdrant client object.
    
    Returns:
    - List of hits (points) from Qdrant.
    """
    #print("query collection", Config.COLLECTION_NAME)
    hits = vdb_client.query_points(
        collection_name=Config.COLLECTION_NAME,
        query=query_embed,
        limit=10).points

    return hits

# Get the count of vectors in the collection
def get_vector_count(collection_name: str, vdb_client) -> int:
    count_response = vdb_client.count(collection_name=collection_name)
    return count_response.count

def preprocess(text):
    """
    Preprocess the text by removing URLs, references, and converting to lowercase."""
    patterns = [
        r"\bReferences\b",  # Matches 'References' as a standalone word
        r"\bBibliography\b",  # Matches 'Bibliography' as a standalone word
    ]
    # Combine patterns and locate the start of the references section
    pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        # Remove everything from the start of the references section
        text = text[:match.start()]
    text = text.lower()
    text = text.replace("\n", " ")
    text = ' '.join(text.split())
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    reference_pattern = r"\[\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+)*\]"
    # Replace URLs with an empty string
    text = re.sub(url_pattern, '', text)
    text = re.sub(reference_pattern, '', text)

    return text
