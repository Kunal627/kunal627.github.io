import warnings
warnings.filterwarnings("ignore")
import arxiv
import pymupdf
from qdrant_client.models import VectorParams, PointStruct
from config import Config
import requests

# Function to fetch papers from arXiv
def fetch_arxiv_papers(query=Config.ARVIX_QUERY, max_results=Config.ARVIX_MAX_RESULTS):
    search = arxiv.Search(query=query, max_results=max_results)
    papers = search.results()
    return papers

# Function to parse and extract the title and abstract from the arXiv paper
def parse_arxiv_paper(paper):
    title = paper.title
    abstract = paper.summary
    pdf_url = paper.pdf_url  # PDF URL for extracting content from the PDF
    return title, abstract, pdf_url

# Function to extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_url):
    # Open the PDF using PyMuPDF (fitz)
    print(f"Extracting text from PDF: {pdf_url}")
    r = requests.get(pdf_url)
    data = r.content
    pdf_doc = pymupdf.Document(stream=data)
    text = ""
    for page in pdf_doc:
        text += page.get_text()  # Extract text from each page
    return text

def chunk_text_by_length(text, chunk_size):
    """
    Splits the given text into smaller chunks of a specified character length.

    Args:
    - text (str): The input text to chunk.
    - chunk_size (int): The maximum number of characters per chunk.

    Returns:
    - list of str: A list of text chunks.
    """
    # Split text into chunks of the specified size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks

def get_embeddings(model, chunks, llm_client):

    embed = []
    text = []
    for ch in chunks:
        embed.append(llm_client.embeddings(model=model, prompt=ch)['embedding'])
        text.append(ch)

    return embed, text


def create_qdrant_index(vdb_client, collection_name=Config.COLLECTION_NAME):
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

