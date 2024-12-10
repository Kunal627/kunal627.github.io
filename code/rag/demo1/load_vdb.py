import warnings
warnings.filterwarnings("ignore")
import arxiv
import pymupdf
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import os
import requests
import ollama


class Config:
    COLLECTION_NAME = "arvix_papers"
    HOSTNAME = "localhost"
    QDRANT_PORT = 6333
    SOURCE = "arxiv"
    ARVIX_QUERY = "quantum computing"    # Query to search for papers on arXiv
    ARVIX_MAX_RESULTS = 1                # Maximum number of papers to fetch from arXiv
    

# Initialize Ollama client
oclient = ollama.Client(host=Config.HOSTNAME)

# Initialize Qdrant client
qclient = QdrantClient(host=Config.HOSTNAME, port=Config.QDRANT_PORT)


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

def get_embeddings(model, chunks):

    embed = []
    text = []
    for ch in chunks:
        embed.append(oclient.embeddings(model=model, prompt=ch)['embedding'])
        text.append(ch)

    return embed, text


def create_qdrant_index(collection_name=Config.COLLECTION_NAME):
    collections = qclient.get_collections()
    existing_coll = [collection.name for collection in collections.collections]
    print(f"Existing collections {existing_coll}")
    if collection_name not in existing_coll:
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2048, distance="Cosine")  # tinyllama embed dimension
        )


# Function to index the chunks and their embeddings into Qdrant
def index_chunks_in_qdrant(chunks, embeddings, collection_name=Config.COLLECTION_NAME):
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk}  # The chunk of text as metadata
        )
        points.append(point)
    
    # Upsert the points in Qdrant
    qclient.upsert(collection_name=collection_name, points=points)



def main():
    if Config.SOURCE == "arxiv":
        papers = fetch_arxiv_papers(query="quantum computing", max_results=1)
    else:
        raise ValueError("Invalid source. Please specify 'arxiv' as the source.")

    for paper in papers:
        print(f"Processing paper {paper.title}")

        final_embed = []
        final_text = []

        title, abstract, url = parse_arxiv_paper(paper)
        print(f"Extracting {title}")

        pdf_text = extract_text_from_pdf(url)

        full_text = abstract + "\n\n" + pdf_text

        print(f"Chunking text into smaller pieces.")
        chunks = chunk_text_by_length(full_text, 500)

        embeds, text = get_embeddings(model="tinyllama", chunks=chunks)
        print(embeds[0], text[0])

        # list concatenation
        final_embed += embeds
        final_text += text

        # create collection
    create_qdrant_index()

    index_chunks_in_qdrant(final_text[0:2], final_embed[0:2])
    print("Indexing complete")


if __name__ == "__main__":
    main()
