import warnings
warnings.filterwarnings("ignore")
from qdrant_client import QdrantClient
from config import Config
import ollama
from common import *


# Initialize Ollama client
oclient = ollama.Client(host=Config.HOSTNAME)

# Initialize Qdrant client
qclient = QdrantClient(host=Config.HOSTNAME, port=Config.QDRANT_PORT)



def main():
    if Config.SOURCE == "arxiv":
        papers = fetch_arxiv_papers(query=Config.ARVIX_QUERY, max_results=Config.ARVIX_MAX_RESULTS)
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

        embeds, text = get_embeddings(model="tinyllama", chunks=chunks, llm_client=oclient)
        #print(embeds[0], text[0])

        # list concatenation
        final_embed += embeds
        final_text += text

        # create collection
    create_qdrant_index(vdb_client=qclient)

    index_chunks_in_qdrant(final_text, final_embed, vdb_client=qclient)
    print("Indexing complete")


if __name__ == "__main__":
    main()
