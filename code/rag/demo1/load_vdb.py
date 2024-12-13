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
    
    final_embed = []
    final_text = []

    for paper in papers:
        print("===============================")
        print(f"Processing paper {paper.title}")
        print("===============================")

        title, abstract, url = parse_arxiv_paper(paper)
        print(f"Extracting {title}")

        pdf_text = extract_text_from_pdf(url)
        #print(pdf_text)

        print("After preprocessing")
        pdf_text = preprocess(pdf_text)
        #print(pdf_text)

        full_text = abstract + " " + pdf_text

        print(f"Chunking text into smaller pieces.")
        chunks = chunk_text_by_length(full_text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        print(">>>>>>> printing chunks  >>>>>>>>>>>>>>>>>>>>>")
        for i, c in enumerate(chunks):
            print(f"############# chunk {i}")
            print(c)

        embeds, text = get_embeddings(model="tinyllama", chunks=chunks, llm_client=oclient)
        print(">>>>>>>>>>>", len(embeds), len(text))

        # list concatenation
        final_embed += embeds
        final_text += text

        print("==============================================")
        print(f"processed paper {paper.title}")
        print("==============================================")
        print("###########", len(final_embed), len(final_text))
        # create collection
    print("Number of chunks to index", len(final_embed), len(final_text))
    create_qdrant_index(vdb_client=qclient, collection_name=Config.COLLECTION_NAME)

    index_chunks_in_qdrant(final_text, final_embed, vdb_client=qclient)
    print("Indexing complete")

    print("Total vectors inserted", get_vector_count(Config.COLLECTION_NAME, qclient))

if __name__ == "__main__":
    main()
