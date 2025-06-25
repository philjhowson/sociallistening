from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import shared_functions
import torch
import pandas as pd

path_to_processed = 'data/processed'
path_to_RAG = 'data/RAG'

def build_database():
    """
    Loads in the data and splits the text for document chunking. I used
    RecursiveCharacterTextSplitter() because reddit often has very long posts
    and it needs to be chunked in a way that the encoder won't truncate it.
    It's also advantageous because often reddit posts are long, but only a
    few sentences are of interest.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    documents = []

    print('Chunking comments for archive...')
    """
    This takes the chunks and cleans the text appropriately. I did this so I can
    accurately extract which original comment refers to which document and keep
    track of what the LLM actually receives as its input. For each row, relevant
    information, such as 'text', 'date', and 'source' which will be used for metadata.
    """

    for index, (idx, row) in enumerate(data.iterrows()):

        if index % 1000 == 0:
            print(f"Processing row {index}/{len(data)}...")

        text = row['text']
        date = row['date']
        source = row['source']
        cleaned_text = row['cleaned_text']
        chunks = splitter.split_text(text)

        for chunk in chunks:

            cleaned_chunk = shared_functions.pipeline(chunk, custom_stops, whitelist)

            documents.append(Document(page_content = cleaned_chunk,
                                      metadata = {'index' : index,
                                                  'raw_index' : idx,
                                                  'text' : chunk,
                                                  'date' : date,
                                                  'source' : source}))

    """
    Embeddings are generated and the FAISS archive is built and saved.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_function = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                                               model_kwargs = {'device': device})

    print('Building FAISS Archive...')
    faiss_archive = FAISS.from_documents(documents, embedding_function)
    faiss_archive.save_local(f"{path_to_RAG}/complete_faiss_archive")

    

    print('Archive successfully built and saved!')

if __name__ == '__main__':
    build_database()